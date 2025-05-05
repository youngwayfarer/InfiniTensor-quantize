#include "operators/conv.h"
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "oneapi/dnnl/dnnl.hpp"

namespace infini {

class NaiveConv : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<ConvObj>(_op);
        T *iptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *wptr = op->getInputs(1)->getRawDataPtr<T *>();
        T *optr = op->getOutput()->getRawDataPtr<T *>();
        //  Clang will give an error of " reference to local binding 'sh'
        //  declared in enclosing function" if we write like this:
        //        auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        int n, c, h, w, f, r, s;
        std::tie(n, c, h, w, f, r, s) = op->getNCHWFRS();
        int ph, pw, sh, sw, dh, dw;
        std::tie(ph, pw, sh, sw, dh, dw) = op->getPadStrideDilation();
        int cpg = op->getChannelPerGroup();
        // g = 1
        int g = op->getNumGroups();
        IT_ASSERT(f % g == 0, "Illegal number of channel");
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];
        for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
            for (int ff = 0; ff < f; ff++) {
                for (int hh = 0; hh < oh; hh++)
                    for (int ww = 0; ww < ow; ww++) {
                        int gidx = ff / (f / g);
                        T val = 0;
                        for (int cc = 0; cc < cpg; cc++)
                            for (int rr = 0; rr < r; rr++)
                                for (int ss = 0; ss < s; ss++) {
                                    // clang-format off
                                int posH = hh * sh + rr * dh - ph;
                                int posW = ww * sw + ss * dw - pw;
                                if (posH < 0 || posH >= h || posW < 0 || posW >= w)
                                    continue;
                                auto iOffset = posW + w * (posH + h * ((cc + gidx * cpg) + c * nn)),
                                    wOffset = ss + s * (rr + r * (cc + cpg * ff));
                                auto inputVal = iptr[iOffset], weightVal = wptr[wOffset];
                                val += weightVal * inputVal;
                                    // clang-format on
                                }
                        // TODO: check correctness, oh & ow or h & w?
                        auto oOffset = ww + ow * (hh + oh * (ff + f * nn));
                        optr[oOffset] = val;
                    }
            }
        }
    }

    // for int8 input and weight, int32 output
    void doComputeInt8ToInt32(const Operator &_op,
                              const RuntimeObj *context) const {
        auto op = as<ConvObj>(_op);
        int8_t *iptr = op->getInputs(0)->getRawDataPtr<int8_t *>();
        int8_t *wptr = op->getInputs(1)->getRawDataPtr<int8_t *>();
        int32_t *optr = op->getOutput()->getRawDataPtr<int32_t *>();
        int n, c, h, w, f, r, s;
        std::tie(n, c, h, w, f, r, s) = op->getNCHWFRS();
        int ph, pw, sh, sw, dh, dw;
        std::tie(ph, pw, sh, sw, dh, dw) = op->getPadStrideDilation();
        int cpg = op->getChannelPerGroup();
        int g = op->getNumGroups();
        IT_ASSERT(f % g == 0, "Illegal number of channel");
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];
        for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
            for (int ff = 0; ff < f; ff++) {
                for (int hh = 0; hh < oh; hh++)
                    for (int ww = 0; ww < ow; ww++) {
                        int gidx = ff / (f / g);
                        int32_t val = 0;
                        for (int cc = 0; cc < cpg; cc++)
                            for (int rr = 0; rr < r; rr++)
                                for (int ss = 0; ss < s; ss++) {
                                    int posH = hh * sh + rr * dh - ph;
                                    int posW = ww * sw + ss * dw - pw;
                                    if (posH < 0 || posH >= h || posW < 0 ||
                                        posW >= w)
                                        continue;
                                    auto iOffset = posW +
                                                   w * (posH +
                                                        h * ((cc + gidx * cpg) +
                                                             c * nn)),
                                         wOffset =
                                             ss +
                                             s * (rr + r * (cc + cpg * ff));
                                    int32_t inputVal =
                                        static_cast<int32_t>(iptr[iOffset]);
                                    int32_t weightVal =
                                        static_cast<int32_t>(wptr[wOffset]);
                                    val += weightVal * inputVal;
                                }
                        auto oOffset = ww + ow * (hh + oh * (ff + f * nn));
                        optr[oOffset] = val;
                    }
            }
        }
    }

    // onednn
    void doComputeInt8OneDnn(const Operator &_op,
                             const RuntimeObj *context) const {
        auto op = as<ConvObj>(_op);
        int8_t *iptr = op->getInputs(0)->getRawDataPtr<int8_t *>();
        int8_t *wptr = op->getInputs(1)->getRawDataPtr<int8_t *>();
        int32_t *optr = op->getOutput()->getRawDataPtr<int32_t *>();

        int n, c, h, w, f, r, s;
        std::tie(n, c, h, w, f, r, s) = op->getNCHWFRS();
        int ph, pw, sh, sw, dh, dw;
        std::tie(ph, pw, sh, sw, dh, dw) = op->getPadStrideDilation();
        int cpg = op->getChannelPerGroup();
        // int g = op->getNumGroups();
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];

        try {
            // 创建OneDNN引擎和流
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream stream(engine);

            // 创建用户内存描述符，描述实际数据布局
            auto userSrcMd =
                dnnl::memory::desc({n, c, h, w}, dnnl::memory::data_type::s8,
                                   dnnl::memory::format_tag::nchw);
            auto userWMd =
                dnnl::memory::desc({f, cpg, r, s}, dnnl::memory::data_type::s8,
                                   dnnl::memory::format_tag::oihw);
            auto userDstMd =
                dnnl::memory::desc({n, f, oh, ow}, dnnl::memory::data_type::s32,
                                   dnnl::memory::format_tag::nchw);

            // 创建用户内存对象
            auto userSrcMemory = dnnl::memory(userSrcMd, engine, iptr);
            auto userWMemory = dnnl::memory(userWMd, engine, wptr);

            // 创建format_tag::any的内存描述符，让OneDNN选择最优内存格式
            auto srcMd =
                dnnl::memory::desc({n, c, h, w}, dnnl::memory::data_type::s8,
                                   dnnl::memory::format_tag::any);
            auto wMd =
                dnnl::memory::desc({f, cpg, r, s}, dnnl::memory::data_type::s8,
                                   dnnl::memory::format_tag::any);
            auto dstMd =
                dnnl::memory::desc({n, f, oh, ow}, dnnl::memory::data_type::s32,
                                   dnnl::memory::format_tag::any);

            // 设置卷积参数
            dnnl::memory::dims strides = {sh, sw};
            dnnl::memory::dims pads = {ph, pw};
            dnnl::memory::dims dilations = {dh - 1, dw - 1};

            // 创建卷积描述符
            // auto convDesc = dnnl::convolution_forward::desc(
            //     dnnl::prop_kind::forward_inference,
            //     dnnl::algorithm::convolution_direct, srcMd, wMd, dstMd,
            //     strides, dilations, pads, pads);

            // 创建卷积原语描述符
            // auto primDesc =
            //     dnnl::convolution_forward::primitive_desc(convDesc, engine);

            auto primDesc = dnnl::convolution_forward::primitive_desc(
                engine, dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_auto, srcMd, wMd, dstMd, strides,
                dilations, pads, pads);

            // 如果需要，重排输入数据
            auto srcMemory = userSrcMemory;
            if (primDesc.src_desc() != userSrcMemory.get_desc()) {
                srcMemory = dnnl::memory(primDesc.src_desc(), engine);
                dnnl::reorder(userSrcMemory, srcMemory)
                    .execute(stream, {{DNNL_ARG_FROM, userSrcMemory},
                                      {DNNL_ARG_TO, srcMemory}});
            }

            // 如果需要，重排权重数据
            auto wMemory = userWMemory;
            if (primDesc.weights_desc() != userWMemory.get_desc()) {
                wMemory = dnnl::memory(primDesc.weights_desc(), engine);
                dnnl::reorder(userWMemory, wMemory)
                    .execute(stream, {{DNNL_ARG_FROM, userWMemory},
                                      {DNNL_ARG_TO, wMemory}});
            }

            // 处理输出内存
            if (primDesc.dst_desc() == userDstMd) {
                // 输出格式匹配，直接使用用户内存
                auto dstMemory = dnnl::memory(userDstMd, engine, optr);

                // 执行卷积
                dnnl::convolution_forward(primDesc).execute(
                    stream, {{DNNL_ARG_SRC, srcMemory},
                             {DNNL_ARG_WEIGHTS, wMemory},
                             {DNNL_ARG_DST, dstMemory}});
            } else {
                // 输出格式不匹配，需要重排
                auto dstMemory = dnnl::memory(primDesc.dst_desc(), engine);

                // 执行卷积
                dnnl::convolution_forward(primDesc).execute(
                    stream, {{DNNL_ARG_SRC, srcMemory},
                             {DNNL_ARG_WEIGHTS, wMemory},
                             {DNNL_ARG_DST, dstMemory}});

                // 将结果重排回用户格式
                auto userDstMemory = dnnl::memory(userDstMd, engine, optr);
                dnnl::reorder(dstMemory, userDstMemory)
                    .execute(stream, {{DNNL_ARG_FROM, dstMemory},
                                      {DNNL_ARG_TO, userDstMemory}});
            }

            // 等待执行完成
            stream.wait();
        } catch (const dnnl::error &e) {
            IT_ASSERT(false, "OneDNN error: " + std::string(e.what()));
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        if (_op->getDType() == DataType::Int8) {
            // 尝试使用OneDNN实现，如果失败则回退到朴素实现
            try {
                doComputeInt8OneDnn(_op, context);
                // std::cout << "OneDNN Conv executed successfully." << std::endl;
                return;
            } catch (...) {
                doComputeInt8ToInt32(_op, context);
                // std::cout << "OneDNN Conv failed, falling back to naive "
                //              "implementation."
                //           << std::endl;
                return;
            }
        }

#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Conv, NaiveConv, "ConvNaive_CPU");

} // namespace infini
