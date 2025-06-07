#include "operators/conv.h"
#include "core/common.h"
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>

namespace infini {

struct ConvCuDnnPerfRecordObj : public PerfRecordObj {
    int algo = 0; // cudnnConvolutionFwdAlgo_t
    int mode = 1;
    // size_t workspaceSize = 100000;
    size_t workspaceSize = 1000;
    bool fuseAct = false;
    void to_json(json &j) override {
        j["type"] = 1;
        j["data"] = std::make_tuple(algo, mode, fuseAct, time, workspaceSize);
    }
    static PerfRecord from_json(const json &j) {
        ConvCuDnnPerfRecordObj tmp;
        auto [Algo, Mode, FuseAct, Time, WorkspaceSize] =
            j["data"].get<tuple<int, int, bool, double, size_t>>();
        tmp.algo = Algo;
        tmp.mode = Mode;
        tmp.fuseAct = FuseAct;
        tmp.time = Time;
        tmp.workspaceSize = WorkspaceSize;
        return make_ref<ConvCuDnnPerfRecordObj>(tmp);
    }
};

using ConvCuDnnPerfRecord = Ref<ConvCuDnnPerfRecordObj>;

class convCudnn : public Kernel {

    static constexpr int N_ALGO = 8;
    static constexpr int N_MODE = 2;
    static constexpr cudnnConvolutionFwdAlgo_t ALGOS[8] = {
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

    static constexpr cudnnConvolutionMode_t MODES[2] = {
        CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION};

    std::tuple<void *, void *, void *, cudnnTensorDescriptor_t,
               cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
               cudnnConvolutionDescriptor_t, cudnnActivationDescriptor_t,
               cudnnTensorDescriptor_t>
    createCuDNNDescriptor(const Ref<ConvObj> &op,
                          const ConvCuDnnPerfRecord &record) const {
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        // Bias is not supported yet
        if (op->getInputs().size() > 2) {
            IT_TODO_HALT();
        }
        // int8 -> 3
        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());

        cudnnDataType_t computeType = cudnnDataType;
        if (cudnnDataType == CUDNN_DATA_INT8) {
            // std::cout << "cudnnDataType is CUDNN_DATA_INT8" << std::endl;
            computeType = CUDNN_DATA_INT32;
        }

        // void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        int channelsPerGrp = cpg, channels = c;

        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        if (cudnnDataType == CUDNN_DATA_INT8) {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                inDesc, CUDNN_TENSOR_NHWC, cudnnDataType, n, channels, h,
                w));
        } else {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                inDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, channels, h,
                w));
        }

        // checkCudnnError(cudnnSetTensor4dDescriptor(
        //     inDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, channels, h, w));

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        if (cudnnDataType == CUDNN_DATA_INT8) {
            checkCudnnError(cudnnSetFilter4dDescriptor(
                knDesc, cudnnDataType, CUDNN_TENSOR_NHWC, f, channelsPerGrp,
                r, s));
        } else {
            checkCudnnError(cudnnSetFilter4dDescriptor(
                knDesc, cudnnDataType, CUDNN_TENSOR_NCHW, f, channelsPerGrp,
                r, s));
        }

        // checkCudnnError(cudnnSetFilter4dDescriptor(
        //     knDesc, cudnnDataType, CUDNN_TENSOR_NCHW, f, channelsPerGrp, r, s));

        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        if (cudnnDataType == CUDNN_DATA_INT8) {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                biasDesc, CUDNN_TENSOR_NHWC, cudnnDataType, 1, f, 1, 1));
        } else {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                biasDesc, CUDNN_TENSOR_NCHW, cudnnDataType, 1, f, 1, 1));
        }

        // checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
        //                                            cudnnDataType, 1, f, 1, 1));

        // get convolution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        // checkCudnnError(cudnnSetConvolution2dDescriptor(
        //     convDesc, ph, pw, sh, sw, dh, dw, MODES[record->mode],
        //     cudnnDataType));
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw,
            MODES[record->mode], // 卷积模式 (Convolution / Cross-correlation)
            computeType));       // *** 使用计算类型 ***

        // bool flag = false;
        // if (op->getDType() == DataType::Float16) {
        //     if (op->getInputs(0)->getDims()[1] % 8 == 0 &&
        //         op->getInputs(1)->getDims()[0] % 8 == 0) {
        //         flag = true;
        //     }
        // }
        // if ((cudnnDataType == CUDNN_DATA_INT8) || flag) {
        //     checkCudnnError(
        //         cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
        // } else if (cudnnDataType == CUDNN_DATA_FLOAT) {
        //     checkCudnnError(
        //         cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH));
        // }

        checkCudnnError(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
        if (op->getDType() == DataType::Float32) {
            checkCudnnError(cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH));
        }

        if (g > 1) {
            checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
        }

        // get activation descriptor
        cudnnActivationDescriptor_t actDesc;
        checkCudnnError(cudnnCreateActivationDescriptor(&actDesc));
        // NOT_PROPAGATE_NAN is requierd by
        // cudnnConvolotionBiasActivationForward
        switch (op->getAct()) {
        case ActType::Relu:
            checkCudnnError(cudnnSetActivationDescriptor(
                actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case ActType::Sigmoid:
            checkCudnnError(cudnnSetActivationDescriptor(
                actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case ActType::None:
            checkCudnnError(
                cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_IDENTITY,
                                             CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        default:
            assert(false);
        }

        // get output descriptor
        int outn, outc, outh, outw;
        checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
            convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        if (cudnnDataType == CUDNN_DATA_INT8) {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                outDesc, CUDNN_TENSOR_NHWC, computeType, outn, outc, outh,
                outw));
        } else {
            checkCudnnError(cudnnSetTensor4dDescriptor(
                outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, outn, outc, outh,
                outw));
        }
        // checkCudnnError(cudnnSetTensor4dDescriptor(
        //     outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, outn, outc, outh,
        //     outw));

        if (cudnnDataType == CUDNN_DATA_INT8) {
            // NHWC
            IT_ASSERT((vector{outn, outh, outw, outc}) ==
                          op->getOutput()->getDims(),
                      "cuDNN output shape mismatches with OP output shape");
        } else {
            // NCHW
            IT_ASSERT((vector{outn, outc, outh, outw}) ==
                          op->getOutput()->getDims(),
                      "cuDNN output shape mismatches with OP output shape");
        }

        // IT_ASSERT((vector{outn, outc, outh, outw}) ==
        //               op->getOutput()->getDims(),
        //           "cuDNN output shape mismatches with OP output shape");

        return tuple(inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc);
    }

    bool cuDNNUnfused(const Ref<ConvObj> &op, const ConvCuDnnPerfRecord &record,
                      const CudaRuntimeObj *context) const {
        cudnnStatus_t stat;

        const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc] =
            createCuDNNDescriptor(op, record);

        if (op->getDType() == DataType::Int8) {
            record->algo = 2;
        }
        if (op->getDType() == DataType::Float16 && op->getIntQuantized()) {
            record->algo = 2;
            record->workspaceSize = 5000000;
        }

        // * test(conv_pass)
        // if (op->getDType() == DataType::Float16 ||
        //     op->getDType() == DataType::Int8) {
        //     if (op->getConvPass()) {
        //         int returned = 0;
        //         cudnnConvolutionFwdAlgoPerf_t perf[N_ALGO];
        //         checkCudnnError(cudnnGetConvolutionForwardAlgorithm_v7(
        //             context->cudnnHandle(), inDesc, knDesc, convDesc,
        //             outDesc, N_ALGO, &returned, perf));
        //         for (int i = 0; i < returned; ++i) {
        //             if (perf[i].status == CUDNN_STATUS_SUCCESS &&
        //                 perf[i].memory <= context->getWorkspaceSize()) {
        //                 // 找到最优算法，更新 record
        //                 int best_algo = static_cast<int>(std::distance(
        //                     ALGOS,
        //                     std::find(ALGOS, ALGOS + N_ALGO, perf[i].algo)));
        //                 if (best_algo != N_ALGO) {
        //                     record->algo = best_algo;
        //                     record->workspaceSize = perf[i].memory;
        //                     op->setAlgo(best_algo);
        //                     op->setWsize(perf[i].memory);
        //                     // std::cout
        //                     //     << "in
        //                     cudnnGetConvolutionForwardAlgorithm_v7: "
        //                     //     << op->getDType().toString() << std::endl;
        //                     // std::cout << "in conv, best algo is " <<
        //                     best_algo
        //                     //           << std::endl;
        //                     // std::cout << "workspace size is "
        //                     //           << record->workspaceSize <<
        //                     std::endl;
        //                 }
        //                 break;
        //             }
        //         }
        //     } else {
        //         record->algo = op->getAlgo();
        //         record->workspaceSize = op->getWsize();
        //     }
        // }

        size_t wsSize = record->workspaceSize;
        CudaPtr wsData = context->getWorkspace(wsSize);

        if (op->getDType() == DataType::Int8) {
            //! important
            int32_t alpha = 1;
            int32_t beta = 0;
            stat = cudnnConvolutionForward(
                context->cudnnHandle(), &alpha, inDesc, inData, knDesc, knData,
                convDesc, ALGOS[record->algo], wsData, wsSize, &beta, outDesc,
                outData);
        } else {
            float alpha = 1.f, beta = 0.f;
            stat = cudnnConvolutionForward(
                context->cudnnHandle(), &alpha, inDesc, inData, knDesc, knData,
                convDesc, ALGOS[record->algo], wsData, wsSize, &beta, outDesc,
                outData);
        }

        // float alpha = 1.f, beta = 0.f;
        // stat = cudnnConvolutionForward(context->cudnnHandle(), &alpha, inDesc,
        //                                inData, knDesc, knData, convDesc,
        //                                ALGOS[record->algo], wsData, wsSize,
        //                                &beta, outDesc, outData);
        if (stat != CUDNN_STATUS_SUCCESS) {
            return false;
        }
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
        checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
        checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        return true;
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record = make_ref<ConvCuDnnPerfRecordObj>(); // with paramters in
                                                          // default ctor
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        ConvCuDnnPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<ConvObj>(_op);
        // Both modes have the same performance. Only run cross-correlation.
        for (int mode = 1; mode < 2; mode++) {
            // Try every possible algorithm of convolution
            for (int algo = 0; algo < N_ALGO; algo++) {
                auto recordRef = make_ref<ConvCuDnnPerfRecordObj>();
                auto &record = *recordRef;
                record.mode = mode;
                record.algo = algo;
                cudnnStatus_t stat;
                const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                             convDesc, actDesc, outDesc] =
                    createCuDNNDescriptor(op, recordRef);

                // get workspace
                stat = cudnnGetConvolutionForwardWorkspaceSize(
                    context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                    ALGOS[record.algo], &record.workspaceSize);
                if (stat != CUDNN_STATUS_SUCCESS) {
                    continue;
                }
                if (record.workspaceSize > context->getWorkspaceSize()) {
                    continue;
                }
                CudaPtr wsData = context->getWorkspace(record.workspaceSize);
                float alpha = 1.f, beta = 0.f;

                stat = cudnnConvolutionForward(
                    context->cudnnHandle(), &alpha, inDesc, inData, knDesc,
                    knData, convDesc, ALGOS[record.algo], wsData,
                    record.workspaceSize, &beta, outDesc, outData);
                if (stat != CUDNN_STATUS_SUCCESS) {
                    continue;
                }
                record.time = timeit(
                    [&]() {
                        cudnnConvolutionForward(context->cudnnHandle(), &alpha,
                                                inDesc, inData, knDesc, knData,
                                                convDesc, ALGOS[record.algo],
                                                wsData, record.workspaceSize,
                                                &beta, outDesc, outData);
                    },
                    [&]() { context->sync(); });
                // printf("mode:%d algo:%d :%.8lf\n", mode, algo, record.time);

                // Update the tune result
                if (ret.time > record.time) {
                    ret = record;
                }
                checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
                checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
                checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
                checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
            }
        }
        // printf("the best algo is %d, the best conv mode is %d\n", ret.algo,
        //        ret.mode);
        IT_ASSERT(ret.time < std::numeric_limits<double>::max(), "No valid "
                                                                 "algorithm "
                                                                 "found");
        return make_ref<ConvCuDnnPerfRecordObj>(ret);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto record = as<ConvCuDnnPerfRecordObj>(_record);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = false;
        for (int i = 0; i < 8; i++) {
            record->algo = i;
            success = cuDNNUnfused(op, record, context);
            if (success) {
                break;
            }
        }
        if (!success) {
            IT_TODO_HALT_MSG("no supported algo");
        }
        IT_ASSERT(success);
    }

  public:
    convCudnn() {
        auto computePtr = [this](const Operator &op, const PerfRecord &record,
                                 const RuntimeObj *context) {
            this->compute(op, record, context);
        };
        funcVec.emplace_back(computePtr);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv, convCudnn, "Conv_cuDNN_CUDA");

REGISTER_CONSTRUCTOR(1, ConvCuDnnPerfRecordObj::from_json);
} // namespace infini
