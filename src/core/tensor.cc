#include "core/tensor.h"
#include "core/blob.h"
#include "core/common.h"
#include "core/data_type.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "utils/data_convert.h"
#include "utils/dataloader.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace infini {

TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)
    : TensorBaseObj(shape_.size(), dtype, runtime), shape(std::move(shape_)),
      _size(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{})) {
}

void TensorObj::updateRange() {
    float max_val = 0;
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        auto max_range = std::max_element(
            tensor_data.begin(), tensor_data.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        max_val = std::abs(*max_range);
    } else if (dtype == DataType::Float16) {
        std::vector<uint16_t> tensor_data = copyout<uint16_t>();
        std::vector<float> data_float(tensor_data.size());
        for (size_t i = 0; i < tensor_data.size(); ++i) {
            data_float[i] = fp16_to_float(tensor_data[i]);
        }
        auto max_range = std::max_element(
            data_float.begin(), data_float.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        max_val = std::abs(*max_range);
    } else {
        IT_TODO_HALT();
    }
    // max
    if (max_val > range) {
        range = max_val;
    }
    // power2_range
}

float TensorObj::getSimulatedLoss() {
    // setQuantizeScale();
    float loss = 0;
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            float temp = tensor_data[i] / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            int8_t q_val = static_cast<int8_t>(t);
            float dq_val = q_val * scale;
            float val_loss = tensor_data[i] - dq_val;
            val_loss = val_loss * val_loss;
            loss += val_loss;
        }
        loss = loss / tensor_data.size();
    } else if (dtype == DataType::Float16) {
        std::vector<uint16_t> tensor_data = copyout<uint16_t>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            float data_fp32 = fp16_to_float(tensor_data[i]);
            float temp = data_fp32 / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            int8_t q_val = static_cast<int8_t>(t);
            float dq_val = q_val * scale;
            float val_loss = data_fp32 - dq_val;
            val_loss = val_loss * val_loss;
            loss += val_loss;
        }
        loss = loss / tensor_data.size();
    } else {
        IT_TODO_HALT();
    }
    return loss;
}

void TensorObj::simulatedQuantize() {
    IT_ASSERT(scale != 0);
    IT_ASSERT(bits_width != 0);
    IT_ASSERT(quantized_range != 0);
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            // float st = tensor_data[i];
            float temp = tensor_data[i] / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            int8_t q_val = static_cast<int8_t>(t);
            tensor_data[i] = q_val * scale;
        }
        copyin(tensor_data);
    } else if (dtype == DataType::Float16) {
        std::vector<uint16_t> tensor_data = copyout<uint16_t>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            float data_fp32 = fp16_to_float(tensor_data[i]);
            float temp = data_fp32 / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            int8_t q_val = static_cast<int8_t>(t);
            tensor_data[i] = float_to_fp16(q_val * scale);
        }
        copyin(tensor_data);
    } else {
        IT_TODO_HALT();
    }
}

void TensorObj::quantize() {
    IT_ASSERT(scale != 0);
    IT_ASSERT(bits_width != 0);
    IT_ASSERT(quantized == true);
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            float temp = tensor_data[i] / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            // int8_t q_val = static_cast<int8_t>(t);
            // tensor_data[i] = q_val;
            tensor_data[i] = t;
        }
        copyin(tensor_data);
    } else if (dtype == DataType::Float16) {
        // 暂时不存在这种情况
        IT_TODO_HALT_MSG("only fp32 -> int8");
        std::vector<uint16_t> tensor_data = copyout<uint16_t>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            float data_fp32 = fp16_to_float(tensor_data[i]);
            float temp = data_fp32 / scale;
            // round
            int t = static_cast<int>(std::rint(temp));
            // clip
            if (t + quantized_range < 0) {
                t = -quantized_range;
            }
            if (t > quantized_range) {
                t = quantized_range;
            }
            // int8_t q_val = static_cast<int8_t>(t);
            // tensor_data[i] = float_to_fp16(q_val);
            tensor_data[i] = float_to_fp16(t);
        }
        copyin(tensor_data);
    } else {
        IT_TODO_HALT();
    }
}

void TensorObj::deQuantize(float val) {
    IT_ASSERT(quantized == true);
    IT_ASSERT(scale != 0);
    IT_ASSERT(bits_width != 0);
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            // int8_t q_val = static_cast<int8_t>(tensor_data[i]);
            // tensor_data[i] = q_val * scale;
            // tensor_data[i] = tensor_data[i] * scale;
            tensor_data[i] = tensor_data[i] * val;
        }
        copyin(tensor_data);
    } else if (dtype == DataType::Float16) {
        // 暂时不存在这种情况
        IT_TODO_HALT_MSG("only int8 -> fp32");
    } else {
        IT_TODO_HALT();
    }
}

void TensorObj::deQuantize() {
    IT_ASSERT(output_scale != 0);
    IT_ASSERT(quantized == true);
    IT_ASSERT(scale != 0);
    IT_ASSERT(bits_width != 0);
    if (dtype == DataType::Float32) {
        std::vector<float> tensor_data = copyout<float>();
        for (size_t i = 0; i < tensor_data.size(); i++) {
            tensor_data[i] = tensor_data[i] * output_scale;
        }
        copyin(tensor_data);
    } else if (dtype == DataType::Float16) {
        // 暂时不存在这种情况
        IT_TODO_HALT_MSG("only int8 -> fp32");
    } else {
        IT_TODO_HALT();
    }
}

void TensorObj::shape_NCHW2NHWC() {
    IT_ASSERT(shape.size() == 4);
    IT_ASSERT(dtype == DataType::Int8 || dtype == DataType::Int32);
    Shape nchw_shape = shape;
    Shape nhwc_shape = {nchw_shape[0], nchw_shape[2], nchw_shape[3],
                        nchw_shape[1]};
    shape = nhwc_shape;
}

void TensorObj::shape_NHWC2NCHW() {
    IT_ASSERT(shape.size() == 4);
    IT_ASSERT(dtype == DataType::Int8 || dtype == DataType::Int32);
    Shape nhwc_shape = shape;
    Shape nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1],
                        nhwc_shape[2]};
    shape = nchw_shape;
}

void TensorObj::data_NCHW2NHWC() {
    // shape_NCHW2NHWC之后
    IT_ASSERT(shape.size() == 4);
    IT_ASSERT(dtype == DataType::Int8 || dtype == DataType::Int32);

    if (dtype == DataType::Int8) {
        std::vector<int8_t> old_data = copyout<int8_t>();
        std::vector<int8_t> new_data(old_data.size());
        int nn = shape[0];
        int hh = shape[1];
        int ww = shape[2];
        int cc = shape[3];
        int n_nchw_strides = hh * ww * cc;
        int c_nchw_strides = hh * ww;
        int h_nchw_strides = ww;
        int n_nhwc_strides = hh * ww * cc;
        int h_nhwc_strides = ww * cc;
        int w_nhwc_strides = cc;
        // int nchw_index = 0;
        // int nhwc_index = 0;
#pragma omp parallel for collapse(3)
        for (int ni = 0; ni < nn; ni++) {
            for (int hi = 0; hi < hh; hi++) {
                for (int wi = 0; wi < ww; wi++) {
                    int nhwc_base = ni * n_nhwc_strides + hi * h_nhwc_strides +
                                    wi * w_nhwc_strides;
                    int nchw_base = ni * n_nchw_strides + hi * h_nchw_strides + wi;
                    for (int ci = 0; ci < cc; ci++) {
                        // nchw_index =
                        //     ni * hh * ww * cc + ci * hh * ww + hi * ww + wi;
                        // nhwc_index =
                        //     ni * hh * ww * cc + hi * ww * cc + wi * cc + ci;
                        // new_data[nhwc_index] = old_data[nchw_index];
                        new_data[nhwc_base + ci] =
                            old_data[nchw_base + ci * c_nchw_strides];
                    }
                }
            }
        }
        copyin(new_data);
    } else if (dtype == DataType::Int32) {
        std::vector<int32_t> old_data = copyout<int32_t>();
        std::vector<int32_t> new_data(old_data.size());
        int nn = shape[0];
        int hh = shape[1];
        int ww = shape[2];
        int cc = shape[3];
        int n_nchw_strides = hh * ww * cc;
        int c_nchw_strides = hh * ww;
        int h_nchw_strides = ww;
        int n_nhwc_strides = hh * ww * cc;
        int h_nhwc_strides = ww * cc;
        int w_nhwc_strides = cc;
        // int nchw_index = 0;
        // int nhwc_index = 0;
#pragma omp parallel for collapse(3)
        for (int ni = 0; ni < nn; ni++) {
            for (int hi = 0; hi < hh; hi++) {
                for (int wi = 0; wi < ww; wi++) {
                    int nhwc_base = ni * n_nhwc_strides + hi * h_nhwc_strides +
                                    wi * w_nhwc_strides;
                    int nchw_base = ni * n_nchw_strides + hi * h_nchw_strides + wi;
                    for (int ci = 0; ci < cc; ci++) {
                        // nchw_index =
                        //     ni * hh * ww * cc + ci * hh * ww + hi * ww + wi;
                        // nhwc_index =
                        //     ni * hh * ww * cc + hi * ww * cc + wi * cc + ci;
                        // new_data[nhwc_index] = old_data[nchw_index];
                        new_data[nhwc_base + ci] =
                            old_data[nchw_base + ci * c_nchw_strides];
                    }
                }
            }
        }
        copyin(new_data);
    } else {
        IT_TODO_HALT();
    }
}

void TensorObj::data_NHWC2NCHW() {
    // shape_NHWC2NCHW之后
    IT_ASSERT(shape.size() == 4);
    IT_ASSERT(dtype == DataType::Int8 || dtype == DataType::Int32);

    if (dtype == DataType::Int8) {
        std::vector<int8_t> old_data = copyout<int8_t>();
        std::vector<int8_t> new_data(old_data.size());
        int nn = shape[0];
        int cc = shape[1];
        int hh = shape[2];
        int ww = shape[3];
        int n_nchw_strides = hh * ww * cc;
        int c_nchw_strides = hh * ww;
        int h_nchw_strides = ww;
        int n_nhwc_strides = hh * ww * cc;
        int h_nhwc_strides = ww * cc;
        int w_nhwc_strides = cc;
        // int nhwc_index = 0;
        // int nchw_index = 0;
#pragma omp parallel for collapse(3)
        for (int ni = 0; ni < nn; ni++) {
            for (int hi = 0; hi < hh; hi++) {
                for (int wi = 0; wi < ww; wi++) {
                    int nhwc_base = ni * n_nhwc_strides + hi * h_nhwc_strides +
                                    wi * w_nhwc_strides;
                    int nchw_base = ni * n_nchw_strides + hi * h_nchw_strides + wi;
                    for (int ci = 0; ci < cc; ci++) {
                        // nhwc_index =
                        //     ni * hh * ww * cc + hi * ww * cc + wi * cc + ci;
                        // nchw_index =
                        //     ni * hh * ww * cc + ci * hh * ww + hi * ww + wi;
                        // new_data[nchw_index] = old_data[nhwc_index];
                        new_data[nchw_base + ci * c_nchw_strides] = old_data[nhwc_base + ci];
                    }
                }
            }
        }
        copyin(new_data);
    } else if (dtype == DataType::Int32) {
        std::vector<int32_t> old_data = copyout<int32_t>();
        std::vector<int32_t> new_data(old_data.size());
        int nn = shape[0];
        int cc = shape[1];
        int hh = shape[2];
        int ww = shape[3];
        int n_nchw_strides = hh * ww * cc;
        int c_nchw_strides = hh * ww;
        int h_nchw_strides = ww;
        int n_nhwc_strides = hh * ww * cc;
        int h_nhwc_strides = ww * cc;
        int w_nhwc_strides = cc;
        // int nhwc_index = 0;
        // int nchw_index = 0;
#pragma omp parallel for collapse(3)
        for (int ni = 0; ni < nn; ni++) {
            for (int hi = 0; hi < hh; hi++) {
                for (int wi = 0; wi < ww; wi++) {
                    int nhwc_base = ni * n_nhwc_strides + hi * h_nhwc_strides +
                                    wi * w_nhwc_strides;
                    int nchw_base = ni * n_nchw_strides + hi * h_nchw_strides + wi;
                    for (int ci = 0; ci < cc; ci++) {
                        // nhwc_index =
                        //     ni * hh * ww * cc + hi * ww * cc + wi * cc + ci;
                        // nchw_index =
                        //     ni * hh * ww * cc + ci * hh * ww + hi * ww + wi;
                        // new_data[nchw_index] = old_data[nhwc_index];
                        new_data[nchw_base + ci * c_nchw_strides] = old_data[nhwc_base + ci];
                    }
                }
            }
        }
        copyin(new_data);
    } else {
        IT_TODO_HALT();
    }
}

string TensorObj::toString() const {
    // Convert data pointer to string
    std::stringstream ss;
    if (data != nullptr)
        ss << data->getPtr<void *>();
    else
        ss << "nullptr data";
    string ret = "Tensor " + std::to_string(guid) + ", Fuid " +
                 std::to_string(fuid) + ", shape " + vecToString(shape) +
                 ", dtype " + dtype.toString() + ", " + runtime->toString() +
                 ", " + ss.str() + ", " + tensorTypeToString() + "\n";
    vector<UidBaseType> targetGuids;
    for (const auto &op : targets)
        targetGuids.emplace_back(op.lock()->getGuid());
    if (auto o = source.lock())
        ret += ", source " + std::to_string(o->getGuid());
    else
        ret += ", source None";
    ret += ", targets " + vecToString(targetGuids);
    return ret;
}

size_t TensorObj::getOffset(const vector<int> &pos) const {
    auto nDim = pos.size();
    IT_ASSERT(shape.size() == nDim);
    if (pos.empty())
        return 0;
    for (size_t i = 0; i < nDim; ++i)
        IT_ASSERT(pos[i] < 0 || pos[i] >= shape[i]);
    size_t idx = pos[0];
    size_t dm = 0;
    while (++dm < nDim)
        idx = idx * shape[dm] + pos[dm];
    return idx;
}

Shape TensorObj::getStride() const {
    Shape stride(getRank());
    ShapeElem p = 1;
    for (auto i = getRank(); i > 0; --i) {
        stride[i - 1] = p;
        p = p * shape[i - 1];
    }
    return stride;
}

void TensorObj::setShape(Shape shape_) {
    shape = shape_;
    size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                  [](auto acc, auto x) { return acc * x; });
    _size = size;
}

void TensorObj::dumpData(std::ofstream &ofs) const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();

#define TRY_DUMP(N)                                                            \
    if (dtype == DataType(N))                                                  \
        ofs << dataToString<DT<N>::t>() << std::endl;

    TRY_DUMP(0)           // fmt: new line
    else TRY_DUMP(1)      //
        else TRY_DUMP(2)  //
        else TRY_DUMP(3)  //
        else TRY_DUMP(4)  //
        else TRY_DUMP(5)  //
        else TRY_DUMP(6)  //
        else TRY_DUMP(7)  //
        else TRY_DUMP(8)  //
        else TRY_DUMP(9)  //
        else TRY_DUMP(10) //
        else TRY_DUMP(11) //
        else TRY_DUMP(12) //
        else TRY_DUMP(13) //
        else TRY_DUMP(16) //
        else IT_TODO_HALT();
    ofs.flush();

#undef TRY_DUMP
}

void TensorObj::printData() const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();

#define TRY_PRINT(N)                                                           \
    if (dtype == DataType(N))                                                  \
        std::cout << dataToString<DT<N>::t>() << std::endl;

    TRY_PRINT(0)           // fmt: new line
    else TRY_PRINT(1)      //
        else TRY_PRINT(2)  //
        else TRY_PRINT(3)  //
        else TRY_PRINT(4)  //
        else TRY_PRINT(5)  //
        else TRY_PRINT(6)  //
        else TRY_PRINT(7)  //
        else TRY_PRINT(8)  //
        else TRY_PRINT(9)  //
        else TRY_PRINT(10) //
        else TRY_PRINT(11) //
        else TRY_PRINT(12) //
        else TRY_PRINT(13) //
        else TRY_PRINT(16) //
        else IT_TODO_HALT();

#undef TRY_PRINT
}

bool TensorObj::equalData(const Tensor &rhs, double relativeError) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    IT_ASSERT(getDType() == rhs->getDType());
    IT_ASSERT(runtime->isCpu());
    IT_ASSERT(rhs->getRuntime()->isCpu());
    if (size() != rhs->size())
        return false;

#define TEST_EQUAL(N)                                                          \
    if (dtype == DataType(N))                                                  \
        return equalDataImpl(getRawDataPtr<DT<N>::t *>(),                      \
                             rhs->getRawDataPtr<DT<N>::t *>(), size(),         \
                             relativeError);

    TEST_EQUAL(0)           // fmt: new line
    else TEST_EQUAL(1)      //
        else TEST_EQUAL(2)  //
        else TEST_EQUAL(3)  //
        else TEST_EQUAL(4)  //
        else TEST_EQUAL(5)  //
        else TEST_EQUAL(6)  //
        else TEST_EQUAL(7)  //
        else TEST_EQUAL(8)  //
        else TEST_EQUAL(9)  //
        else TEST_EQUAL(10) //
        else TEST_EQUAL(11) //
        else TEST_EQUAL(12) //
        else TEST_EQUAL(13) //
        else TEST_EQUAL(16) //
        else IT_TODO_HALT();

#undef TEST_EQUAL
}

void TensorObj::dataMalloc() {
    if (!data)
        data = runtime->allocBlob(getBytes());
}

void TensorObj::copyData(const TensorObj *src) {
    IT_ASSERT(dtype == src->getDType());
    IT_ASSERT(size() == src->size());
    runtime->copyBlob(this, src);
}

void TensorObj::setData(
    const std::function<void(void *, size_t, DataType)> &generator) const {
    IT_ASSERT(data != nullptr);
    if (runtime->isCpu()) {
        generator(getRawDataPtr<void *>(), size(), dtype);
    } else {
        // Create a CPU buffer for the generetor and copy results to the device
        auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
        size_t nBytes = size() * dtype.getSize();
        Blob buffer = cpuRuntime->allocBlob(nBytes);
        generator(buffer->getPtr<void *>(), size(), dtype);
        runtime->copyBlobFromCPU(getRawDataPtr<void *>(),
                                 buffer->getPtr<void *>(), nBytes);
    }
}

void TensorObj::setDataBlob(const Blob &blob) { this->data = blob; }

void TensorObj::load(std::string file_path) { loadTensorData(this, file_path); }

void TensorObj::save(std::string file_path) { saveTensorData(this, file_path); }

Shape TensorObj::getPosByOffset(size_t offset, Shape dim) const {
    Shape pos = dim;
    for (int i = dim.size() - 1; i >= 0; i--) {
        pos[i] = offset % dim.at(i);
        offset = (offset - pos[i]) / dim.at(i);
    }
    return pos;
}

size_t TensorObj::getOffsetByPos(Shape pos, Shape dim) const {
    int n = dim.size();
    size_t offset = pos.at(0);
    for (auto i = 1; i < n; i++)
        offset = offset * dim.at(i) + pos.at(i);

    return offset;
}

size_t TensorObj::getOffsetByBroadcastOffset(size_t bcOffset,
                                             Shape bcDim) const {
    Shape bcPos = getPosByOffset(bcOffset, bcDim);

    Shape pos = bcPos;
    int n = shape.size();
    for (auto i = 0; i < n; i++)
        if (shape.at(i) == 1)
            pos[i] = 0;

    return getOffsetByPos(pos, shape);
}
}; // namespace infini
