#include "operators/batch_norm.h"
#include "core/kernel.h"
#include "core/runtime.h"

namespace infini {
class BatchNormCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);

        // 获取输入输出张量的原始数据指针
        const float *inData = static_cast<const float *>(
            op->getInputs(0)->getRawDataPtr<void *>());
        float *outData =
            static_cast<float *>(op->getOutput()->getRawDataPtr<void *>());
        const float *meanData = static_cast<const float *>(
            op->getInputs(1)->getRawDataPtr<void *>());
        const float *varData = static_cast<const float *>(
            op->getInputs(2)->getRawDataPtr<void *>());
        const float *scaleData = static_cast<const float *>(
            op->getInputs(3)->getRawDataPtr<void *>());
        const float *biasData = static_cast<const float *>(
            op->getInputs(4)->getRawDataPtr<void *>());

        // 获取原始输入张量的维度
        auto inputShape = op->getInputs(0)->getDims();
        auto outputShape = op->getOutput()->getDims();

        // 获取通道数
        int C = op->getInputs(1)->getDims()[0];

        // 计算各维度的元素数
        int N = outputShape.size() > 0 ? outputShape[0] : 1;
        int H = outputShape.size() > 2 ? outputShape[2] : 1;
        int W = outputShape.size() > 3 ? outputShape[3] : 1;

        float eps = op->getEps();

        // 对每个通道执行归一化
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                float mean = meanData[c];
                float var = varData[c];
                float scale = scaleData[c];
                float bias = biasData[c];
                float stddev = std::sqrt(var + eps);

                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        // 计算输入和输出的索引
                        int index = ((n * C + c) * H + h) * W + w;

                        // 执行批归一化计算: y = ((x - mean) / sqrt(var + eps))
                        // * scale + bias
                        outData[index] =
                            ((inData[index] - mean) / stddev) * scale + bias;
                    }
                }
            }
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::BatchNormalization, BatchNormCpu,
                "BatchNorm_CPU");
} // namespace infini
