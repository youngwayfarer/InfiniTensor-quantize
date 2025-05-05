#include "core/common.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "operators/conv.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <vector>

namespace infini {

TEST(conv, int8) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{1, 4, 224, 224}, DataType::Int8);
    Tensor b = g->addTensor(Shape{64, 4, 7, 7}, DataType::Int8);
    Tensor c = g->addTensor(Shape{1, 64, 112, 112}, DataType::Int32);

    auto conv1 = g->addOpWithOutputs<ConvObj>(a, b, c, 3, 3, 2, 2, 1, 1);

    std::cout << "=======================" << std::endl;

    Tensor a_fp = g->addTensor(Shape{1, 4, 224, 224}, DataType::Float32);
    Tensor b_fp = g->addTensor(Shape{64, 4, 7, 7}, DataType::Float32);
    Tensor c_fp = g->addTensor(Shape{1, 64, 112, 112}, DataType::Float32);
    auto conv1_fp = g->addOpWithOutputs<ConvObj>(a_fp, b_fp, c_fp, 3, 3, 2, 2, 1, 1);

    g->dataMalloc();

    a->copyin(
        std::vector<int8_t>(1 * 4 * 224 * 224, 1)); // Fill with 1s for testing
    b->copyin(
        std::vector<int8_t>(64 * 4 * 7 * 7, 1)); // Fill with 1s for testing

    a_fp->copyin(
        std::vector<float>(1 * 4 * 224 * 224, 1)); // Fill with 1s for testing
    b_fp->copyin(
        std::vector<float>(64 * 4 * 7 * 7, 1)); // Fill with 1s for testing

    runtime->run(g);

    // g->print_data();
    vector<int32_t> result1 = c->copyout<int32_t>();
    // std::vector<int8_t> result1 = c->copyout<int8_t>();
    vector<float> res_fp32(result1.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        res_fp32[i] = static_cast<float>(result1[i]);
    }

    vector<float> result2 = c_fp->copyout<float>();

    bool equal = true;
    for (size_t i = 0; i < res_fp32.size(); i++) {
        if (res_fp32[i] != result2[i]) {
            equal = false;
            break;
        }
    }
    if (equal) {
        std::cout << "equal" << std::endl;
    } else {
        std::cout << "not equal" << std::endl;
    }
}


TEST(conv, int8_mini) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    // 创建小尺寸张量用于测试
    // NHWC格式：[1, 2, 2, 2] - 批次大小为1，高宽各2，通道数2
    Tensor a = g->addTensor(Shape{1, 4, 2, 2}, DataType::Int8);
    // KHWC格式：[1, 1, 1, 2] - 1个卷积核，大小1x1，输入通道2
    Tensor b = g->addTensor(Shape{4, 4, 1, 1}, DataType::Int8);
    // 输出：[1, 2, 2, 1] - 批次大小为1，高宽各2，通道数1
    Tensor c = g->addTensor(Shape{1, 4, 2, 2}, DataType::Int32);

    // 创建卷积操作 - 无填充，步长为1，膨胀系数为1
    auto conv1 = g->addOpWithOutputs<ConvObj>(a, b, c, 0, 0, 1, 1, 1, 1);

    // 创建对应的Float32版本做对比
    Tensor a_fp = g->addTensor(Shape{1, 4, 2, 2}, DataType::Float32);
    Tensor b_fp = g->addTensor(Shape{4, 4, 1, 1}, DataType::Float32);
    Tensor c_fp = g->addTensor(Shape{1, 4, 2, 2}, DataType::Float32);
    auto conv2 =
        g->addOpWithOutputs<ConvObj>(a_fp, b_fp, c_fp, 0, 0, 1, 1, 1, 1);

    g->dataMalloc(true);

    // 设置特定的输入数据以便进行验证
    // 输入为每个位置值为1-8的数据
    std::vector<int8_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 
                                      9, 10, 11, 12, 13, 14, 15, 16};
    a->copyin(input_data);

    // 权重为{3, 4}
    std::vector<int8_t> weight_data = {1, 1, 1, 10, 1, 1, 1, 1,
                                       2, 3, 2, 2, 2, 3, 4, 2};
    b->copyin(weight_data);

    // 对Float32版本使用相同的数据
    std::vector<float> input_data_fp(16);
    for (int i = 0; i < 16; i++) {
        input_data_fp[i] = static_cast<float>(input_data[i]);
    }
    a_fp->copyin(input_data_fp);

    std::vector<float> weight_data_fp(16);
    for (int i = 0; i < 16; i++) {
        weight_data_fp[i] = static_cast<float>(weight_data[i]);
    }
    b_fp->copyin(weight_data_fp);

    // 执行计算
    runtime->run(g);


    // g->print_data();
    vector<int32_t> result1 = c->copyout<int32_t>();
    // for (size_t i = 0; i < result1.size(); ++i) {
    //     std::cout << "result1[" << i << "]: " << result1[i] << std::endl;
    // }
    vector<float> result2 = c_fp->copyout<float>();
    // for (size_t i = 0; i < result2.size(); ++i) {
    //     std::cout << "result2[" << i << "]: " << result2[i] << std::endl;
    // }

    for (size_t i = 0; i < result1.size(); i++) {
        if (result1[i] != result2[i]) {
            std::cout << "not equal" << std::endl;
            break;
        }
    }

}

}
