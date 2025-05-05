#include "core/blob.h"
#include "core/common.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/graph_handler.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/unary.h"
#include "test.h"
#include "utils/data_convert.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <sys/types.h>
#include <vector>

namespace infini {

//! not right now
// TEST(tensor, quantized_bits) {
//     auto runtime = make_ref<CudaRuntimeObj>();
//     // auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);

//     auto a = g->addTensor(Shape{1, 2, 3, 4}, DataType::Float32);
//     EXPECT_EQ(a->isQuantized(), false);
//     EXPECT_EQ(a->getBitsWidth(), 0);
//     EXPECT_EQ(a->getRange(), 0);
//     EXPECT_EQ(a->getScale(), 0);
    

//     a->setQuantized(true);
//     a->setBitsWidth(8);
//     a->setRange(1);
//     a->setScale(0.2);
//     EXPECT_EQ(a->isQuantized(), true);
//     EXPECT_EQ(a->getBitsWidth(), 8);
//     EXPECT_EQ(a->getRange(), 1);
//     // EXPECT_EQ(a->getScale(), 0.1);
//     EXPECT_FLOAT_EQ(a->getScale(), 0.2);

//     a->setQuantized(false);
//     EXPECT_EQ(a->isQuantized(), false);
//     EXPECT_EQ(a->getBitsWidth(), 0);
// }

TEST(tensor, modify_value) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float16);

    g->dataMalloc();

    a->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    b->copyin(vector<uint16_t>{float_to_fp16(2), float_to_fp16(3), float_to_fp16(10), float_to_fp16(14), float_to_fp16(5), float_to_fp16(6)});
    g->print_data();
    std::cout << "==================================" << std::endl;
    vector<float> a_data = a->copyout<float>();
    for (auto& data : a_data) {
        data = data * 2;
    }
    a->copyin(a_data);
    vector<uint16_t> b_data = b->copyout<uint16_t>();
    for (auto& data : b_data) {
        data = float_to_fp16(fp16_to_float(data) * 2);
    }
    b->copyin(b_data);
    g->print_data();
    std::cout << "==================================" << std::endl;
    vector<float> a_data2 = a->copyout<float>();
    for (size_t i = 0; i < a_data2.size(); i++) {
        a_data2[i] = a_data2[i] + 1;
    }
    a->copyin(a_data2);
    g->print_data();
}

TEST(tensor, getSimulatedLoss) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{2, 3}, DataType::Float32);
    g->dataMalloc();
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    a->copyin(a_data);
    a->setRange(6);
    a->setBitsWidth(8);
    float loss = a->getSimulatedLoss();

    float scale = a->getScale();
    int quantized_range = 127;

    float ans = 0;
    for (size_t i = 0; i < a_data.size(); i++) {
        float temp = a_data[i] / scale;
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
        float val_loss = a_data[i] - dq_val;
        val_loss = val_loss * val_loss;
        ans += val_loss;
    }
    ans = ans / 6;

    std::cout << "loss: " << loss << std::endl;
    std::cout << "ans: " << ans << std::endl;
    EXPECT_FLOAT_EQ(loss, ans);
}

} // namespace infini
