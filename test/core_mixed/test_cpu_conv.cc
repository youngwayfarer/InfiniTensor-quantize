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
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{1, 3, 224, 224}, DataType::Int8);
    Tensor b = g->addTensor(Shape{64, 3, 7, 7}, DataType::Int8);
    Tensor c = g->addTensor(Shape{1, 64, 112, 112}, DataType::Int32);

    auto conv1 = g->addOpWithOutputs<ConvObj>(a, b, c, 3, 3, 2, 2, 1, 1);

    std::cout << "=======================" << std::endl;

    Tensor a_fp = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    Tensor b_fp = g->addTensor(Shape{64, 3, 7, 7}, DataType::Float32);
    Tensor c_fp = g->addTensor(Shape{1, 64, 112, 112}, DataType::Float32);
    auto conv1_fp =
        g->addOpWithOutputs<ConvObj>(a_fp, b_fp, c_fp, 3, 3, 2, 2, 1, 1);

    g->dataMalloc();

    a->copyin(
        std::vector<int8_t>(1 * 3 * 224 * 224, 3)); // Fill with 1s for testing
    b->copyin(
        std::vector<int8_t>(64 * 3 * 7 * 7, 4)); // Fill with 1s for testing

    a_fp->copyin(
        std::vector<float>(1 * 3 * 224 * 224, 3)); // Fill with 1s for testing
    b_fp->copyin(
        std::vector<float>(64 * 3 * 7 * 7, 4)); // Fill with 1s for testing

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

} // namespace infini
