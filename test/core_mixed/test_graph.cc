#include "core/blob.h"
#include "core/common.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/graph_handler.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "operators/conv.h"
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

TEST(graph, setAlpha) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{2, 3}, DataType::Float32);
    Tensor b = g->addTensor(Shape{2, 3}, DataType::Float32);
    Tensor c = g->addTensor(Shape{2, 3}, DataType::Float32);

    auto op = g->addOpWithOutputs<MulObj>(a, b, c);

    EXPECT_EQ(g->getAlpha(), 0);
    EXPECT_EQ(op->getAlpha(), 0);

    g->setAlpha(0.5);
    g->setOpAlpha();

    EXPECT_EQ(g->getAlpha(), 0.5);
    EXPECT_EQ(op->getAlpha(), 0.5);
}

TEST(graph, layout) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{1, 4, 224, 224}, DataType::Float32);
    Tensor b = g->addTensor(Shape{64, 4, 7, 7}, DataType::Float32);
    Tensor c = g->addTensor(Shape{1, 64, 112, 112}, DataType::Float32);

    Tensor a_int8 = g->addTensor(Shape{1, 4, 224, 224}, DataType::Int8);
    Tensor b_int8 = g->addTensor(Shape{64, 4, 7, 7}, DataType::Int8);
    Tensor c_int32 = g->addTensor(Shape{1, 64, 112, 112}, DataType::Int32);

    auto cast1 = g->addOpWithOutputs<CastObj>(a, a_int8, CastType::Float2Int8);
    auto cast2 = g->addOpWithOutputs<CastObj>(b, b_int8, CastType::Float2Int8);
    auto conv =
        g->addOpWithOutputs<ConvObj>(a_int8, b_int8, c_int32, 3, 3, 2, 2, 1, 1);
    auto cast3 =
        g->addOpWithOutputs<CastObj>(c_int32, c, CastType::Int322Float);


    Tensor a_fp = g->addTensor(Shape{1, 4, 224, 224}, DataType::Float32);
    Tensor b_fp = g->addTensor(Shape{64, 4, 7, 7}, DataType::Float32);
    Tensor c_fp = g->addTensor(Shape{1, 64, 112, 112}, DataType::Float32);
    auto conv_fp = g->addOpWithOutputs<ConvObj>(a_fp, b_fp, c_fp, 3, 3, 2, 2, 1, 1);

    g->dataMalloc(true);

    a->copyin(std::vector<float>(1 * 4 * 224 * 224, 1.0f));
    b->copyin(std::vector<float>(64 * 4 * 7 * 7, 1.0f));

    a_fp->copyin(std::vector<float>(1 * 4 * 224 * 224, 1.0f));
    b_fp->copyin(std::vector<float>(64 * 4 * 7 * 7, 1.0f));

    // g->print_data();

    std::cout << "=======================" << std::endl;

    runtime->run(g);

    // g->print_data();

    auto c_data = c->copyout<float>();
    auto c_fp_data = c_fp->copyout<float>();
    for (size_t i = 0; i < c->size(); i++) {
        if (c_data[i] != c_fp_data[i]) {
            std::cout << "c_data[" << i << "] = " << c_data[i] << ", c_fp_data["
                      << i << "] = " << c_fp_data[i] << std::endl;
            break;
        }
    }
}

}
