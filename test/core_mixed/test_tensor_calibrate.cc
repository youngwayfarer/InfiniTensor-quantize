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
#include "test.h"
#include "utils/data_convert.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <sys/types.h>
#include <vector>
#include <iostream>


namespace infini {

TEST(calibrate, record_range_fp32) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float32);

    g->addOpWithOutputs<AddObj>(a, b, c);
    g->addOpWithOutputs<SubObj>(c, d, e);

    g->dataMalloc(true);

    a->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    b->copyin(vector<float>{3, 4, 5, 6, 7, 8});
    d->copyin(vector<float>{1, 2, 3, 4, 5, 6});

    b->setWeight();

    for (auto op : g->getOperators()) {
        op->initInfiniOp(cudaRuntime);
    }

    cudaRuntime->run(g, false, false, true);

    g->print_data();
    EXPECT_EQ(a->getRange(), 6);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 14);

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    a->copyin(vector<float>{-1, -2, -3, -4, -5, -12});
    cudaRuntime->run(g, false, false, true);
    g->print_data();
    EXPECT_EQ(a->getRange(), 12);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 14);

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    b->copyin(vector<float>{-3, -4, -5, -6, -7, -10});
    cudaRuntime->run(g, false, false, true);
    g->print_data();
    EXPECT_EQ(a->getRange(), 12);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 22);
}

TEST(calibrate, record_range_fp16) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float16);

    g->addOpWithOutputs<AddObj>(a, b, c);
    g->addOpWithOutputs<SubObj>(c, d, e);

    g->dataMalloc(true);

    a->copyin(vector<uint16_t>{float_to_fp16(1), float_to_fp16(2),
                               float_to_fp16(3), float_to_fp16(4),
                               float_to_fp16(5), float_to_fp16(6)});
    b->copyin(vector<uint16_t>{float_to_fp16(3), float_to_fp16(4),
                               float_to_fp16(5), float_to_fp16(6),
                               float_to_fp16(7), float_to_fp16(8)});
    d->copyin(vector<uint16_t>{float_to_fp16(1), float_to_fp16(2),
                               float_to_fp16(3), float_to_fp16(4),
                               float_to_fp16(5), float_to_fp16(6)});

    b->setWeight();

    for (auto op : g->getOperators()) {
        op->initInfiniOp(cudaRuntime);
    }

    cudaRuntime->run(g, false, false, true);

    g->print_data();
    EXPECT_EQ(a->getRange(), 6);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 14);

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    a->copyin(vector<uint16_t>{float_to_fp16(-1), float_to_fp16(-2),
                               float_to_fp16(-3), float_to_fp16(-4),
                               float_to_fp16(-5), float_to_fp16(-12)});
    cudaRuntime->run(g, false, false, true);
    g->print_data();
    EXPECT_EQ(a->getRange(), 12);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 14);

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    b->copyin(vector<uint16_t>{float_to_fp16(-3), float_to_fp16(-4),
                               float_to_fp16(-5), float_to_fp16(-6),
                               float_to_fp16(-7), float_to_fp16(-10)});
    cudaRuntime->run(g, false, false, true);
    g->print_data();
    EXPECT_EQ(a->getRange(), 12);
    EXPECT_EQ(b->getRange(), 8);
    EXPECT_EQ(c->getRange(), 22);
}
}
