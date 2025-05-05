#include "core/blob.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/graph_handler.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "operators/element_wise.h"
#include "utils/data_convert.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <sys/types.h>
#include <vector>

namespace infini {

TEST(modify, add_sub_fp32) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto f = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto h = g->addTensor(Shape{2, 3}, DataType::Float32);

    g->addOpWithOutputs<AddObj>(a, b, c);
    g->addOpWithOutputs<SubObj>(c, d, e);
    g->addOpWithOutputs<MulObj>(e, f, h);

    g->dataMalloc(true);

    a->copyin(vector<float>{1, 1, 1, 1, 1, 1});
    b->copyin(vector<float>{2, 2, 2, 2, 2, 2});
    d->copyin(vector<float>{4, 3, 3, 3, 3, 3});
    f->copyin(vector<float>{5, 5, 5, 5, 5, 5});

    for (auto op : g->getOperators()) {
        op->initInfiniOp(cudaRuntime);
    }

    g->print_data();
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;

    cudaRuntime->run(g);

    g->print_data();
}

TEST(modify, add_sub_fp16) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto f = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto h = g->addTensor(Shape{2, 3}, DataType::Float16);

    g->addOpWithOutputs<AddObj>(a, b, c);
    g->addOpWithOutputs<SubObj>(c, d, e);
    g->addOpWithOutputs<MulObj>(e, f, h);

    g->dataMalloc(true);

    a->copyin(vector<uint16_t>{float_to_fp16(1), float_to_fp16(1),
                               float_to_fp16(1), float_to_fp16(1),
                               float_to_fp16(1), float_to_fp16(1)});
    b->copyin(vector<uint16_t>{float_to_fp16(2), float_to_fp16(2),
                               float_to_fp16(2), float_to_fp16(2),
                               float_to_fp16(2), float_to_fp16(2)});
    d->copyin(vector<uint16_t>{float_to_fp16(4), float_to_fp16(3),
                               float_to_fp16(3), float_to_fp16(3),
                               float_to_fp16(3), float_to_fp16(3)});
    f->copyin(vector<uint16_t>{float_to_fp16(5), float_to_fp16(5),
                               float_to_fp16(5), float_to_fp16(5),
                               float_to_fp16(5), float_to_fp16(5)});

    for (auto op : g->getOperators()) {
        op->initInfiniOp(cudaRuntime);
    }

    g->print_data();
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "=========================================" << std::endl;

    cudaRuntime->run(g);

    g->print_data();
}

} // namespace infini

