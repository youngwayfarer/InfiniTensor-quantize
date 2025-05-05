#include "core/blob.h"
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
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <sys/types.h>

namespace infini {

TEST(graph, int8_quantize_fp32) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto f = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto h = g->addTensor(Shape{2, 3}, DataType::Float32);

    a->setInput();
    b->setWeight();
    d->setWeight();
    e->setWeight();
    f->setOutput();
    h->setOutput();

    a->setQuantized(true);
    b->setQuantized(true);
    c->setQuantized(true);

    // c = a mul b
    // f = c add d
    // h = c add e
    auto mul = g->addOpWithOutputs<MulObj>(a, b, c);
    auto add1 = g->addOpWithOutputs<AddObj>(c, d, f);
    auto add2 = g->addOpWithOutputs<AddObj>(c, e, h);

    g->print();

    g->int_quantize();

    g->print();
}

TEST(graph, int8_quantize_fp16) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    auto a = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto b = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto c = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto d = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto e = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto f = g->addTensor(Shape{2, 3}, DataType::Float16);
    auto h = g->addTensor(Shape{2, 3}, DataType::Float16);

    a->setInput();
    b->setWeight();
    d->setWeight();
    e->setWeight();
    f->setOutput();
    h->setOutput();

    a->setQuantized(true);
    b->setQuantized(true);
    c->setQuantized(true);

    // c = a mul b
    // f = c add d
    // h = c add e
    auto mul = g->addOpWithOutputs<MulObj>(a, b, c);
    auto add1 = g->addOpWithOutputs<AddObj>(c, d, f);
    auto add2 = g->addOpWithOutputs<AddObj>(c, e, h);

    g->print();

    g->int_quantize();

    g->print();
}
}
