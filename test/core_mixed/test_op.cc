#include "core/common.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "cuda/cuda_runtime.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <vector>

namespace infini {

TEST(conv, quantizedOps) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    auto g = make_ref<GraphObj>(runtime);

    auto a = g->addTensor(Shape{3, 4}, DataType::Float32);
    auto b = g->addTensor(Shape{3, 4}, DataType::Float32);
    auto c = g->addTensor(Shape{3, 4}, DataType::Float32);
    auto d = g->addTensor(Shape{3, 4}, DataType::Float32);
    auto mul = g->addOpWithOutputs<MulObj>(a, b, c);
    auto add = g->addOpWithOutputs<AddObj>(a, b, d);

    IT_ASSERT(!g->emptyQuantizedOps());
    IT_ASSERT(g->numQuantizedOps() == 1);
    g->printQuantizedOps();
    IT_ASSERT(g->inQuantizedOps(OpType::Conv));
    IT_ASSERT(!g->inQuantizedOps(OpType::Mul));

    g->addQuantizedOp(OpType::Conv);

    IT_ASSERT(!g->emptyQuantizedOps());
    IT_ASSERT(g->numQuantizedOps() == 1);
    IT_ASSERT(g->inQuantizedOps(OpType::Conv));

    g->addQuantizedOp(OpType::Mul);

    IT_ASSERT(!g->emptyQuantizedOps());
    IT_ASSERT(g->numQuantizedOps() == 2);
    IT_ASSERT(g->inQuantizedOps(OpType::Conv));
    IT_ASSERT(g->inQuantizedOps(OpType::Mul));

    IT_ASSERT(g->inQuantizedOps(mul->getOpType()));
    IT_ASSERT(!g->inQuantizedOps(add->getOpType()));

    g->clearQuantizedOps();

    IT_ASSERT(g->emptyQuantizedOps());
    IT_ASSERT(g->numQuantizedOps() == 0);
    IT_ASSERT(!g->inQuantizedOps(OpType::Conv));
    IT_ASSERT(!g->inQuantizedOps(OpType::Mul));

    IT_ASSERT(!g->inQuantizedOps(mul->getOpType()));
    IT_ASSERT(!g->inQuantizedOps(add->getOpType()));

    g->addQuantizedOp(OpType::Mul);

    IT_ASSERT(g->inQuantizedOps(mul->getOpType()));
    IT_ASSERT(!g->inQuantizedOps(add->getOpType()));
}

} // namespace infini
