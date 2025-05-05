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

TEST(graph, run) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor a = g->addTensor(Shape{1, 4, 224, 224}, DataType::Float32);
    Tensor b = g->addTensor(Shape{64, 4, 7, 7}, DataType::Float32);
    Tensor c = g->addTensor(Shape{1, 64, 112, 112}, DataType::Float32);

    auto conv =
        g->addOpWithOutputs<ConvObj>(a, b, c, 3, 3, 2, 2, 1, 1);

    g->dataMalloc(true);

    a->copyin(std::vector<float>(1 * 4 * 224 * 224, 1.0f));
    b->copyin(std::vector<float>(64 * 4 * 7 * 7, 1.0f));

    
}

} // namespace infini
