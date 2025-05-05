#include "core/blob.h"
#include "core/data_type.h"
#include "core/graph.h"
#include "core/graph_handler.h"
#include "core/runtime.h"
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

// TEST(mixed_quantize, resize) {
//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     auto handler = make_ref<GraphHandlerObj>(cudaRuntime);
//     auto i = handler->tensor({1, 2, 3}, DataType::Float16.getIndex());
//     auto o = handler->tensor({1, 4, 6}, DataType::Float16.getIndex());
//     auto scales = handler->tensor({3}, DataType::Float32.getIndex());
//     handler->resize(i, o, std::nullopt, nullptr, scales, nullptr, {}, {1, 2, 2},
//                     {}, "nearest", "stretch", "floor", "asymmetric");

//     handler->data_malloc();

//     handler->print();

//     auto resize_op = o->getSource();

//     handler->mixed_quantize();
//     handler->print();

//     auto cast_op = resize_op->getPredecessors()[0];

//     EXPECT_EQ(i->getTargets().size(), 1);
//     EXPECT_EQ(cast_op->numInputs(), 1);
//     EXPECT_EQ(cast_op->numOutputs(), 1);
//     auto t6 = cast_op->getOutputs()[0];
//     EXPECT_EQ(t6->getTargets().size(), 1);
//     EXPECT_EQ(resize_op->numInputs(), 2);
//     EXPECT_EQ(resize_op->numOutputs(), 1);
// }

TEST(mixed_quantize, add) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);

    // auto runtime = NativeCpuRuntimeObj::getInstance();
    // Graph g = make_ref<GraphObj>(runtime);

    auto t0 = g->addTensor({1, 2, 3}, DataType::Float32);
    auto t1 = g->addTensor({1, 2, 3}, DataType::Float32);
    auto t2 = g->addTensor({1, 2, 3}, DataType::Float32);
    auto t3 = g->addTensor({1, 2, 3}, DataType::Float32);
    auto t4 = g->addTensor({1, 2, 3}, DataType::Float32);

    g->addOpWithOutputs<AddObj>(t0, t1, t2);
    g->addOpWithOutputs<AddObj>(t2, t3, t4);

    // g->dataMalloc(true);
    g->dataMalloc();

    t0->copyin(vector<float>{2, 4, 6, 8, 11, 14});
    t1->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    t3->copyin(vector<float>{1, 1, 1, 1, 1, 1});

    g->print();

    g->mixed_quantize();
    g->print();

    // EXPECT_EQ(g->getTensors().size(), 8);
    // EXPECT_EQ(g->getOperators().size(), 5);

    cudaRuntime->run(g);
    // runtime->run(g);

    // vector<float> result;
    // result = t4->copyout<float>();
    // for (auto i : result) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    for (auto tensor : g->getTensors()) {
        tensor->print();
        std::cout << "data below: " << std::endl;
        if (tensor->getDType() == DataType::Float32) {
            auto data = tensor->copyout<float>();
            for (auto i : data) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        } else if (tensor->getDType() == DataType::Float16) {
            vector<uint16_t> data = tensor->copyout<uint16_t>();
            for (auto i : data) {
                auto fp32_data = fp16_to_float(i);
                std::cout << fp32_data << " ";
            }
            std::cout << std::endl;
        }
        // tensor->printData();
        std::cout << "------------------------" << std::endl;
    }
}

// TEST(mixed_quantize, sub1) {
//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     Graph g = make_ref<GraphObj>(cudaRuntime);

//     // auto runtime = NativeCpuRuntimeObj::getInstance();
//     // Graph g = make_ref<GraphObj>(runtime);

//     auto t0 = g->addTensor({1, 2, 3}, DataType::Float32);
//     auto t1 = g->addTensor({1, 2, 3}, DataType::Float32);
//     auto t2 = g->addTensor({1, 2, 3}, DataType::Float32);
//     auto t3 = g->addTensor({1, 2, 3}, DataType::Float32);
//     auto t4 = g->addTensor({1, 2, 3}, DataType::Float32);

//     g->addOpWithOutputs<SubObj>(t0, t1, t2);
//     auto sub = g->addOpWithOutputs<SubObj>(t2, t3, t4);

//     // g->dataMalloc(true);
//     g->dataMalloc();

//     t0->copyin(vector<float>{2, 4, 6, 8, 11, 14});
//     t1->copyin(vector<float>{1, 2, 3, 4, 5, 6});
//     t3->copyin(vector<float>{1, 1, 1, 1, 1, 1});

//     g->print();

//     g->mixed_quantize();
//     g->print();

//     // EXPECT_EQ(g->getTensors().size(), 8);
//     // EXPECT_EQ(g->getOperators().size(), 5);

//     //! 这里可能会遇到段错误
//     // 是因为在使用infiniop时，在operator类的析构函数中（作用域结束，graphobj析构，调用op析构），如~ElementWiseObj中，调用了infiniopDestroyAddDescriptor
//     // 但此时根本没有计算，创建AddDescriptor，因此会引发段错误

//     cudaRuntime->run(g);
//     // runtime->run(g);

//     // vector<float> result;
//     // result = t4->copyout<float>();
//     // for (auto i : result) {
//     //     std::cout << i << " ";
//     // }
//     // std::cout << std::endl;

//     for (auto tensor : g->getTensors()) {
//         tensor->print();
//         std::cout << "data below: " << std::endl;
//         if (tensor->getDType() == DataType::Float32) {
//             auto data = tensor->copyout<float>();
//             for (auto i : data) {
//                 std::cout << i << " ";
//             }
//             std::cout << std::endl;
//         } else if (tensor->getDType() == DataType::Float16) {
//             vector<uint16_t> data = tensor->copyout<uint16_t>();
//             for (auto i : data) {
//                 auto fp32_data = fp16_to_float(i);
//                 std::cout << fp32_data << " ";
//             }
//             std::cout << std::endl;
//         }
//         // tensor->printData();
//         std::cout << "------------------------" << std::endl;
//     }
// }

// TEST(mixed_quantize, sub_fp16) {
//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     Graph g = make_ref<GraphObj>(cudaRuntime);

//     auto t0 = g->addTensor({1, 2, 3}, DataType::Float16);
//     auto t1 = g->addTensor({1, 2, 3}, DataType::Float16);
//     auto t2 = g->addTensor({1, 2, 3}, DataType::Float16);
//     auto t3 = g->addTensor({1, 2, 3}, DataType::Float16);
//     auto t4 = g->addTensor({1, 2, 3}, DataType::Float16);

//     g->addOpWithOutputs<SubObj>(t0, t1, t2);
//     auto sub = g->addOpWithOutputs<SubObj>(t2, t3, t4);

//     g->dataMalloc(true);
//     // g->dataMalloc();

//     t0->copyin(vector<uint16_t>{float_to_fp16(2), float_to_fp16(4), float_to_fp16(6), float_to_fp16(8), float_to_fp16(11), float_to_fp16(14)});
//     t1->copyin(vector<uint16_t>{float_to_fp16(1), float_to_fp16(2), float_to_fp16(3), float_to_fp16(4), float_to_fp16(5), float_to_fp16(6)});
//     t3->copyin(vector<uint16_t>{float_to_fp16(1), float_to_fp16(1), float_to_fp16(1), float_to_fp16(1), float_to_fp16(1), float_to_fp16(1)});

//     g->print();

//     //! 这里可能会遇到段错误
//     // 是因为在使用infiniop时，在operator类的析构函数中（作用域结束，graphobj析构，调用op析构），如~ElementWiseObj中，调用了infiniopDestroyAddDescriptor
//     // 但此时根本没有计算，创建AddDescriptor，因此会引发段错误

//     cudaRuntime->run(g);

//     for (auto tensor : g->getTensors()) {
//         tensor->print();
//         std::cout << "data below: " << std::endl;
//         if (tensor->getDType() == DataType::Float32) {
//             auto data = tensor->copyout<float>();
//             for (auto i : data) {
//                 std::cout << i << " ";
//             }
//             std::cout << std::endl;
//         } else if (tensor->getDType() == DataType::Float16) {
//             vector<uint16_t> data = tensor->copyout<uint16_t>();
//             for (auto i : data) {
//                 auto fp32_data = fp16_to_float(i);
//                 std::cout << fp32_data << " ";
//             }
//             std::cout << std::endl;
//         }
//         // tensor->printData();
//         std::cout << "------------------------" << std::endl;
//     }
// }


} // namespace infini
