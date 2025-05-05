#pragma once
#include "core/common.h"
#include "core/lazy_allocator.h"
#include "core/op_type.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/conv.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>

namespace infini {

class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    OpVec ops;
    LazyAllocator allocator;

    // alpha
    float alpha = 0;
    // QEM
    float QEM = 0;
    // time
    double run_time = 0;
    bool conv_pass = false;

    // operators that could be quantized
    std::unordered_set<OpType> quantizedOps = {OpType::Conv};

  public:
    explicit GraphObj(Runtime runtime)
        : runtime(runtime), allocator(runtime), sorted(false){};
    GraphObj(Runtime runtime, OpVec ops_in);
    string toString() const override;
    Runtime getRuntime() const { return runtime; }

    Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
    Tensor addTensor(const Tensor &tensor);
    TensorVec addTensor(const TensorVec &tensors);
    /**
     * @brief Clone a tensor and add it to the graph.
     */
    Tensor cloneTensor(const Tensor &tensor) {
        return addTensor(tensor->clone(runtime));
    }
    void removeOperator(Operator op) {
        auto it = std::find(ops.begin(), ops.end(), op);
        if (it != ops.end())
            ops.erase(it);
    }

    void removeTensor(Tensor tensor) {
        auto it = std::find(tensors.begin(), tensors.end(), tensor);
        if (it != tensors.end())
            tensors.erase(it);
    }

    void deleteConnection(Tensor tensor, Operator op);
    void addConnection(Tensor tensor, Operator op);
    void replaceConnection(Tensor oldInput, Tensor newInput, Operator op);

    Operator cloneOperator(Operator op, TensorVec inputs, TensorVec outputs) {
        auto opClone = op->clone(inputs, outputs);
        addOperatorAndConnect(opClone);
        return opClone;
    }

    const TensorVec &getTensors() const { return tensors; }
    const OpVec &getOperators() const { return ops; }
    OpVec getComputeOps() const;
    Tensor getTensor(int) const;

    void clearQuantizedOps() { quantizedOps.clear(); }
    void addQuantizedOp(OpType optype) { quantizedOps.insert(optype); }
    bool emptyQuantizedOps() { return quantizedOps.empty(); }
    size_t numQuantizedOps() { return quantizedOps.size(); }
    bool inQuantizedOps(OpType optype) {
        return quantizedOps.find(optype) != quantizedOps.end();
    }
    void printQuantizedOps() {
        std::cout << "quantizedOps num: " << numQuantizedOps() << std::endl;
        for (auto op : quantizedOps) {
            std::cout << op.toString() << std::endl;
        }
    }

    bool getConvPass() const { return conv_pass; }
    void setConvPass(bool val) {
        conv_pass = val;
        for (auto &op : ops) {
            if (op->getOpType() == OpType::Conv) {
                auto conv = std::dynamic_pointer_cast<ConvObj>(op);
                conv->setConvPass(val);
            }
        }
    }
    double get_run_time() const { return run_time; }
    void set_run_time(double val) { run_time = val; }
    float getAlpha() const { return alpha; }
    void setAlpha(float val) { alpha = val; }
    void setOpAlpha() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (inQuantizedOps(op->getOpType())) {
                op->setAlpha(alpha);
            }
        }
    }
    float getQEM() const { return QEM; }
    void setQEM(float val) { QEM = val; }
    void setOpQEM() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (inQuantizedOps(op->getOpType())) {
                op->setQEM(QEM);
            }
        }
    }
    void setBestConfig() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            // if (op->getOpType() == OpType::Conv) {
            if (inQuantizedOps(op->getOpType())) {

                if ((op->getOpType() == OpType::Conv)) {
                    // 不量化到int8
                    int c = op->getInputs(0)->getDims()[1];
                    int f = op->getInputs(1)->getDims()[0];
                    Shape kernel_shape = op->getInputs(1)->getDims();
                    int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
                    // 必须是4的倍数
                    //* kernel size >= [256, 256, 3, 3]会使用conv2d_grouped_direct_kernel
                    //* 避免使用conv2d_grouped_direct_kernel
                    if ((c % 4 != 0 || f % 4 != 0) || (kernel_size >= 589824)) {
                        continue;
                    }
                }

                op->averageLoss();
                op->setInputBit();
                // 暂时认为每个op都量化
                int input_num = op->numInputs();
                IT_ASSERT(input_num == 2);
                for (int i = 0; i < input_num; i++) {
                    Tensor input = op->getInputs(i);
                    input->setQuantized(true);
                }
                int output_num = op->numOutputs();
                IT_ASSERT(output_num == 1);
                for (int i = 0; i < output_num; i++) {
                    Tensor output = op->getOutput(i);
                    output->setQuantized(true);
                    output->setBitsWidth(8);
                }
            }
        }
    }
    // cpu, setbestconfig
    void setBestConfigCPU() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (op->getOpType() == OpType::Conv) {
                op->averageLoss();
                op->setInputBit();
                // 暂时认为每个op都量化
                int input_num = op->numInputs();
                IT_ASSERT(input_num == 2);
                for (int i = 0; i < input_num; i++) {
                    Tensor input = op->getInputs(i);
                    input->setQuantized(true);
                }
                int output_num = op->numOutputs();
                IT_ASSERT(output_num == 1);
                for (int i = 0; i < output_num; i++) {
                    Tensor output = op->getOutput(i);
                    output->setQuantized(true);
                    output->setBitsWidth(8);
                }
            }
        }
    }

    // bit_loss 清空
    void emptyBitLoss() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (inQuantizedOps(op->getOpType())) {
                op->emptyBitLoss();
            }
        }
    }
    void init_g() {
        for (auto &op : ops) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (inQuantizedOps(op->getOpType())) {
                op->initOp();
            }
        }
    }
    void setIntQuantized() {
        for (auto &op : ops) {
            if (op->getOpType() == OpType::Conv) {
                auto conv = std::dynamic_pointer_cast<ConvObj>(op);
                conv->setIntQuantized(true);
            }
        }
    }

    /**
     * Sort the nodes in topological order.
     * It returns true if the sorting is successful.
     * Otherwise false is returned, means that there are rings in the graph,
     * so the topological sorting fails.
     */
    bool topo_sort();

    void optimize();

    // mixed precision quantization
    void mixed_quantize();

    void int_quantize();

    void print_data();

    // output的bit为8， 根据range直接计算scale
    void output_calibrate();

    void int_quantize_cpu();

    void shape_infer();

    void dataMalloc(bool useNaiveAllocator = false, size_t memPoolSize = 0);

    Tensor cloneKV(Tensor &tensor);

    void freeHeap();

    /**
     * @brief Add an operator and create its outputs. Output tensor arguments
     * should be empty Refs (e.g., nullptr).
     */
    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief Add an operator with its outputs specified.
     */
    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief Gets input tensors of this graph.
     */
    inline TensorVec getInputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (!t->getSource())
                ret.emplace_back(t);
        return ret;
    }

    /**
     * @brief Gets output tensors of this graph.
     */
    inline TensorVec getOutputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (t->getTargets().empty())
                ret.emplace_back(t);
        return ret;
    }

    bool checkValid() const;

  private:
    /**
     * @brief Add reverse connections and Op relationship in ctor.
     */
    void addOperatorAndConnect(const Operator &op);

    /**
     * @brief If the nodes is sorted in topological order.
     */
    bool sorted;

    /**
     * @brief If the weight tensors are allocated.
     */
    bool weightAllocated = false;
};

} // namespace infini
