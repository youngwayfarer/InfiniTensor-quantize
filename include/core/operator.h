#pragma once

#include "core/common.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "utils/infiniop_utils.h"

namespace infini {
using KernelAttrs = std::tuple<Device, OpType::underlying_t>;

struct OpPerfKey {
    HashType hash;
    OpType::underlying_t opType;
    vector<int> attrs;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    OpPerfKey() = default;
    OpPerfKey(HashType hash, OpType opType, vector<int> attrs = {})
        : hash(hash), opType(opType.underlying()), attrs(attrs) {}
    bool operator==(const OpPerfKey &rhs) const {
        if (hash != rhs.hash)
            return false;
        if (opType != rhs.opType)
            return false;
        if (attrs != rhs.attrs)
            return false;
        return true;
    }

    // TODO: remove this function after we use unordered_map in PerfEngine
    bool operator<(const OpPerfKey &rhs) const {
        if (hash != rhs.hash)
            return hash < rhs.hash;
        if (opType != rhs.opType)
            return opType < rhs.opType;
        if (attrs.size() != rhs.attrs.size())
            return attrs.size() < rhs.attrs.size();
        for (size_t i = 0; i < attrs.size(); ++i)
            if (attrs[i] != rhs.attrs[i])
                return attrs[i] < rhs.attrs[i];
        return false;
    }
};

class GraphObj;
class OperatorObj : public Object {
    friend class GraphObj;

  protected:
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    vector<WRef<OperatorObj>> predecessors;
    vector<WRef<OperatorObj>> successors;
    void *opDesc;

    // alpha
    float alpha = 0;
    // {w, a} : loss
    std::map<std::pair<int, int>, float> bit_loss;
    // the num of calibrate data
    int calibrate_num = 0;
    float QEM = 0;

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0;
    virtual vector<DataType> inferDataType(const TensorVec &inputs) const;
    /**
     * @brief Constructs outputs (if requried) and check whether the operator is
     * valid.
     *
     * @param graph If graph is not nullptr, outputs should be created in this
     * function.
     */
    bool checkValid(GraphObj *graph);
    virtual void initInfiniOp(const Runtime context);
    OpPerfKey getOpPerfKey() const;
    /**
     * @brief Hash operator attributes. Input and output shapes are not
     * considered.
     */
    HashType hash() const;

  public:
  public: // getter and setter
    const TensorVec &getInputs() const { return inputs; }
    const TensorVec &getOutputs() const { return outputs; }
    void *getOpDesc() const { return opDesc; }
    Tensor getInputs(size_t i) const { return inputs.at(i); }
    Tensor getOutput() const {
        IT_ASSERT(outputs.size() == 1, "Unimplemented");
        return outputs[0];
    }
    Tensor getOutput(size_t i) const {
        IT_ASSERT(i < outputs.size(), "Index exceeded");
        return outputs.at(i);
    }
    OpVec getPredecessors() const { return wrefs_to_refs(predecessors); }
    OpVec getSuccessors() const { return wrefs_to_refs(successors); }
    OpType getOpType() const { return type; }
    // HACK: set correct data type
    DataType getDType() const { return getInputs(0)->getDType(); }
    DataType getOutDType() const { return getOutput()->getDType(); }
    virtual int numInputs() const = 0;
    virtual int numOutputs() const = 0;

    // 暂时只考虑conv, mul
    // 其余算子未必正确
    float getAlpha() const { return alpha; }
    void setAlpha(float val) { alpha = val; }
    float getQEM() const { return QEM; }
    void setQEM(float val) { QEM = val; }
    void computeLoss();
    int getCalibrateNum() const { return calibrate_num; }
    void calibrateNumIncrement() { calibrate_num++; }
    void averageLoss() {
        IT_ASSERT(calibrate_num > 0);
        // IT_ASSERT(calibrate_num == 2);
        IT_ASSERT(bit_loss.size() == 25);
        for (auto it = bit_loss.begin(); it != bit_loss.end(); it++) {
            it->second /= calibrate_num;
        }
        calibrate_num = 0;
    }
    // 外部调用averageLoss后， 才能调用getBestConfig
    std::pair<int, int> getBestConfig() {
        IT_ASSERT(calibrate_num == 0);
        std::pair<int, int> bestConfig;
        float base_loss = 0;
        for (int w = 8; w >= 4; w--) {
            for (int a = 8; a >= 4; a--) {
                auto it = bit_loss.find(std::make_pair(w, a));
                IT_ASSERT(it != bit_loss.end());
                if (w == 8 && a == 8) {
                    // 双int8，姑且认为最好
                    base_loss = it->second;
                    bestConfig = std::make_pair(w, a);
                    continue;
                }

                if (it->second < base_loss * QEM) {
                    bestConfig = std::make_pair(w, a);
                }
            }
        }
        return bestConfig;
    }
    void setInputBit() {
        std::pair<int, int> bestConfig = getBestConfig();
        int w = bestConfig.first;
        int a = bestConfig.second;
        Tensor input_a = inputs[0];
        input_a->setBitsWidth(a);
        Tensor input_w = inputs[1];
        input_w->setBitsWidth(w);

        //* test for int6, int7 of activation
        // input_a->setBitsWidth(7);
        // input_a->setBitsWidth(6);
        // input_a->setBitsWidth(5);
        // input_a->setBitsWidth(4);
    }
    void simulatedInput() {
        //* 这里与真正量化改变图结构时保持一致(setBestConfig)
        //* 因为setBestConfig会跳过不量化的算子，这里不跳过的话会导致这些算子出现巨大误差
        if (getOpType() == OpType::Conv) {
            int c = getInputs(0)->getDims()[1];
            int f = getInputs(1)->getDims()[0];
            Shape kernel_shape = getInputs(1)->getDims();
            int kernel_size = kernel_shape[0] * kernel_shape[1] *
                              kernel_shape[2] * kernel_shape[3];
            if ((c % 4 != 0 || f % 4 != 0) || (kernel_size >= 589824)) {
                return;
            }
        }

        Tensor input_a = inputs[0];
        input_a->simulatedQuantize();
        Tensor input_w = inputs[1];
        input_w->simulatedQuantize();
    }
    void emptyBitLoss() { bit_loss.clear(); }
    void initOp() {
        calibrate_num = 0;
        bit_loss.clear();
        IT_ASSERT(bit_loss.size() == 0);
    }

    /**
     * @brief Clone this operator and replace its inputs and outputs.
     *
     * @param newInputs
     * @param newOutputs
     * @return Operator
     */
    virtual Operator clone(const TensorVec &newInputs,
                           const TensorVec &newOutputs) const = 0;

  protected:
    optional<vector<Shape>> inferShape();
    vector<DataType> inferDataType() const;

  private:
    /**
     * @brief The returned vector includes operator attributes, such as paddings
     * in Conv and transpose in Matmul. However, the input and output shapes are
     * not taken into consideration.
     */
    virtual vector<int> getOpAttrVector() const { IT_TODO_HALT(); }
    /**
     * @brief Besides operator attributes, the returned vector includes input
     * and output shapes.
     */
    virtual vector<int> getWorkloadVector() const { IT_TODO_HALT(); }

    void addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
    void addSuccessors(const Operator &op) { successors.emplace_back(op); }
    void removePredecessors(const Operator &op);
    void removeSuccessors(const Operator &op);
    void replaceInput(Tensor t1, Tensor t2);
};

#define OP_CLONE(OpObj)                                                        \
    virtual Operator clone(const TensorVec &newInputs,                         \
                           const TensorVec &newOutputs) const override {       \
        auto op = infini::make_ref<OpObj>(*this);                              \
        op->inputs = newInputs;                                                \
        op->outputs = newOutputs;                                              \
        op->predecessors.clear();                                              \
        op->successors.clear();                                                \
        IT_ASSERT(op->checkValid(nullptr));                                    \
        return op;                                                             \
    }

} // namespace infini

namespace std {
template <> struct hash<infini::OpPerfKey> {
    size_t operator()(const infini::OpPerfKey &key) const { return key.hash; }
};
} // namespace std
