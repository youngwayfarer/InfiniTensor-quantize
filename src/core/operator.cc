#include "core/operator.h"
#include "core/common.h"
#include "core/graph.h"
#include "core/hash.h"

namespace infini {

OperatorObj::OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)
    : type(opType), inputs(inputs), outputs(outputs) {
    if (opType != OpType::Recv) {
        for (const auto &t : inputs)
            IT_ASSERT(t);
    }
}

void OperatorObj::removePredecessors(const Operator &op) {
    for (auto it = predecessors.begin(); it != predecessors.end();) {
        if (it->lock() == op)
            it = predecessors.erase(it);
        else
            ++it;
    }
}

void OperatorObj::removeSuccessors(const Operator &op) {
    for (auto it = successors.begin(); it != successors.end();) {
        if (it->lock() == op)
            it = successors.erase(it);
        else
            ++it;
    }
}

// 暂时只考虑conv, mul
// 2个输入
void OperatorObj::computeLoss() {
    // weight, activation
    // 4~8
    // IT_ASSERT(alpha <= 0.5);
    IT_ASSERT(alpha <= 1);
    IT_ASSERT(alpha >= 0);
    for (int w = 8; w >= 4; w--) {
        for (int a = 8; a >= 4; a--) {
            auto input_w = getInputs(1);
            input_w->setBitsWidth(w);
            float loss_w = input_w->getSimulatedLoss();
            auto input_a = getInputs(0);
            input_a->setBitsWidth(a);
            float loss_a = input_a->getSimulatedLoss();

            float loss;
            if (input_w->isWeight()) {
                loss = (1 - alpha) * loss_w + alpha * loss_a;
            } else {
                loss = 0.5 * loss_w + 0.5 * loss_a;
            }

            std::pair<int, int> keyToFind = {w, a};
            auto it = bit_loss.find(keyToFind);
            if (it != bit_loss.end()) {
                it->second += loss;
            } else {
                bit_loss[keyToFind] = loss;
            }
        }
    }
    calibrateNumIncrement();
}

void OperatorObj::replaceInput(Tensor t1, Tensor t2) {
    for (auto itr = inputs.begin(); itr != inputs.end(); ++itr) {
        if (*itr == t1) {
            *itr = t2;
        }
    }
}

OpPerfKey OperatorObj::getOpPerfKey() const {
    auto workloadVector = getWorkloadVector();
    // Calculate hash of workload, i.e. hash with shape. This is different from
    // Operator::hash, which hashes operator attributes and ignores tensor
    // shapes.
    HashType hash = 0;
    hash = hashAppend(hash, type.underlying());
    hash = hashAppend(hash, hashVector(workloadVector));
    return OpPerfKey(hash, type, workloadVector);
}

HashType OperatorObj::hash() const {
    HashType hash = 0;
    hash = hashAppend(hash, type.underlying());
    hash = hashAppend(hash, hashVector(getOpAttrVector()));
    return hash;
}

bool OperatorObj::checkValid(GraphObj *graph) {
    auto optShapes = inferShape();
    if (!optShapes) { // shape inference failed
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Shape inference failed for " << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        return false;
    }

    const vector<Shape> &shapes = *optShapes;
    if (shapes.size() != outputs.size()) {
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Size not equal " << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "=====================" << std::endl;
        return false;
    }
    if (graph) { // if graph != nullptr, outputs should be created
        auto dataTypes = inferDataType();
        for (size_t i = 0; i < outputs.size(); i++) {
            IT_ASSERT(!outputs[i], "Find empty output while operator creation");
            outputs[i] = graph->addTensor(shapes[i], dataTypes[i]);
        }
        initInfiniOp(graph->getRuntime());
    } else { // if outputs have been created, check their shapes
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i] != outputs[i]->getDims()) {
                std::cout << "*********************" << std::endl;
                std::cout << "*********************" << std::endl;
                std::cout << "shapes[i]: " << std::endl;
                for (auto s : shapes[i]) {
                    std::cout << s << " ";
                }
                std::cout << std::endl;
                std::cout << "outputs[i]->getDims(): " << std::endl;
                for (auto s : outputs[i]->getDims()) {
                    std::cout << s << " ";
                }
                std::cout << std::endl;
                std::cout << "*********************" << std::endl;
                std::cout << "*********************" << std::endl;
                std::cout << "=====================" << std::endl;
                std::cout << "=====================" << std::endl;
                std::cout << "=====================" << std::endl;
                std::cout << "Size not equal 222 " << std::endl;
                std::cout << "=====================" << std::endl;
                std::cout << "=====================" << std::endl;
                std::cout << "=====================" << std::endl;
                return false;
            }
        }
    }

    return true;
}

void OperatorObj::initInfiniOp(const Runtime context) {
    // do nothing by default
    opDesc = nullptr;
}

optional<vector<Shape>> OperatorObj::inferShape() { return inferShape(inputs); }

vector<DataType> OperatorObj::inferDataType(const TensorVec &inputs) const {
    auto dataType = inputs[0]->getDType();
    return vector(numOutputs(), dataType);
}

vector<DataType> OperatorObj::inferDataType() const {
    return inferDataType(inputs);
}

} // namespace infini
