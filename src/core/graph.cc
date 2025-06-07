#include "core/graph.h"
#include "core/common.h"
#include "core/data_type.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "core/simple_op_perf_db.h"
#include "core/tensor.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "utils/data_convert.h"
// #include <__clang_cuda_math.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime), sorted(false) {
    map<UidBaseType, Tensor> tensorPool;
    // Clone tensors
    for (const auto &op : ops_in) {
        for (const auto &t : op->getInputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
        for (const auto &t : op->getOutputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
    }
    // Clone operators and add connections
    for (const auto &op : ops_in) {
        TensorVec inputs, outputs;
        for (const auto &t : op->getInputs()) {
            if (t) {
                inputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }

        for (const auto &t : op->getOutputs()) {
            if (t) {
                outputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }
        addOperatorAndConnect(op->clone(inputs, outputs));
    }
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

void GraphObj::optimize() {
    for (auto &op : ops) {
        switch (op->getOpType().underlying()) {
        default:
            break;
        }
    }
}

// mixed precision quantization
void GraphObj::mixed_quantize() {
    // topological sort
    IT_ASSERT(topo_sort() == true);

    auto &perfDb = SimpleOpPerfDB::getInstance();
    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";

    std::ifstream testFile(perfFilePath);
    if (testFile.good()) {
        testFile.close();
        perfDb.loadFromFile(perfFilePath);
    }

    int op_num = ops.size();
    for (int i = 0; i < op_num; i++) {
        auto op = ops[i];
        if (op->getOpType() == OpType::Resize || op->getOpType() == OpType::BatchNormalization) {
            // only consider: input[0] x
            // input
            auto input = op->getInputs(0);
            if (input->getDType() == DataType::Float16) {
                auto input_fp32 = addTensor(input->getDims(), DataType::Float32);
                input_fp32->dataMalloc();

                auto pred = input->getSource();
                if (pred) {
                    pred->removeSuccessors(op);
                    op->removePredecessors(pred);
                }

                op->replaceInput(input, input_fp32);
                input_fp32->addTarget(op);
                input->removeTarget(op);

                addOpWithOutputs<CastObj>(input, input_fp32,
                                          CastType::Float162Float);
            }

            // output
            auto output = op->getOutput(0);
            if (output->getDType() == DataType::Float16) {
                auto output_fp32 =
                    addTensor(output->getDims(), DataType::Float32);
                output_fp32->dataMalloc();

                output->setSource(nullptr);
                op->outputs[0] = output_fp32;
                output_fp32->setSource(op);

                for (auto target : output->getTargets()) {
                    target->replaceInput(output, output_fp32);
                    output_fp32->addTarget(target);
                    output->removeTarget(target);
                }

                // remove the old output
                removeTensor(output);
            }
        }

        else if (!(op->getOpType() == OpType::Cast)) {
            // 如果是最后一层（最后一个算子），使用fp32
            if (i == op_num - 1) {
                // input
                int input_num = op->numInputs();
                for (int j = 0; j < input_num; j++) {
                    auto input = op->getInputs(j);
                    if (input->getDType() == DataType::Float16) {
                        auto input_fp32 =
                            addTensor(input->getDims(), DataType::Float32);
                        input_fp32->dataMalloc();

                        // 为了便于确定权重，我们在此设定，权重cast得到的张量仍为权重
                        // 这里认为模型不会只有一层
                        if (input->isWeight()) {
                            input_fp32->setWeight();
                        }

                        auto pred = input->getSource();
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }

                        op->replaceInput(input, input_fp32);
                        input_fp32->addTarget(op);
                        input->removeTarget(op);

                        auto cast = addOpWithOutputs<CastObj>(
                            input, input_fp32, CastType::Float162Float);
                    }
                }

                // output
                int output_num = op->numOutputs();
                // 一般只有一个输出
                IT_ASSERT(output_num == 1);
                for (int j = 0; j < output_num; j++) {
                    auto output = op->getOutput(j);
                    if (output->getDType() == DataType::Float16) {
                        auto output_fp32 =
                            addTensor(output->getDims(), DataType::Float32);
                        output_fp32->dataMalloc();

                        // 为新的输出设置为output
                        if (output->isOutput()) {
                            output_fp32->setOutput();
                        }

                        output->setSource(nullptr);
                        op->outputs[j] = output_fp32;
                        output_fp32->setSource(op);

                        for (auto target : output->getTargets()) {
                            target->replaceInput(output, output_fp32);
                            output_fp32->addTarget(target);
                            output->removeTarget(target);
                        }

                        // remove the old output
                        removeTensor(output);
                    }
                }

                continue;
            }


            SimpleOpPerfRecord fp16Record = perfDb.getRecord(
                op->getOpType().toString(), "Float16", DeviceType::CUDA);
            SimpleOpPerfRecord fp32Record = perfDb.getRecord(
                op->getOpType().toString(), "Float32", DeviceType::CUDA);
            bool bothRecordsExist = (fp16Record.executionTimeMus > 0) &&
                                    (fp32Record.executionTimeMus > 0);

            if (bothRecordsExist &&
                (fp16Record.executionTimeMus > fp32Record.executionTimeMus)) {
                // 如果fp16的性能比fp32差，直接使用fp32
                std::cout << "fp16 performance is worse than fp32, use fp32"
                          << std::endl;
                continue;
            }

            // input
            //* 分情况
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (op->getOpType() == OpType::Conv) {
                int input_num = op->numInputs();
                for (int j = 0; j < input_num; j++) {
                    auto input = op->getInputs(j);
                    if (input->getDType() == DataType::Float32) {
                        auto input_fp16 =
                            addTensor(input->getDims(), DataType::Float16);
                        input_fp16->dataMalloc();

                        // 为了便于确定权重，我们在此设定，权重cast得到的张量仍为权重
                        if (input->isWeight()) {
                            input_fp16->setWeight();
                        }

                        // 输入得到的仍为输入
                        if (input->isInput()) {
                            input_fp16->setInput();
                        }

                        auto pred = input->getSource();
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }

                        op->replaceInput(input, input_fp16);
                        input_fp16->addTarget(op);
                        input->removeTarget(op);

                        auto cast = addOpWithOutputs<CastObj>(
                            input, input_fp16, CastType::Float2Float16);
                    }
                }
            } else {
                // fp32 权重直接换成 fp16
                int input_num = op->numInputs();
                for (int j = 0; j < input_num; j++) {
                    auto input = op->getInputs(j);
                    if (input->getDType() == DataType::Float32) {
                        if (input->isWeight()) {
                            // fp32 权重直接换成 fp16
                            auto input_fp16 =
                                addTensor(input->getDims(), DataType::Float16);
                            input_fp16->dataMalloc();
                            input_fp16->setWeight();

                            // 权重没有source
                            auto pred = input->getSource();
                            IT_ASSERT(!pred);

                            op->replaceInput(input, input_fp16);
                            input_fp16->addTarget(op);
                            input->removeTarget(op);
                            input_fp16->setSource(nullptr);

                            // 把权重中的值拷贝到fp16中
                            std::vector<float> data_fp32 =
                                input->copyout<float>();
                            std::vector<uint16_t> data_fp16(data_fp32.size());
                            for (size_t k = 0; k < data_fp32.size(); k++) {
                                data_fp16[k] = float_to_fp16(data_fp32[k]);
                            }
                            input_fp16->copyin(data_fp16);

                            removeTensor(input);
                        } else {
                            // 非权重正常处理
                            auto input_fp16 =
                                addTensor(input->getDims(), DataType::Float16);
                            input_fp16->dataMalloc();

                            if (input->isInput()) {
                                input_fp16->setInput();
                            }

                            auto pred = input->getSource();
                            if (pred) {
                                pred->removeSuccessors(op);
                                op->removePredecessors(pred);
                            }

                            op->replaceInput(input, input_fp16);
                            input_fp16->addTarget(op);
                            input->removeTarget(op);

                            auto cast = addOpWithOutputs<CastObj>(
                                input, input_fp16, CastType::Float2Float16);
                        }
                    }
                }
            }

            // output
            //* 输出处理相同
            int output_num = op->numOutputs();
            for (int j = 0; j < output_num; j++) {
                auto output = op->getOutput(j);
                if (output->getDType() == DataType::Float32) {
                    auto output_fp16 =
                        addTensor(output->getDims(), DataType::Float16);
                    output_fp16->dataMalloc();

                    output->setSource(nullptr);
                    op->outputs[j] = output_fp16;
                    output_fp16->setSource(op);

                    for (auto target : output->getTargets()) {
                        target->replaceInput(output, output_fp16);
                        output_fp16->addTarget(target);
                        output->removeTarget(target);
                    }
                    // remove the old output
                    removeTensor(output);
                }
            }
        }
    }

    //! important
    for (auto op : ops) {
        op->opDesc = nullptr;
        op->initInfiniOp(runtime);
    }

    IT_ASSERT(topo_sort() == true);
}

// int8 mixed precision quantization
void GraphObj::int_quantize() {
    // topological sort
    IT_ASSERT(topo_sort() == true);

    output_calibrate();
    setIntQuantized();

    auto &perfDb = SimpleOpPerfDB::getInstance();
    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";

    std::ifstream testFile(perfFilePath);
    if (testFile.good()) {
        testFile.close();
        perfDb.loadFromFile(perfFilePath);
    }

    int op_num = ops.size();
    for (int i = 0; i < op_num; i++) {
        auto op = ops[i];
        // if (op->getOpType() == OpType::Conv || op->getOpType() == OpType::Mul) {
        if (op->getOpType() == OpType::Conv) {
        // if (inQuantizedOps(op->getOpType())) {
            //* 可能有两个卷积，input_a是同一个张量，而这两个卷积一个量化，一个不量化
            //* 如果只检测input_a，那么本该不量化的卷积也会量化
            //* 并且，由于setBestConfig函数中跳过了量化参数的设定，导致结果出现错误
            //* 因此这里使用更严谨的判断，输入和权重都需要isQuantized
            //* 是否只检测权重即可，因为权重只有一个target

            SimpleOpPerfRecord int8Record = perfDb.getRecord(
                op->getOpType().toString(), "Int8", DeviceType::CUDA);
            std::string dataType = op->getDType().toString();
            SimpleOpPerfRecord fpRecord = perfDb.getRecord(
                op->getOpType().toString(), dataType, DeviceType::CUDA);
            bool bothRecordsExist = (int8Record.executionTimeMus > 0) &&
                                    (fpRecord.executionTimeMus > 0);

            if (bothRecordsExist &&
                (int8Record.executionTimeMus > fpRecord.executionTimeMus)) {
                // 如果int8的性能比fp16/fp32差，直接使用fp16/fp32
                std::cout << "int8 performance is worse than fp16/fp32, use fp16/fp32"
                          << std::endl;
                continue;
            }

            Tensor input_a = op->getInputs(0);
            Tensor input_w = op->getInputs(1);
            if (!input_a->isQuantized() || !input_w->isQuantized()) {
                // 不量化到int8
                continue;
            }

            // input
            int input_num = op->numInputs();
            IT_ASSERT(input_num == 2);
            for (int j = 0; j < input_num; j++) {
                auto input = op->getInputs(j);
                if (input->getDType() == DataType::Float32) {
                    if (input->isWeight()) {
                        // 直接将权重替换成处理好的张量，减小运行时开销
                        // 创建时张量形状为NCHW， 后面再转换成NHWC
                        auto input_int8_NHWC =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8_NHWC->dataMalloc();
                        input_int8_NHWC->setWeight();

                        input_int8_NHWC->setQuantized(true);
                        input_int8_NHWC->setRange(input->getRange());
                        input_int8_NHWC->setBitsWidth(input->getBitsWidth());
                        input_int8_NHWC->setScale(input->getScale());

                        // 权重没有source
                        auto pred = input->getSource();
                        IT_ASSERT(!pred);

                        op->replaceInput(input, input_int8_NHWC);
                        input_int8_NHWC->addTarget(op);
                        input->removeTarget(op);
                        input->setSource(nullptr);

                        // 把权重中的值拷贝到int8中
                        std::vector<float> data_fp32 = input->copyout<float>();
                        float s = input->getScale();
                        std::vector<int8_t> data_int8(data_fp32.size());
                        for (size_t k = 0; k < data_fp32.size(); k++) {
                            float temp = data_fp32[k] / s;
                            // round
                            int t = static_cast<int>(std::rint(temp));
                            // clip
                            // 这里选择quantized_range还是127
                            // 与cast kernel中保持一致，127
                            t = std::max(-127, std::min(127, t));
                            data_int8[k] = static_cast<int8_t>(t);
                        }
                        input_int8_NHWC->copyin(data_int8);
                        // 这里转换数据布局
                        input_int8_NHWC->shape_NCHW2NHWC();
                        input_int8_NHWC->data_NCHW2NHWC();

                        removeTensor(input);
                    } else {
                        // fp32 -> int8
                        auto input_int8 =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8->dataMalloc();
                        Shape nchw_shape = input->getDims();
                        Shape nhwc_shape = {nchw_shape[0], nchw_shape[2],
                                            nchw_shape[3], nchw_shape[1]};
                        auto input_int8_NHWC =
                            addTensor(nhwc_shape, DataType::Int8);
                        input_int8_NHWC->dataMalloc();

                        if (input->isInput()) {
                            input_int8->setInput();
                            input_int8_NHWC->setInput();
                        }

                        //! 是否有必要
                        input_int8->setQuantized(true);
                        input_int8->setRange(input->getRange());
                        input_int8->setBitsWidth(input->getBitsWidth());
                        input_int8->setScale(input->getScale());
                        input_int8_NHWC->setQuantized(true);
                        input_int8_NHWC->setRange(input->getRange());
                        input_int8_NHWC->setBitsWidth(input->getBitsWidth());
                        input_int8_NHWC->setScale(input->getScale());

                        auto pred = input->getSource();
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }

                        op->replaceInput(input, input_int8_NHWC);
                        input_int8_NHWC->addTarget(op);
                        input->removeTarget(op);

                        auto cast = addOpWithOutputs<CastObj>(
                            input, input_int8, CastType::Float2Int8);
                        auto trans = addOpWithOutputs<TransposeObj>(
                            input_int8, input_int8_NHWC, Shape{0, 2, 3, 1});
                    }
                } else if (input->getDType() == DataType::Float16) {
                    if (input->isWeight()) {
                        // 直接将权重替换成处理好的张量，减小运行时开销
                        // 创建时张量形状为NCHW， 后面再转换成NHWC
                        auto input_int8_NHWC =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8_NHWC->dataMalloc();
                        input_int8_NHWC->setWeight();

                        input_int8_NHWC->setQuantized(true);
                        input_int8_NHWC->setRange(input->getRange());
                        input_int8_NHWC->setBitsWidth(input->getBitsWidth());
                        input_int8_NHWC->setScale(input->getScale());

                        // 权重fp16， 前面一定有一个cast(fp32 -> fp16)
                        auto pred = input->getSource();
                        IT_ASSERT(pred->getOpType() == OpType::Cast);
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }
                        auto input_fp32 = pred->getInputs(0);

                        op->replaceInput(input, input_int8_NHWC);
                        input_int8_NHWC->addTarget(op);
                        input->removeTarget(op);
                        input->setSource(nullptr);

                        // 把权重中的值拷贝到int8中
                        std::vector<float> data_fp32 =
                            input_fp32->copyout<float>();
                        float s = input->getScale();
                        std::vector<int8_t> data_int8(data_fp32.size());
                        for (size_t k = 0; k < data_fp32.size(); k++) {
                            float temp = data_fp32[k] / s;
                            // round
                            int t = static_cast<int>(std::rint(temp));
                            // clip
                            // 这里选择quantized_range还是127
                            // 与cast kernel中保持一致，127
                            t = std::max(-127, std::min(127, t));
                            data_int8[k] = static_cast<int8_t>(t);
                        }
                        input_int8_NHWC->copyin(data_int8);
                        // 这里转换数据布局
                        input_int8_NHWC->shape_NCHW2NHWC();
                        input_int8_NHWC->data_NCHW2NHWC();

                        removeTensor(input);
                        removeTensor(input_fp32);
                        removeOperator(pred);
                        i--;
                        op_num--;
                    } else {
                        // fp16 -> int8
                        auto input_int8 =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8->dataMalloc();
                        Shape nchw_shape = input->getDims();
                        Shape nhwc_shape = {nchw_shape[0], nchw_shape[2],
                                            nchw_shape[3], nchw_shape[1]};
                        auto input_int8_NHWC =
                            addTensor(nhwc_shape, DataType::Int8);
                        input_int8_NHWC->dataMalloc();

                        if (input->isInput()) {
                            input_int8->setInput();
                            input_int8_NHWC->setInput();
                        }

                        input_int8->setQuantized(true);
                        input_int8->setRange(input->getRange());
                        input_int8->setBitsWidth(input->getBitsWidth());
                        input_int8->setScale(input->getScale());
                        input_int8_NHWC->setQuantized(true);
                        input_int8_NHWC->setRange(input->getRange());
                        input_int8_NHWC->setBitsWidth(input->getBitsWidth());
                        input_int8_NHWC->setScale(input->getScale());

                        auto pred = input->getSource();
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }

                        op->replaceInput(input, input_int8_NHWC);
                        input_int8_NHWC->addTarget(op);
                        input->removeTarget(op);

                        addOpWithOutputs<CastObj>(input, input_int8,
                                                  CastType::Float162Int8);
                        addOpWithOutputs<TransposeObj>(
                            input_int8, input_int8_NHWC, Shape{0, 2, 3, 1});
                    }
                } else {
                    IT_TODO_HALT_MSG("only support fp32 and fp16 in int_quantize for input on gpu");
                }
            }

            // output
            int output_num = op->numOutputs();
            IT_ASSERT(output_num == 1);
            for (int j = 0; j < output_num; j++) {
                auto output = op->getOutput(j);
                if (output->getDType() == DataType::Float32) {
                    // int32 -> cast -> fp32
                    auto output_int32 =
                        addTensor(output->getDims(), DataType::Int32);
                    output_int32->dataMalloc();
                    Shape nchw_shape = output->getDims();
                    Shape nhwc_shape = {nchw_shape[0], nchw_shape[2],
                                        nchw_shape[3], nchw_shape[1]};
                    auto output_int32_NHWC =
                        addTensor(nhwc_shape, DataType::Int32);
                    output_int32_NHWC->dataMalloc();

                    if (output->isOutput()) {
                        output_int32->setOutput();
                        output_int32_NHWC->setOutput();
                    }

                    output_int32->setQuantized(true);
                    output_int32->setRange(output->getRange());
                    int bit = output->getBitsWidth();
                    IT_ASSERT(bit == 8);
                    output_int32->setBitsWidth(bit);
                    output_int32->setScale(output->getScale());
                    float output_s = output->getOutputScale();
                    IT_ASSERT(output_s != 0);
                    output_int32->setOutputScale(output_s);
                    output_int32_NHWC->setQuantized(true);
                    output_int32_NHWC->setRange(output->getRange());
                    output_int32_NHWC->setBitsWidth(bit);
                    output_int32_NHWC->setScale(output->getScale());
                    output_int32_NHWC->setOutputScale(output_s);

                    // output->setSource(nullptr);
                    // op->outputs[j] = output_int32;
                    // output_int32->setSource(op);
                    output->setSource(nullptr);
                    op->outputs[j] = output_int32_NHWC;
                    output_int32_NHWC->setSource(op);

                    for (auto target : output->getTargets()) {
                        if (target) {
                            op->removeSuccessors(target);
                            target->removePredecessors(op);
                        }
                    }

                    addOpWithOutputs<TransposeObj>(output_int32_NHWC, output_int32, Shape{0, 3, 1, 2});
                    addOpWithOutputs<CastObj>(output_int32, output,
                                              CastType::Int322Float);
                } else if (output->getDType() == DataType::Float16) {
                    // int32 -> cast -> fp16
                    auto output_int32 =
                        addTensor(output->getDims(), DataType::Int32);
                    output_int32->dataMalloc();
                    Shape nchw_shape = output->getDims();
                    Shape nhwc_shape = {nchw_shape[0], nchw_shape[2],
                                        nchw_shape[3], nchw_shape[1]};
                    auto output_int32_NHWC =
                        addTensor(nhwc_shape, DataType::Int32);
                    output_int32_NHWC->dataMalloc();

                    if (output->isOutput()) {
                        output_int32->setOutput();
                        output_int32_NHWC->setOutput();
                    }

                    output_int32->setQuantized(true);
                    output_int32->setRange(output->getRange());
                    int bit = output->getBitsWidth();
                    IT_ASSERT(bit == 8);
                    output_int32->setBitsWidth(bit);
                    output_int32->setScale(output->getScale());
                    float output_s = output->getOutputScale();
                    IT_ASSERT(output_s != 0);
                    output_int32->setOutputScale(output_s);
                    output_int32_NHWC->setQuantized(true);
                    output_int32_NHWC->setRange(output->getRange());
                    output_int32_NHWC->setBitsWidth(bit);
                    output_int32_NHWC->setScale(output->getScale());
                    output_int32_NHWC->setOutputScale(output_s);

                    output->setSource(nullptr);
                    op->outputs[j] = output_int32_NHWC;
                    output_int32_NHWC->setSource(op);

                    for (auto target : output->getTargets()) {
                        if (target) {
                            op->removeSuccessors(target);
                            target->removePredecessors(op);
                        }
                    }

                    addOpWithOutputs<TransposeObj>(
                        output_int32_NHWC, output_int32, Shape{0, 3, 1, 2});
                    addOpWithOutputs<CastObj>(output_int32, output,
                                              CastType::Int322Float16);
                } else {
                    IT_TODO_HALT_MSG("only support fp32 and fp16 in int_quantize for output");
                }
            }
        }
    }

    //! important
    for (auto op : ops) {
        op->opDesc = nullptr;
        op->initInfiniOp(runtime);
    }

    IT_ASSERT(topo_sort() == true);
}


void GraphObj::int_quantize_cpu() {
    IT_ASSERT(topo_sort() == true);

    output_calibrate();

    auto &perfDb = SimpleOpPerfDB::getInstance();
    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";

    std::ifstream testFile(perfFilePath);
    if (testFile.good()) {
        testFile.close();
        perfDb.loadFromFile(perfFilePath);
    }

    int op_num = ops.size();
    for (int i = 0; i < op_num; i++) {
        auto op = ops[i];
        // if (op->getOpType() == OpType::Conv || op->getOpType() == OpType::Mul) {
        if (op->getOpType() == OpType::Conv) {
        // if (inQuantizedOps(op->getOpType())) {
            SimpleOpPerfRecord int8Record = perfDb.getRecord(
                op->getOpType().toString(), "Int8", DeviceType::CPU);
            SimpleOpPerfRecord fp32Record = perfDb.getRecord(
                op->getOpType().toString(), "Float32", DeviceType::CPU);
            bool bothRecordsExist = (int8Record.executionTimeMus > 0) &&
                                    (fp32Record.executionTimeMus > 0);

            if (bothRecordsExist &&
                (int8Record.executionTimeMus > fp32Record.executionTimeMus)) {
                // 如果int8的性能比fp32差，直接使用fp32
                std::cout << "int8 performance is worse than fp32, use fp32"
                          << std::endl;
                continue;
            }

            Tensor input_a = op->getInputs(0);
            Tensor input_w = op->getInputs(1);
            if (!input_a->isQuantized() || !input_w->isQuantized()) {
                continue;
            }

            // input
            int input_num = op->numInputs();
            IT_ASSERT(input_num == 2);
            for (int j = 0; j < input_num; j++) {
                // 输入一定是float32， CPU
                auto input = op->getInputs(j);
                if (input->getDType() == DataType::Float32) {
                    if (input->isWeight()) {
                        // 直接将权重替换成处理好的张量，减小运行时开销
                        auto input_int8 =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8->dataMalloc();
                        input_int8->setWeight();
                        input_int8->setQuantized(true);
                        input_int8->setRange(input->getRange());
                        input_int8->setBitsWidth(input->getBitsWidth());
                        input_int8->setScale(input->getScale());

                        // 权重没有source
                        auto pred = input->getSource();
                        IT_ASSERT(!pred);

                        op->replaceInput(input, input_int8);
                        input_int8->addTarget(op);
                        input->removeTarget(op);
                        input->setSource(nullptr);

                        // 把权重中的值拷贝到int8中
                        std::vector<float> data_fp32 = input->copyout<float>();
                        float s = input->getScale();
                        std::vector<int8_t> data_int8(data_fp32.size());
                        for (size_t k = 0; k < data_fp32.size(); k++) {
                            float temp = data_fp32[k] / s;
                            // round
                            int t = static_cast<int>(std::rint(temp));
                            // clip
                            t = std::max(-127, std::min(127, t));
                            data_int8[k] = static_cast<int8_t>(t);
                        }
                        input_int8->copyin(data_int8);

                        removeTensor(input);
                    } else {
                        // fp32 -> int8
                        auto input_int8 =
                            addTensor(input->getDims(), DataType::Int8);
                        input_int8->dataMalloc();

                        if (input->isInput()) {
                            input_int8->setInput();
                        }

                        input_int8->setQuantized(true);
                        input_int8->setRange(input->getRange());
                        input_int8->setBitsWidth(input->getBitsWidth());
                        input_int8->setScale(input->getScale());

                        auto pred = input->getSource();
                        if (pred) {
                            pred->removeSuccessors(op);
                            op->removePredecessors(pred);
                        }

                        op->replaceInput(input, input_int8);
                        input_int8->addTarget(op);
                        input->removeTarget(op);

                        addOpWithOutputs<CastObj>(input, input_int8,
                                                  CastType::Float2Int8);
                    }
                } else {
                    IT_TODO_HALT_MSG("only support fp32 in int_quantize for input on cpu");
                }
            }

            // output
            int output_num = op->numOutputs();
            IT_ASSERT(output_num == 1);
            for (int j = 0; j < output_num; j++) {
                auto output = op->getOutput(j);
                if (output->getDType() == DataType::Float32) {
                    // int32 -> cast -> fp32
                    auto output_int32 =
                        addTensor(output->getDims(), DataType::Int32);
                    output_int32->dataMalloc();

                    if (output->isOutput()) {
                        output_int32->setOutput();
                    }

                    output_int32->setQuantized(true);
                    output_int32->setRange(output->getRange());
                    int bit = output->getBitsWidth();
                    IT_ASSERT(bit == 8);
                    output_int32->setBitsWidth(bit);
                    output_int32->setScale(output->getScale());
                    float output_s = output->getOutputScale();
                    IT_ASSERT(output_s != 0);
                    output_int32->setOutputScale(output_s);

                    output->setSource(nullptr);
                    op->outputs[j] = output_int32;
                    output_int32->setSource(op);

                    for (auto target : output->getTargets()) {
                        if (target) {
                            op->removeSuccessors(target);
                            target->removePredecessors(op);
                        }
                    }

                    addOpWithOutputs<CastObj>(output_int32, output,
                                              CastType::Int322Float);
                } else {
                    IT_TODO_HALT_MSG("only support fp32 in int_quantize for output on cpu");
                }
            }
        }
    }

    //! important
    for (auto op : ops) {
        op->opDesc = nullptr;
        op->initInfiniOp(runtime);
    }

    IT_ASSERT(topo_sort() == true);
}


void GraphObj::print_data() {
    for (auto tensor :tensors) {
        tensor->print();
        std::cout << "data below: " << std::endl;
        if (tensor->getDType() == DataType::Float32) {
            vector<float> data = tensor->copyout<float>();
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
        } else if (tensor->getDType() == DataType::Int8) {
            vector<int8_t> data = tensor->copyout<int8_t>();
            for (auto i : data) {
                std::cout << (int)i << " ";
            }
            std::cout << std::endl;
        } else if (tensor->getDType() == DataType::Int32) {
            vector<int32_t> data = tensor->copyout<int32_t>();
            for (auto i : data) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }
}

void GraphObj::output_calibrate() {
    for (auto op : ops) {
        // if (op->getOpType() == OpType::Conv || op->getOpType() ==
        // OpType::Mul) {
        if (inQuantizedOps(op->getOpType())) {
            int output_num = op->numOutputs();
            IT_ASSERT(output_num == 1);
            for (int i = 0; i < output_num; i++) {
                auto output = op->getOutput(i);
                IT_ASSERT(output->getRange() != 0);
                output->setBitsWidth(8);
                float i_scale = op->getInputs(0)->getScale();
                float w_scale = op->getInputs(1)->getScale();
                output->setOutputScale(i_scale * w_scale);
            }
        }
    }
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc(bool useNaiveAllocator, size_t memPoolSize) {
    // topological sorting first

    IT_ASSERT(topo_sort() == true);
    if (useNaiveAllocator) {
        // can not set memory pool when use naive allocator
        IT_ASSERT(memPoolSize == 0);
        // used for debugging memory out-of-bounds access, tensors will not be
        // released correctly
        // note: behavior may not match running in non-naive mode, and it may
        // not reproduce the bug
        for (auto &tensor : tensors) {
            if (!tensor->isWeight() ||
                (tensor->isWeight() && !weightAllocated)) {
                tensor->dataMalloc();
            }
        }
        return;
    }
    if (memPoolSize > 0) {
        allocator.setMemPool(memPoolSize);
    }
    // count the number of times all tensors are used
    std::unordered_map<TensorObj *, size_t> tensorToRefCount;
    // record the memory address offsets of all tensors to be allocated
    std::unordered_map<TensorObj *, size_t> tensorToOffset;

    // reinit allocator
    allocator.init();

    // record all weight tensors, including weight tensors and kvcache
    // tensors
    std::unordered_set<TensorObj *> weightTensors;
    for (auto &tensor : tensors) {
        if (tensor->isWeight()) {
            // allocate memory for all weight tensors first, and this memory
            // will not be freed until the graph is destroyed
            weightTensors.insert(tensor.get());
            if (!this->weightAllocated) {
                tensorToOffset[tensor.get()] =
                    allocator.allocWeight(tensor->getBytes());
            }
        } else if (tensor->isInput() || tensor->isOutput()) {
            // allocate memory for all input and output tensors, and this memory
            // will not be reused later
            tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
        } else {
            tensorToRefCount[tensor.get()] = tensor->getTargets().size();
            // allocate memory for all user-created tensors
            if (tensor.get()->getSource() == nullptr) {
                tensorToOffset[tensor.get()] =
                    allocator.alloc(tensor->getBytes());
            }
        }
    }
    // if memory has not yet been allocated for weight tensors,
    // allocate memory now and do not allocate again in the future.
    if (!this->weightAllocated) {
        this->weightAllocated = true;
        // only allocate once for weight tensors
        for (auto &tensor : weightTensors) {
            IT_ASSERT(tensorToOffset.find(tensor) != tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getWeightPtr()) +
                    tensorToOffset[tensor]));
        }
    }
    // traverse in topological order and simulate memory allocation
    for (auto &op : ops) {
        // memory should be allocated for the op's output first
        auto outputs = op->getOutputs();
        for (auto &tensor : outputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    tensorToOffset[tensor.get()] =
                        allocator.alloc(tensor->getBytes());
                }
            }
        }
        auto inputs = op->getInputs();
        for (auto &tensor : inputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    auto tensorIter = tensorToRefCount.find(tensor.get());
                    IT_ASSERT(tensorIter != tensorToRefCount.end());
                    IT_ASSERT(tensorToRefCount[tensor.get()] > 0);
                    tensorToRefCount[tensor.get()] -= 1;
                    if (tensorToRefCount[tensor.get()] == 0) {
                        // indicate that this tensor will no longer be used and
                        // perform memory free
                        tensorToRefCount.erase(tensor.get());
                        allocator.free(tensorToOffset[tensor.get()],
                                       tensor->getBytes());
                    }
                }
            }
        }
    }

    // perform actual memory allocation for non-weight tensors
    for (auto &tensor : tensors) {
        if (!tensor->isWeight()) {
            IT_ASSERT(tensorToOffset.find(tensor.get()) !=
                      tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime, static_cast<uint8_t *>(allocator.getPtr()) +
                                     tensorToOffset[tensor.get()]));
        }
    }
}

Tensor GraphObj::cloneKV(Tensor &tensor) {
    auto obj = tensor->clone();
    if (allocator.getMemPoolStatus()) {
        if (tensor->hasData()) {
            obj->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getHeapPtr()) +
                    allocator.heapAlloc(tensor->getBytes())));
            obj->copyData(tensor);
        }
    } else {
        if (tensor->hasData()) {
            obj->dataMalloc();
            obj->copyData(tensor);
        }
    }
    return obj;
}

void GraphObj::freeHeap() { this->allocator.freeHeap(); }

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

OpVec GraphObj::getComputeOps() const {
    OpVec opList;
    for (auto op : ops)
        if (op->getOpType().isMatMulOrConv())
            opList.emplace_back(op);
    return opList;
}

void GraphObj::deleteConnection(Tensor tensor, Operator op) {
    // if op is target
    IT_ASSERT(std::find(tensor->getTargets().begin(),
                        tensor->getTargets().end(),
                        op) != tensor->getTargets().end());
    tensor->removeTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->removeSuccessors(op);
        op->removePredecessors(tensor->getSource());
    }
}

// add op as a target
void GraphObj::addConnection(Tensor tensor, Operator op) {
    tensor->addTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->addSuccessors(op);
        op->addPredecessors(tensor->getSource());
    }
}

void GraphObj::replaceConnection(Tensor oldTensor, Tensor newTensor,
                                 Operator op) {
    // op is a target of old tensor
    IT_ASSERT(std::find(oldTensor->getTargets().begin(),
                        oldTensor->getTargets().end(),
                        op) != oldTensor->getTargets().end());
    addConnection(newTensor, op);
    deleteConnection(oldTensor, op);
    op->replaceInput(oldTensor, newTensor);
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
