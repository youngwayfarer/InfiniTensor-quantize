#include "core/runtime.h"
#include "core/blob.h"
#include "core/common.h"
#include "core/kernel.h"
#include "core/op_type.h"
#include "core/perf_engine.h"
#include "core/simple_op_perf_db.h"
#include "utils/data_generator.h"
#include <chrono>
#include <cstring>
#include <string>
namespace infini {
DeviceEnum toInfiniopDevice(Device device) {
    switch (device) {
    case Device::CPU:
        return DevCpu;
    case Device::CUDA:
        return DevNvGpu;
    case Device::BANG:
        return DevCambriconMlu;
    default:
        IT_TODO_HALT();
    };
}

void CpuRuntimeObj::run(const Graph &graph, bool tune, bool profiling, bool calibrate, bool loss_pass, bool quantized) const {
    // std::cout << "========================" << std::endl;

    //* test
    // graph->printQuantizedOps();

    graph->set_run_time(0);
    IT_ASSERT(graph->get_run_time() == 0);

    double r_time = 0;

    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        //* calibrate, record range
        if (calibrate) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (graph->inQuantizedOps(op->getOpType())) {
                for (auto input : op->getInputs()) {
                    if (input->isWeight() && input->getRange() != 0) {
                        continue;
                    }
                    input->updateRange();
                }
            }
        }

        if (loss_pass) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (graph->inQuantizedOps(op->getOpType())) {
                op->computeLoss();
            }
        }

        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        // If no record and disable tuning, run with the default argument
        if (!perfData && !tune) {
            auto start = std::chrono::high_resolution_clock::now();
            kernel->compute(op, this);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double kernel_time = duration.count() / 1000000.0;
            r_time += kernel_time;

            //* calibrate
            if (calibrate) {
                // if (op->getOpType() == OpType::Conv ||
                //     op->getOpType() == OpType::Mul) {
                if (graph->inQuantizedOps(op->getOpType())) {
                    for (auto output : op->getOutputs()) {
                        output->updateRange();
                    }
                }
            }

            // if (!calibrate && !loss_pass && op->getOpType() == OpType::Conv) {
            //     std::cout << "conv" << std::endl;
            //     std::cout << "datatype: " << op->getDType().toString()
            //               << std::endl;
            //     std::cout << "op, kernel time: " << duration.count()
            //               << std::endl;
            //     std::cout << std::endl;
            // }

            continue;
        }

        // TODO: The copy of record should be eliminated
        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            // TODO: record is not used
            // printf("no record data\n");
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        kernel->computeFuncTune(perfKey, op, record, this);
        ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);

        if (!profiling) {
            funcPtr(op, record, this);
            continue;
        } else {
            double t =
                timeit([&]() { funcPtr(op, record, this); }, []() {}, 1, 1);
            op->print();
            printf(" op_time %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }

    graph->set_run_time(r_time);

    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
}

void CpuRuntimeObj::simulatedRun(const Graph &graph) const {
    graph->setConvPass(false);
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        // if (op->getOpType() == OpType::Conv || op->getOpType() ==
        // OpType::Mul) {
        if (graph->inQuantizedOps(op->getOpType())) {
            op->simulatedInput();
        }

        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);

        kernel->compute(op, this);
    }
}

void CpuRuntimeObj::record_perf_pass(const Graph &graph) const {
    graph->setConvPass(false);
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfDb = SimpleOpPerfDB::getInstance();

    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";

    std::ifstream testFile(perfFilePath);
    if (testFile.good()) {
        testFile.close();
        std::cout << "正在加载已有的性能数据..." << std::endl;
        perfDb.loadFromFile(perfFilePath);
    }

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);

        auto start = std::chrono::high_resolution_clock::now();

        kernel->compute(op, this);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::string opType = op->getOpType().toString();
        std::string dataType = op->getDType().toString();
        DeviceType deviceType =
            (device == Device::CPU) ? DeviceType::CPU : DeviceType::CUDA;
        float timeMus = static_cast<float>(duration.count());
        perfDb.addRecord(opType, dataType, deviceType, timeMus);
    }

    perfDb.saveToFile(perfFilePath);
    std::cout
        << "性能测试完成，结果已保存到~/InfiniTensor/InfiniTensor/perf.json文件"
        << std::endl;
}

double RuntimeObj::getPerfTime(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            // TODO: should tenosrs automatically allocate when access data?
            // allocate memory for empty tensors and release it after profiling
            TensorVec allocatedTensors;
            for (auto t : op->getInputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : op->getOutputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : allocatedTensors) {
                t->dataMalloc();
                t->setData(IncrementalGenerator());
            }

            // Profile operators and record the results
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);

            // Free allocated memory
            for (auto t : allocatedTensors)
                t->freeData();
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;
        if (profiling) {
            op->print();
            printf(" op_time %lf\n", t);
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
    return totalTime;
}

void RuntimeObj::printProfilingData(double totalTime,
                                    const std::map<OpType, double> &opTime,
                                    const std::map<OpType, int> &opCnt) const {
    printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %7.3f %7.1f %7.3f\n", type.toString(), opCnt.at(type),
               t, t / totalTime * 100, t / opCnt.at(type));
    }
}

Blob RuntimeObj::allocBlob(size_t size) {
    return make_ref<BlobObj>(shared_from_this(), alloc(size));
}

void RuntimeObj::copyBlob(const TensorObj *dst, const TensorObj *src) const {
    void *dstPtr = dst->getRawDataPtr<void *>();
    void *srcPtr = src->getRawDataPtr<void *>();
    size_t bytes = dst->getBytes();
    auto dstRuntime = dst->getRuntime();
    auto srcRuntime = src->getRuntime();

    if (dstRuntime.get() == srcRuntime.get()) {
        dstRuntime->copyBlobInsideRuntime(dstPtr, srcPtr, bytes);
    } else if (src->getRuntime()->isCpu()) {
        dstRuntime->copyBlobFromCPU(dstPtr, srcPtr, bytes);
    } else if (dst->getRuntime()->isCpu()) {
        srcRuntime->copyBlobToCPU(dstPtr, srcPtr, bytes);
    } else
        IT_TODO_HALT();
}

void RuntimeObj::simulatedRun(const Graph &graph) const {
    IT_TODO_HALT_MSG(
        "only in cudaRuntime and CpuRuntime, not implemented here");
}

void RuntimeObj::conv_pass(const Graph &graph) const {
    IT_TODO_HALT_MSG(
        "only in cudaRuntime and CpuRuntime, not implemented here");
}

void RuntimeObj::record_perf_pass(const Graph &graph) const {
    IT_TODO_HALT_MSG(
        "only in cudaRuntime and CpuRuntime, not implemented here");
}

void CpuRuntimeObj::copyBlobFromCPU(void *dst, const void *src,
                                    size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobToCPU(void *dst, const void *src,
                                  size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobInsideRuntime(void *dst, const void *src,
                                          size_t bytes) const {
    memcpy(dst, src, bytes);
}

string NativeCpuRuntimeObj::toString() const { return "CPU Runtime"; }

} // namespace infini
