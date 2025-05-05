#include "cuda/cuda_runtime.h"
#include "core/common.h"
#include "core/kernel.h"
#include "core/op_type.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "core/simple_op_perf_db.h"
#include <chrono>
#include <cstddef>
#include <string>
#ifdef INFINI_USE_NCCL
#include "cuda/nccl_communicator.h"
#endif

void CHECK_CUDA_KERNEL_ERROR(infini::Operator op) {
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(kernelError)
                  << std::endl
                  << "Failed Operator: " << op->toString() << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace infini {
void CudaRuntimeObj::runWithoutSync(const Graph &graph, bool calibrate, bool loss_pass, bool quantized) const {
    // std::cout << "=====================" << std::endl;

    graph->setConvPass(false);
    graph->set_run_time(0);
    IT_ASSERT(graph->get_run_time() == 0);

    double r_time = 0;

    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type

        //* calibrate, record range
        if (calibrate) {
            // if (op->getOpType() == OpType::Conv || op->getOpType() == OpType::Mul) {
            if (graph->inQuantizedOps(op->getOpType())) {
                for (auto input : op->getInputs()) {
                    // 这里认为权重的range一定不为0，只更新一次
                    if (input->isWeight() && input->getRange() != 0) {
                        continue;
                    }
                    input->updateRange();
                    // std::cout << "calibrate: " << op->getOpType().toString()
                    //           << std::endl;
                    // std::cout << "input range: " << input->getRange();
                }
            }
        }

        //* loss pass
        //* for conv, mul: calculate the loss of op of (w, a)
        if (loss_pass) {
            // if (op->getOpType() == OpType::Conv ||
            //     op->getOpType() == OpType::Mul) {
            if (graph->inQuantizedOps(op->getOpType())) {
                op->computeLoss();
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        // IT_ASSERT(perfData, "No perf data for OP " + op->toString());
        if (perfData) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            funcPtr(op, perfData, this);
        } else {
            start = std::chrono::high_resolution_clock::now();
            kernel->compute(op, this);
            end = std::chrono::high_resolution_clock::now();
        }
        checkCudaError(cudaGetLastError()) << op->toString();

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double kernel_time = duration.count() / 1000000.0;
        r_time += kernel_time;


        if (calibrate) {
            // if (op->getOpType() == OpType::Conv || op->getOpType() == OpType::Mul) {
            if (graph->inQuantizedOps(op->getOpType())) {
                for (auto output : op->getOutputs()) {
                    output->updateRange();
                }
            }
        }

        // if (!calibrate && !loss_pass) {
        //     auto t = op->getOpType();
        //     std::cout << "op type: " << t.toString() << std::endl;
        //     if (t == OpType::Conv) {
        //         std::cout << "weight shape: " << op->getInputs(1)->getDims()[0]
        //                   << " " << op->getInputs(1)->getDims()[1] << " "
        //                   << op->getInputs(1)->getDims()[2] << " "
        //                   << op->getInputs(1)->getDims()[3] << std::endl;
        //     }
        //     std::cout << "datatype: " << op->getDType().toString() << std::endl;
        //     std::cout << "op, kernel time: " << duration.count() << std::endl;
        //     std::cout << std::endl;
        // }
    }

    graph->set_run_time(r_time);
}

void CudaRuntimeObj::simulatedRun(const Graph &graph) const {
    graph->setConvPass(false);
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        // if (op->getOpType() == OpType::Conv || op->getOpType() == OpType::Mul) {
        if (graph->inQuantizedOps(op->getOpType())) {
            op->simulatedInput();
        }

        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);

        kernel->compute(op, this);

        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::record_perf_pass(const Graph &graph) const {
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

        checkCudaError(cudaGetLastError()) << op->toString();
    }
    perfDb.saveToFile(perfFilePath);
    std::cout
        << "性能测试完成，结果已保存到~/InfiniTensor/InfiniTensor/perf.json文件"
        << std::endl;
}

void CudaRuntimeObj::conv_pass(const Graph &graph) const {
    graph->setConvPass(true);
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);

        kernel->compute(op, this);

        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::runWithCudaGraph(const Graph &graph) {
    if (!isCudaGraphCreated) {
        CUDAStream::createStream();
        checkCudnnError(cudnnSetStream(cudnn, CUDAStream::getCurrentStream()));
        checkCublasError(
            cublasSetStream(cublas, CUDAStream::getCurrentStream()));
        checkCudaError(cudaStreamBeginCapture(CUDAStream::getCurrentStream(),
                                              cudaStreamCaptureModeGlobal));
        runWithoutSync(graph);
        checkCudaError(
            cudaStreamEndCapture(CUDAStream::getCurrentStream(), &cudaGraph));
        checkCudaError(
            cudaGraphInstantiate(&cudaGraphInstance, cudaGraph, NULL, NULL, 0));
        isCudaGraphCreated = true;
    } else {
        checkCudaError(
            cudaGraphLaunch(cudaGraphInstance, CUDAStream::getCurrentStream()));
    }
    checkCudaError(cudaStreamSynchronize(CUDAStream::getCurrentStream()));
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;
        double t = record->time;
        totalTime += t;
        json j;

        kernel->computeFuncTune(perfKey, op, record, this);
        if (profiling) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            double t = timeit([&]() { funcPtr(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }

        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::run(const Graph &graph, bool runTune,
                         bool profiling, bool calibrate, bool loss_pass, bool quantized) const {
    if (profiling)
        IT_TODO_HALT();
    if (runTune)
        tune(graph, profiling);
    else
        runWithoutSync(graph, calibrate, loss_pass, quantized);
    sync();
}

void CudaRuntimeObj::sync() const { checkCudaError(cudaDeviceSynchronize()); }

string CudaRuntimeObj::toString() const { return "CUDA Runtime"; }

void CudaRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_NCCL
    comm = std::make_unique<NcclCommunicatorObj>(name, worldSize, rank);
#else
    IT_TODO_HALT_MSG("Not compiled with NCCL.");
#endif
}

cudaStream_t CUDAStream::_stream = 0;
} // namespace infini
