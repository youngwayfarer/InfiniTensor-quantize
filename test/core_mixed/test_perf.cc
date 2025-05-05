#include "core/simple_op_perf_db.h"
#include <gtest/gtest.h>
#include <sys/types.h>
#include <vector>

namespace infini {

void compareOperatorPerformance(const std::string &opType) {
    auto &perfDb = SimpleOpPerfDB::getInstance();

    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";

    // 加载性能数据
    perfDb.loadFromFile(perfFilePath);

    // 获取所有记录
    auto records = perfDb.getAllRecords();

    // 按设备和数据类型组织数据
    std::map<DeviceType, std::map<std::string, float>> performanceMap;

    // 筛选出指定算子类型的记录
    for (const auto &record : records) {
        if (record.opType == opType) {
            performanceMap[record.device][record.dataType] =
                record.executionTimeMus;
        }
    }

    if (performanceMap.empty()) {
        std::cout << "未找到算子 '" << opType << "' 的性能记录" << std::endl;
        return;
    }

    // 打印表头
    std::cout << "\n===== 算子 " << opType << " 性能比较 =====" << std::endl;
    std::cout << std::left << std::setw(10) << "设备";

    // 收集所有数据类型
    std::set<std::string> dataTypes;
    for (const auto &devicePair : performanceMap) {
        for (const auto &dataPair : devicePair.second) {
            dataTypes.insert(dataPair.first);
        }
    }

    // 打印数据类型列
    for (const auto &dataType : dataTypes) {
        std::cout << std::setw(12) << dataType;
    }
    std::cout << "\n"
              << std::string(10 + dataTypes.size() * 12, '-') << std::endl;

    // 打印每个设备的性能数据
    for (const auto &devicePair : performanceMap) {
        std::cout << std::left << std::setw(10)
                  << deviceTypeToString(devicePair.first);

        for (const auto &dataType : dataTypes) {
            if (devicePair.second.count(dataType)) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                          << devicePair.second.at(dataType);
            } else {
                std::cout << std::setw(12) << "N/A";
            }
        }
        std::cout << std::endl;
    }

    // 打印加速比(如果有CPU和CUDA数据)
    if (performanceMap.count(DeviceType::CPU) &&
        performanceMap.count(DeviceType::CUDA)) {
        std::cout << std::left << std::setw(10) << "CPU/CUDA";

        for (const auto &dataType : dataTypes) {
            if (performanceMap[DeviceType::CPU].count(dataType) &&
                performanceMap[DeviceType::CUDA].count(dataType) &&
                performanceMap[DeviceType::CUDA][dataType] > 0) {

                float speedup = performanceMap[DeviceType::CPU][dataType] /
                                performanceMap[DeviceType::CUDA][dataType];
                std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                          << speedup << "x";
            } else {
                std::cout << std::setw(12) << "N/A";
            }
        }
        std::cout << std::endl;
    }

    std::cout << std::string(10 + dataTypes.size() * 12, '-') << std::endl;
}

TEST(OpPerf, CompareAllOps) {
    auto &perfDb = SimpleOpPerfDB::getInstance();
    std::string perfFilePath =
        "/home/kerr1gan/InfiniTensor/InfiniTensor/perf.json";
    perfDb.loadFromFile(perfFilePath);

    // 获取所有记录
    auto records = perfDb.getAllRecords();

    // 收集所有算子类型
    std::set<std::string> opTypes;
    for (const auto &record : records) {
        opTypes.insert(record.opType);
    }

    // 对每种算子类型进行比较
    for (const auto &opType : opTypes) {
        compareOperatorPerformance(opType);
    }
}

} // namespace infini
