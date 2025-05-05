#include "core/simple_op_perf_db.h"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace infini {

std::string deviceTypeToString(DeviceType device) {
    switch (device) {
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::CUDA:
        return "CUDA";
    default:
        return "Unknown";
    }
}

DeviceType stringToDeviceType(const std::string &str) {
    if (str == "CPU")
        return DeviceType::CPU;
    if (str == "CUDA")
        return DeviceType::CUDA;
    // 默认返回CPU
    return DeviceType::CPU;
}

void SimpleOpPerfDB::addRecord(const std::string &opType,
                               const std::string &dataType, DeviceType device,
                               float timeMus) {
    // 检查是否已存在相同配置的记录
    for (auto &record : records) {
        if (record.opType == opType && record.dataType == dataType &&
            record.device == device) {
            // 更新现有记录
            record.executionTimeMus = record.executionTimeMus * 0.99f + timeMus * 0.01f;
            return;
        }
    }

    // 添加新记录
    records.push_back({opType, dataType, device, timeMus});
}

SimpleOpPerfRecord SimpleOpPerfDB::getRecord(const std::string &opType,
                                             const std::string &dataType,
                                             DeviceType device) {
    for (const auto &record : records) {
        if (record.opType == opType && record.dataType == dataType &&
            record.device == device) {
            return record;
        }
    }
    // 未找到记录时返回空记录
    return {"", "", DeviceType::CPU, 0.0f};
}

void SimpleOpPerfDB::saveToFile(const std::string &filePath) {
    try {
        nlohmann::json j = nlohmann::json::array();
        for (const auto &record : records) {
            nlohmann::json item;
            item["opType"] = record.opType;
            item["dataType"] = record.dataType;
            item["device"] = deviceTypeToString(record.device);
            item["executionTimeMus"] = record.executionTimeMus;
            j.push_back(item);
        }

        std::ofstream file(filePath);
        file << std::setw(4) << j << std::endl; // 格式化输出，缩进4个空格
        file.close();
        std::cout << "性能数据已保存到文件: " << filePath << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "保存性能数据时出错: " << e.what() << std::endl;
    }
}

void SimpleOpPerfDB::loadFromFile(const std::string &filePath) {
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filePath << std::endl;
            return;
        }

        nlohmann::json j;
        file >> j;

        records.clear();
        for (const auto &item : j) {
            SimpleOpPerfRecord record;
            record.opType = item["opType"];
            record.dataType = item["dataType"];
            record.device = stringToDeviceType(item["device"]);
            record.executionTimeMus = item["executionTimeMus"];
            records.push_back(record);
        }

        file.close();
    } catch (const std::exception &e) {
        std::cerr << "加载性能数据时出错: " << e.what() << std::endl;
    }
}

bool SimpleOpPerfDB::shouldQuantize(const std::string &opType,
                                    DeviceType device) {
    float fp32Time = 0.0f;
    float int8Time = 0.0f;

    // 查找FP32和INT8的性能记录
    for (const auto &record : records) {
        if (record.opType == opType && record.device == device) {
            if (record.dataType == "FP32") {
                fp32Time = record.executionTimeMus;
            } else if (record.dataType == "INT8") {
                int8Time = record.executionTimeMus;
            }
        }
    }

    // 如果INT8比FP32快一定比例(例如20%)，则建议量化
    if (fp32Time > 0 && int8Time > 0) {
        return (int8Time < fp32Time * 0.8);
    }

    return false; // 没有足够数据做决定
}

void SimpleOpPerfDB::printAllRecords() const {
    std::cout << "=== 算子性能记录 ===" << std::endl;
    std::cout << std::left << std::setw(15) << "算子类型" << std::setw(10)
              << "数据类型" << std::setw(10) << "设备" << std::setw(15)
              << "执行时间(ms)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (const auto &record : records) {
        std::cout << std::left << std::setw(15) << record.opType
                  << std::setw(10) << record.dataType << std::setw(10)
                  << deviceTypeToString(record.device) << std::setw(15)
                  << record.executionTimeMus << std::endl;
    }
    std::cout << std::string(50, '-') << std::endl;
}

} // namespace infini
