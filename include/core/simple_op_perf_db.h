#pragma once
#include "core/common.h"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace infini {

enum class DeviceType { CPU, CUDA };

// 简单的性能记录结构
struct SimpleOpPerfRecord {
    std::string opType;    // 算子类型名称 (如 "Conv", "MatMul", "Relu")
    std::string dataType;  // 数据类型 (如 "FP32", "FP16", "INT8")
    DeviceType device;     // 设备类型 (CPU 或 CUDA)
    float executionTimeMus; // 执行时间(微秒)
};

// 简单的性能数据库类
class SimpleOpPerfDB {
  public:
    // 单例模式
    static SimpleOpPerfDB &getInstance() {
        static SimpleOpPerfDB instance;
        return instance;
    }

    // 添加一条性能记录
    void addRecord(const std::string &opType, const std::string &dataType,
                   DeviceType device, float timeMus);

    // 获取指定算子和数据类型的性能记录
    SimpleOpPerfRecord getRecord(const std::string &opType,
                                 const std::string &dataType,
                                 DeviceType device);

    // 获取所有记录
    std::vector<SimpleOpPerfRecord> getAllRecords() const { return records; }

    // 保存到文件
    void saveToFile(const std::string &filePath);

    // 从文件加载
    void loadFromFile(const std::string &filePath);

    // 判断是否应该量化
    bool shouldQuantize(const std::string &opType, DeviceType device);

    // 打印所有记录
    void printAllRecords() const;

  private:
    SimpleOpPerfDB() = default;
    ~SimpleOpPerfDB() = default;
    SimpleOpPerfDB(const SimpleOpPerfDB &) = delete;
    SimpleOpPerfDB &operator=(const SimpleOpPerfDB &) = delete;

    std::vector<SimpleOpPerfRecord> records;
};

// 辅助函数 - 将DeviceType转换为字符串
std::string deviceTypeToString(DeviceType device);

// 辅助函数 - 将字符串转换为DeviceType
DeviceType stringToDeviceType(const std::string &str);

} // namespace infini
