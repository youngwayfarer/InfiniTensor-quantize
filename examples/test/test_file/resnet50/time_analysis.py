import re
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Process log file to calculate kernel time.")
# 添加文件路径参数
parser.add_argument("file_path", type=str, help="Path to the log file")

args = parser.parse_args()
time_dict = {}

# 使用 [\s\S]*? 确保跨行匹配
pattern = r"op type: (\w+)[\s\S]*?op, kernel time: (\d+)"

with open(args.file_path, "r") as file:
    content = file.read()  # 一次性读取整个文件（适用于大文件需优化）
    matches = re.findall(pattern, content)

for op_type, time_value in matches:
    time_dict[op_type] = time_dict.get(op_type, 0) + int(time_value)

# 输出结果（秒）
for op_type, time in time_dict.items():
    print(f"{op_type} total kernel time: {time / 1_000_000:.6f} seconds")

total_time = sum(time_dict.values())
print(f"Total kernel time: {total_time / 1_000_000:.6f} seconds")