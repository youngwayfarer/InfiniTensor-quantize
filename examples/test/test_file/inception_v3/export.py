import torch
import torchvision

model = torchvision.models.inception_v3(weights='DEFAULT')
model.eval()
dummy_input = torch.randn(1, 3, 299, 299)
torch.onnx.export(
    model,                      # 模型实例
    dummy_input,                # 示例输入
    "inception_v3.onnx",        # 输出文件名
    export_params=True,         # 导出模型参数
    verbose=True                # 打印导出过程
)

print("ONNX 模型已导出至: inception_v3.onnx")
