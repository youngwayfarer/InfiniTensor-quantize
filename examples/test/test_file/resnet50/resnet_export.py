import torch
import torchvision

model = torchvision.models.resnet50(weights='DEFAULT')
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,                      # 模型实例
    dummy_input,                # 示例输入
    "resnet50.onnx",            # 输出文件名
    export_params=True,         # 导出模型参数
    verbose=True                # 打印导出过程
)

print("ONNX 模型已导出至: resnet50.onnx")
