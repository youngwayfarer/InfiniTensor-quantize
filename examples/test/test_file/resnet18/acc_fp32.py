import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from PIL import Image
import os
import json
from torchvision import transforms
from tqdm import tqdm
import time

with open("../../imagenet/data.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

folder2idx = {}
for idx, v in data.items():
    folder2idx[v[0]] = int(idx)

onnx_model = onnx.load("../../models/resnet18.onnx")
model = OnnxStub(onnx_model, backend.cuda_runtime(), True)

my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

correct = 0
wrong = 0
time_time = 0

imagenet_dir = '../../imagenet'
val_folder = ['n01855672', 'n02091244', 'n02114548', 'n02138441', 'n02174001', 'n02950826', 'n02971356', 'n02981792', 'n03075370', 'n03417042', 'n03535780', 'n03584254', 'n03770439', 'n03773504', 'n03980874', 'n09256479']
# folders = [f for f in os.listdir(imagenet_dir) if os.path.isdir(os.path.join(imagenet_dir, f))]

for folder in tqdm(val_folder, desc="Processing validation folders"):
    folder_path = os.path.join(imagenet_dir, folder)
    if folder not in folder2idx:
        print(f"Warning: {folder} not found in data.json, skip.")
        continue
    idx = folder2idx[folder]
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(imgs, desc=f"Processing {folder}", leave=False):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_tensor = my_preprocess(img)
        img_tensor = np.expand_dims(img_tensor, 0)
        next(iter(model.inputs.values())).copyin_numpy(img_tensor)
        start = time.perf_counter()
        model.run()
        end = time.perf_counter()
        time_time += end - start
        outputs = next(iter(model.outputs.values())).copyout_numpy()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs, (1, 1000))
        _, predicted = torch.max(outputs, 1)
        res = int(predicted.item())
        if res == idx:
            correct += 1
        else:
            wrong += 1

print(f"分类正确数目: {correct}")
print(f"分类错误数目: {wrong}")
print(f"耗时: {time_time / (correct + wrong):.6f} 秒")

accuracy = correct / (correct + wrong) * 100
print(f"准确率: {accuracy:.2f}%")
