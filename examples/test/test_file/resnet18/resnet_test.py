import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from PIL import Image
import os
import json
import random
from torchvision import transforms
from tqdm import tqdm

with open("../../imagenet/data.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

folder2idx = {}
for idx, v in data.items():
    folder2idx[v[0]] = int(idx)

onnx_model = onnx.load("../../models/resnet18.onnx")
# origin
model1 = OnnxStub(onnx_model, backend.cuda_runtime(), True)
# mixed precision quantize
model2 = OnnxStub(onnx_model, backend.cuda_runtime(), True)

my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

correct1 = 0
wrong1 = 0
total_time1 = 0
correct2 = 0
wrong2 = 0
total_time2 = 0

# model2 calibration && quantization
cal_img_paths = [
    "../../figs/cat.png",
    "../../figs/sheepdog.png",
    "../../figs/lipstick.png",
    "../../figs/hotdog.png",
    "../../figs/meerkat.png"
]
calibrate_data = []
for path in cal_img_paths:
    img = Image.open(path).resize((224, 224))
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)
    calibrate_data.append(img)

model2.mixed_quantize()
model2.calibrate(calibrate_data)
model2.int_mixed_quantize(1.1, 0.02, calibrate_data)

imagenet_dir = '../../imagenet'
folders = [f for f in os.listdir(imagenet_dir) if os.path.isdir(os.path.join(imagenet_dir, f))]

for folder in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(imagenet_dir, folder)
    if folder not in folder2idx:
        print(f"Warning: {folder} not found in data.json, skip.")
        continue
    idx = folder2idx[folder]
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled_imgs = random.sample(imgs, 5)
    for img_name in sampled_imgs:
        img_path = os.path.join(folder_path, img_name)
        # img = Image.open(img_path).resize((224, 224))
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_tensor = my_preprocess(img)
        img_tensor = np.expand_dims(img_tensor, 0)

        # origin
        next(iter(model1.inputs.values())).copyin_numpy(img_tensor)
        model1.run()
        time = model1.get_time()
        total_time1 += time
        outputs = next(iter(model1.outputs.values())).copyout_numpy()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs, (1, 1000))
        _, predicted = torch.max(outputs, 1)
        res = int(predicted.item())
        if res == idx:
            correct1 += 1
        else:
            wrong1 += 1

        # mixed precision quantize
        next(iter(model2.inputs.values())).copyin_numpy(img_tensor)
        model2.int_mixed_run()
        time = model2.get_time()
        total_time2 += time
        outputs = next(iter(model2.outputs.values())).copyout_numpy()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs, (1, 1000))
        _, predicted = torch.max(outputs, 1)
        res = int(predicted.item())
        if res == idx:
            correct2 += 1
        else:
            wrong2 += 1

print("初始模型测试结果:")
print(f"分类正确数目: {correct1}")
print(f"分类错误数目: {wrong1}")
print(f"平均耗时: {total_time1 / 500:.6f} 秒")

print("量化模型测试结果:")
print(f"分类正确数目: {correct2}")
print(f"分类错误数目: {wrong2}")
print(f"平均耗时: {total_time2 / 500:.6f} 秒")
