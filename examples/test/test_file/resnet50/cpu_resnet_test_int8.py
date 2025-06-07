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
import time

with open("../../imagenet/data.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

folder2idx = {}
for idx, v in data.items():
    folder2idx[v[0]] = int(idx)

onnx_model = onnx.load("../../models/resnet50.onnx")
model = OnnxStub(onnx_model, backend.cpu_runtime(), True)

my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

correct = 0
wrong = 0
total_time = 0

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

model.calibrate(calibrate_data)
# model.int_mixed_quantize(1.1, 0.16, calibrate_data)
# model.int_mixed_quantize(1.1, 0.02, calibrate_data)
model.int_mixed_quantize_cpu(1.1, 0.01, calibrate_data)

imagenet_dir = '../../imagenet'
folders = [f for f in os.listdir(imagenet_dir) if os.path.isdir(os.path.join(imagenet_dir, f))]

for folder in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(imagenet_dir, folder)
    if folder not in folder2idx:
        print(f"Warning: {folder} not found in data.json, skip.")
        continue
    idx = folder2idx[folder]
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled_imgs = random.sample(imgs, 1)
    for img_name in sampled_imgs:
        img_path = os.path.join(folder_path, img_name)
        # img = Image.open(img_path).resize((224, 224))
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_tensor = my_preprocess(img)
        img_tensor = np.expand_dims(img_tensor, 0)
        next(iter(model.inputs.values())).copyin_numpy(img_tensor)
        start = time.perf_counter()
        model.run()
        end = time.perf_counter()
        total_time += (end - start)
        outputs = next(iter(model.outputs.values())).copyout_numpy()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs, (1, 1000))
        _, predicted = torch.max(outputs, 1)
        res = int(predicted.item())
        if res == idx:
            correct += 1
        else:
            wrong += 1

print(f"平均耗时: {total_time / (correct + wrong):.6f} 秒")
