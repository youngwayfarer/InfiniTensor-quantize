import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from PIL import Image
import time
import os

def test_alpha(alpha: float, calibrate_data: list):
    onnx_model = onnx.load("../../models/inception_v3.onnx")
    # infinitensor
    model = OnnxStub(onnx_model, backend.cuda_runtime())
    model.calibrate(calibrate_data)
    mse = model.test_alpha(1.1, calibrate_data, alpha)
    return mse

if __name__ == '__main__':
    image_path = "../../figs/cat.png"
    img = Image.open(image_path).resize((299, 299))
    new_img_path = "../../figs/sheepdog.png"
    new_img = Image.open(new_img_path).resize((299, 299))
    new_new_img_path = "../../figs/lipstick.png"
    new_new_img = Image.open(new_new_img_path).resize((299, 299))

    # 处理图像并转成Tensor
    from torchvision import transforms

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    # numpy.ndarray
    img = np.expand_dims(img, 0)
    # torch_img = torch.from_numpy(img)

    new_img = my_preprocess(new_img)
    new_img = np.expand_dims(new_img, 0)
    
    new_new_img = my_preprocess(new_new_img)
    new_new_img = np.expand_dims(new_new_img, 0)

    # onnx_model = onnx.load("resnet18.onnx")
    # # infinitensor
    # model = OnnxStub(onnx_model, backend.cuda_runtime())
    # calibrate_data = [img, new_img]
    calibrate_data = [img, new_img, new_new_img]
    
    best_mse = 100000
    best_alpha = 0.5
    
    alphas = np.linspace(0.5, 0, 51)
    for alpha_val in alphas:
        mse = test_alpha(alpha_val, calibrate_data)
        print("alpha: ", alpha_val)
        print("mse", mse)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha_val
            print("update")

    print("best alpha: ", best_alpha)
    print("best_mse: ", best_mse)
    
    # model.test_alpha(1.1, calibrate_data, 0.5)
    # model.test_alpha(1.1, calibrate_data, 0.49)
    # model.test_alpha(1.1, calibrate_data, 0.3)
    
    
    # 检测simulated quantize结果，281， 230
    # 较好
    # res = model.set_best_alpha(0.5, calibrate_data)
    # for item in res:
    #     outputs = torch.tensor(item)
    #     outputs = torch.reshape(outputs, (1, 1000))
    #     _, predicted = torch.max(outputs, 1)
    #     print(predicted)
