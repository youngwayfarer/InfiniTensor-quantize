import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from PIL import Image

if __name__ == '__main__':
    image_path = "../../figs/cat.png"
    img = Image.open(image_path).resize((224, 224))
    new_img_path = "../../figs/sheepdog.png"
    new_img = Image.open(new_img_path).resize((224, 224))
    new_new_img_path = "../../figs/lipstick.png"
    new_new_img = Image.open(new_new_img_path).resize((224, 224))
    new_new_new_img_path = "../../figs/hotdog.png"
    new_new_new_img = Image.open(new_new_new_img_path).resize((224, 224))
    new_new_new_new_img_path = "../../figs/meerkat.png"
    new_new_new_new_img = Image.open(new_new_new_new_img_path).resize((224, 224))

    # 处理图像并转成Tensor
    from torchvision import transforms

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    new_new_new_img = my_preprocess(new_new_new_img)
    new_new_new_img = np.expand_dims(new_new_new_img, 0)
    new_new_new_new_img = my_preprocess(new_new_new_new_img)
    new_new_new_new_img = np.expand_dims(new_new_new_new_img, 0)

    onnx_model = onnx.load("../../models/resnet50.onnx")

    # infinitensor
    model = OnnxStub(onnx_model, backend.cuda_runtime(), True)
    # model = OnnxStub(onnx_model, backend.cuda_runtime())

    model.mixed_quantize()

    calibration_data = [img, new_img]
    model.calibrate(calibration_data)
    # model.int_mixed_quantize(1.1, 0.16, calibration_data)
    model.int_mixed_quantize(1.1, 0.01, calibration_data)

    next(iter(model.inputs.values())).copyin_numpy(img)
    # model.run()
    model.int_mixed_run()
    time1 = model.get_time()
    print(f"耗时: {time1:.6f} 秒")

    outputs = next(iter(model.outputs.values())).copyout_numpy()
    res1 = outputs
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print("--------------------------")
    print("res: ")
    print(predicted)
    print("--------------------------")
    
    next(iter(model.inputs.values())).copyin_numpy(new_img)
    # model.run()
    model.int_mixed_run()
    time1 = model.get_time()
    print(f"耗时: {time1:.6f} 秒")

    outputs = next(iter(model.outputs.values())).copyout_numpy()
    res1 = outputs
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print("--------------------------")
    print("res: ")
    print(predicted)
    print("--------------------------")

    next(iter(model.inputs.values())).copyin_numpy(new_new_img)
    # model.run()
    model.int_mixed_run()
    time1 = model.get_time()
    print(f"耗时: {time1:.6f} 秒")

    outputs = next(iter(model.outputs.values())).copyout_numpy()
    res1 = outputs
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print("--------------------------")
    print("res: ")
    print(predicted)
    print("--------------------------")

    next(iter(model.inputs.values())).copyin_numpy(new_new_new_img)
    # model.run()
    model.int_mixed_run()
    time1 = model.get_time()
    print(f"耗时: {time1:.6f} 秒")

    outputs = next(iter(model.outputs.values())).copyout_numpy()
    res1 = outputs
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print("--------------------------")
    print("res: ")
    print(predicted)
    print("--------------------------")

    next(iter(model.inputs.values())).copyin_numpy(new_new_new_new_img)
    # model.run()
    model.int_mixed_run()
    time1 = model.get_time()
    print(f"耗时: {time1:.6f} 秒")

    outputs = next(iter(model.outputs.values())).copyout_numpy()
    res1 = outputs
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print("--------------------------")
    print("res: ")
    print(predicted)
    print("--------------------------")
