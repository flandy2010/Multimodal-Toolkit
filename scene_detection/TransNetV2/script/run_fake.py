import os
import torch
import numpy as np
from inference_pytorch.transnetv2_pytorch import TransNetV2
from training.visualization_utils import visualize_scenes


# 修复PIL的输出问题
from PIL import ImageDraw
original_rectangle = ImageDraw.ImageDraw.rectangle

def smart_rectangle(self, xy, fill=None, outline=None, width=1):
    # 提取坐标
    if isinstance(xy, list) and len(xy) == 2:
        (x0, y0), (x1, y1) = xy
        # 核心逻辑：确保 x1 >= x0, y1 >= y0
        fixed_xy = [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
    else:
        fixed_xy = xy
    return original_rectangle(self, fixed_xy, fill=fill, outline=outline, width=width)

ImageDraw.ImageDraw.rectangle = smart_rectangle


def test_model():

    print(f">>> 测试加载和调用模型")

    model = TransNetV2()
    state_dict = torch.load("./inference_pytorch/transnetv2-pytorch-weights.pth")

    device = torch.device("cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():

        # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
        input_video = input_video.to(device)
        single_frame_pred, all_frame_pred = model(input_video)

        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

    print(f"[INFO] input_shape: {input_video.shape}")
    print(f"[INFO] single_frame_pred: {single_frame_pred.shape}")
    print(f"[INFO] all_frame_pred: {all_frame_pred.shape}")


def test_visualize():
    print(f">>> 测试可视化工具")

    # 1. 模拟视频数据: 100 帧, 高 64, 宽 64, 3通道 (RGB)
    nf, ih, iw, ic = 100, 64, 64, 3
    fake_frames = np.zeros((nf, ih, iw, ic), dtype=np.uint8)

    # 2. 模拟场景标注:
    # 场景1: 0-19帧 (共20帧)
    # 场景2: 20-59帧 (共40帧)
    # 场景3: 60-99帧 (共40帧)
    fake_scenes = np.array([
        [0, 19],
        [20, 59],
        [60, 99],
    ], dtype=np.int32)

    # 预定义一组颜色 (RGB)
    color_palette = [
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
    ]

    # 3. 在每个片段中间画正方形
    square_size = 20  # 正方形边长
    # 计算中心位置的起始和结束坐标
    y1 = (ih - square_size) // 2
    y2 = y1 + square_size
    x1 = (iw - square_size) // 2
    x2 = x1 + square_size

    for i, (start, end) in enumerate(fake_scenes):
        duration = end - start + 1

        # 挑选颜色：每个片段换一个颜色
        color = color_palette[i % len(color_palette)]

        # 给该片段内的所有帧的中心位置涂上对应颜色
        # fake_frames[start:end+1] 选中帧范围，[:, y1:y2, x1:x2] 选中空间的正方形区域
        fake_frames[start:end + 1, y1:y2, x1:x2] = color

        print(f"片段 {i}: 帧范围 {start}-{end}, 长度 {duration}, 颜色 {color}")

    # 5. 保存结果
    output_dir = "./img"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "visualization_demo.png")

    # 剔除[60, 79]部分，发现可视化工具会对不在scenes中的部分加上灰色阴影
    fake_scenes = np.array([
        [0, 19],
        [20, 59],
        [80, 99],
    ], dtype=np.int32)

    result_img = visualize_scenes(fake_frames, fake_scenes)
    result_img.save(output_path)

    print(f"可视化图片已生成: {os.path.abspath(output_path)}")


if __name__ == "__main__":

    test_model()
    test_visualize()