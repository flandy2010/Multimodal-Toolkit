import os
import time
import torch
import ffmpeg
import numpy as np
from tqdm import tqdm
from transnetv2_pytorch import TransNetV2

import sys
sys.path.append("/Users/wuxiuyu.wxy/Desktop/代码库/MultimodelTool/TransNetV2/")
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


# 新建模型
model = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth")

# 自动检测设备
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")

model.load_state_dict(state_dict)
model.to(device)
model.eval()


def load_video(video_path):

    print(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 1. 颜色空间转换：BGR (OpenCV默认) -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. 缩放到模型要求的尺寸：48x27
        frame = cv2.resize(frame, (48, 27))
        frames.append(frame)

    cap.release()

    # 3. 转换为 numpy 数组并归一化到 [0, 1]
    video_array = np.array(frames, dtype=np.float32)
    video_tensor = torch.from_numpy(video_array)
    video_tensor = video_tensor.round().clamp(0, 255).to(torch.uint8)

    # 4. 转换为 Torch Tensor
    # TransNet V2 要求的输入维度通常是 [Frames, Height, Width, Channels]
    # 在 PyTorch 模型中可能需要调整为 [1, Frames, Channels, Height, Width] 取决于具体实现
    return video_tensor

def load_video_by_ffmpeg(video_path, width=48, height=27):

    print(f"Loading video from: {video_path}")
    video_stream, err = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    video_tensor = torch.from_numpy(video)
    return video_tensor


def predictions_to_scenes(predictions, threshold=0.5):
    # 1. 预处理：降维并根据阈值二值化
    # predictions shape: [batch_size, frames, 1] -> [frames]
    probs = predictions[0, :, 0]
    is_transition = (probs > threshold).astype(np.uint8)

    # 2. 找到转场的起始和结束位置
    # np.diff 会计算 [i+1] - [i]，结果中：
    #  1 表示从 0 变到 1（转场开始）
    # -1 表示从 1 变到 0（转场结束）
    changes = np.diff(is_transition, prepend=0, append=0)
    trans_starts = np.where(changes == 1)[0]
    trans_ends = np.where(changes == -1)[0]

    # 3. 根据转场点构建镜头区间 (Scenes)
    # 镜头就是两个转场点之间的部分
    scene_list = []
    prev_scene_start = 0

    for start, end in zip(trans_starts, trans_ends):
        # 如果当前转场起始点大于上一段的起点，则记录这一段镜头
        if start > prev_scene_start:
            scene_list.append((prev_scene_start, start - 1))
        # 下一段镜头的起点是当前转场结束后的那一帧
        prev_scene_start = end

    # 4. 处理最后一段镜头（直到视频结束）
    num_frames = len(probs)
    if prev_scene_start < num_frames:
        scene_list.append((prev_scene_start, num_frames - 1))

    return scene_list



if __name__ == "__main__":

    # visualize()
    # raise

    base_dir = "/Users/wuxiuyu.wxy/Desktop/数据库/MultimodelAI/RAIDataset/videos"
    for file_name in os.listdir(base_dir):

        file_path = os.path.join(base_dir, file_name)

        start_time = time.time()
        video_tensor = load_video_by_ffmpeg(file_path)
        end_time = time.time()
        print(f"Loading video took {end_time - start_time:.2f} seconds")

        # 1 x video frames x frame height x frame width x RGB (not BGR) channels
        video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(device)

        chunk_size = 500  # 每组处理 500 帧
        num_frames = video_tensor.shape[1]
        all_single_frame_predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, num_frames, chunk_size)):
                # 切片：[1, i:i+chunk_size, 27, 48, 3]
                chunk = video_tensor[:, i:i + chunk_size].to(device)

                # 推理
                single_frame_pred, _ = model(chunk)

                # 将结果转回 CPU 存储
                all_single_frame_predictions.append(single_frame_pred.cpu())
                # if i > 2500:
                #     break

        # 最后合并结果
        # batch_size x video_frames x 1, 表示该帧是镜头切换点（中心）的概率
        single_frame_pred = torch.cat(all_single_frame_predictions, dim=1)
        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()

        scenes = predictions_to_scenes(single_frame_pred, threshold=0.9)

        print(scenes)

        result_img = visualize_scenes(video_tensor[0].cpu().numpy(), scenes)
        result_img.save(f"debug_visualization_{file_name}.png")

        break

