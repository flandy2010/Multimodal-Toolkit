import os
import time
import torch
import ffmpeg
import numpy as np
from tqdm import tqdm
from inference_pytorch.transnetv2_pytorch import TransNetV2
from training.visualization_utils import visualize_scenes
from training.video_utils import get_frames


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
state_dict = torch.load("./inference_pytorch/transnetv2-pytorch-weights.pth")

# 自动检测设备
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")

model.load_state_dict(state_dict)
model.to(device)
model.eval()


def predictions_to_scenes(predictions, threshold=0.5):

    probs = predictions[0, :, 0]
    is_transition = (probs > threshold).astype(np.uint8)

    # 找到转场开始的位置
    changes = np.diff(is_transition, prepend=0)
    trans_starts = np.where(changes == 1)[0]

    scene_list = []
    prev_scene_start = 0
    num_frames = len(probs)

    for start in trans_starts:
        # --- 核心修改处 ---
        # 如果觉得偏前，就把结束点往后推一帧
        # 原来是 start - 1，现在改为 start
        current_scene_end = start

        if current_scene_end >= prev_scene_start:
            scene_list.append((prev_scene_start, current_scene_end))

        # 下一段镜头的起点相应地往后推一帧
        prev_scene_start = current_scene_end + 1

    # 处理最后一段
    if prev_scene_start < num_frames:
        scene_list.append((prev_scene_start, num_frames - 1))
    elif prev_scene_start == num_frames:
        # 如果最后一段正好结束在最后一帧，补救一下防止丢掉最后一帧
        pass

    return scene_list



if __name__ == "__main__":

    base_dir = "../../data/RAIDataset/videos"

    for file_name in os.listdir(base_dir):

        file_path = os.path.join(base_dir, file_name)

        # 使用ffmpeg加载模型
        start_time = time.time()
        print(f"Loading video from: {file_path}")

        video = get_frames(file_path, width=48, height=27)
        video_tensor = torch.from_numpy(video)

        end_time = time.time()
        print(f"Loading video took {end_time - start_time:.2f} seconds")

        # 1 x video frames x frame height x frame width x RGB (not BGR) channels
        video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(device)

        chunk_size = 500  # 每个窗口的大小
        overlap = 50  # 重叠的帧数 (例如 50 帧)
        step = chunk_size - overlap  # 实际移动的步长

        num_frames = video_tensor.shape[1]

        # 创建一个全零张量来存储所有帧的预测值 (logits)
        # 同时也创建一个计数器，记录每帧被预测了多少次（用于重叠部分取平均）
        all_logits = torch.zeros((1, num_frames, 1), device="cpu")
        count_mask = torch.zeros((1, num_frames, 1), device="cpu")

        with torch.no_grad():
            # 使用 step 作为步长进行循环
            for i in tqdm(range(0, num_frames, step)):
                start = i
                end = min(i + chunk_size, num_frames)

                # 切片处理
                chunk = video_tensor[:, start:end].to(device)

                # 推理，注意TransNetV2 返回的是 Logits
                single_frame_pred, _ = model(chunk)

                # 将预测结果累加到对应位置
                all_logits[:, start:end] += single_frame_pred.cpu()
                # 记录这些帧被覆盖了一次
                count_mask[:, start:end] += 1

                # 如果已经处理到了视频末尾，提前跳出防止无限循环
                if end == num_frames:
                    break

        # 对重叠区域取平均值：总和 / 覆盖次数
        # 这样处理比直接覆盖（Overwrite）更平滑
        avg_logits = all_logits / count_mask

        # 之后再进行 Sigmoid 激活得到概率
        single_frame_pred = torch.sigmoid(avg_logits).numpy()

        # list[(left, right)]，表示识别出的场景的(左边界, 右边界)
        scenes = predictions_to_scenes(single_frame_pred, threshold=0.9)
        print(scenes)

        result_img = visualize_scenes(video_tensor[0].cpu().numpy(), scenes)
        result_img.save(f"./img/visualization_ret_{file_name.rsplit('.', 1)[0]}.png")

        break

