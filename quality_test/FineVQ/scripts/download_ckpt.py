import os
from huggingface_hub import snapshot_download

# 1. 设置镜像站（国内必做，否则 InternVL 这种大模型基本下不动）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 执行下载
# repo_id: 模型名字
# local_dir: 你想存到的本地目录
# resume_download: 断点续传，断网了重跑脚本就行
try:
    print("开始下载 FineVQ_score，请稍候...")
    snapshot_download(
        repo_id="IntMeGroup/FineVQ_score",
        local_dir="./ckpt/FineVQ_score",
        resume_download=True,
        max_workers=8  # 开启多线程下载
    )
    print("下载完成！模型已保存至 ./ckpt 目录")
except Exception as e:
    print(f"下载出错: {e}")