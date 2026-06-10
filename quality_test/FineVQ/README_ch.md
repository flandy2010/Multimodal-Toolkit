<div align="center">
  
<h1>  FineVQ: Fine-Grained User Generated Content Video Quality Assessment (CVPR 2025 Highlight💡)

</div>

<div align="center">
  <div>
      <!-- <a href="https://arxiv.org/abs/2412.19238"><img src="https://arxiv.org/abs/2412.19238"/></a> -->
      <a href="https://arxiv.org/abs/2412.19238"><img src="https://img.shields.io/badge/Arxiv-2412.19238-red"/></a>
<a href="https://huggingface.co/datasets/IntMeGroup/FineVD">
   <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" alt="Hugging Face Dataset Badge"/>
</a>

<p align="center">
  <img width="1000" alt="Fine" src="https://github.com/user-attachments/assets/bca3c5c7-e448-4b25-ad26-92e9c8572402" />
</p>
</div>
</div>
---

原始论文链接: [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](https://openaccess.thecvf.com/content/CVPR2025/html/Duan_FineVQ_Fine-Grained_User_Generated_Content_Video_Quality_Assessment_CVPR_2025_paper.html):

代码仓库链接：[github](https://github.com/IntMeGroup/FineVQ)

# 工具性能

# 环境配置
```shell
conda create -n py39FineVQ python=3.9 -y
conda activate py39FineVQ

pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation # Mac电脑可能无法安装

# 如果pip install decord失败，可以注释掉requirements中的decord，然后尝试下述命令
pip install eva-decord
```

# 数据下载

# 使用介绍

## 模型参数下载
```shell
# 优先使用huggingface自带的命令行
huggingface-cli download OpenGVLab/InternVL2-8B --local_dir ./ckpt/InternVL2-8B

# 如果huggingface抽风的话，也可以使用模型python
python scripts/download_ckpt.py
```