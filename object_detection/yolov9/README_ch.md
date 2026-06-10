# YOLOv9

原始论文链接：[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

原始代码仓库：[github](https://github.com/WongKinYiu/yolov9)

[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2402.13616-B31B1B.svg)](https://arxiv.org/abs/2402.13616)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kadirnar/Yolov9)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/merve/yolov9)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov9-object-detection-on-custom-dataset.ipynb)
[![OpenCV](https://img.shields.io/badge/OpenCV-BlogPost-black?logo=opencv&labelColor=blue&color=black)](https://learnopencv.com/yolov9-advancing-the-yolo-legacy/)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>


## 工具性能

MS COCO

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-T**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) | 640 | **38.3%** | **53.1%** | **41.3%** | **2.0M** | **7.7G** |
| [**YOLOv9-S**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt) | 640 | **46.8%** | **63.4%** | **50.7%** | **7.1M** | **26.4G** |
| [**YOLOv9-M**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt) | 640 | **51.4%** | **68.1%** | **56.1%** | **20.0M** | **76.3G** |
| [**YOLOv9-C**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) | 640 | **53.0%** | **70.2%** | **57.8%** | **25.3M** | **102.1G** |
| [**YOLOv9-E**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) | 640 | **55.6%** | **72.8%** | **60.6%** | **57.3M** | **189.0G** |
<!-- | [**YOLOv9 (ReLU)**]() | 640 | **51.9%** | **69.1%** | **56.5%** | **25.3M** | **102.1G** | -->

<!-- tiny, small, and medium models will be released after the paper be accepted and published. -->

# 环境配置
python版本使用3.8，环境直接使用配置好的requirements.txt
```shell
conda create -n py38Yolov9 python=3.8
pip install -r requirements.txt
```

# 数据下载
测试过程先偷懒直接使用案例里面的小马图片吧

# 使用介绍

## 样例数据测试
```shell
# GPU版本
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c-converted.pt' --name yolov9_c_c_640_detect

# CPU版本
python detect.py --source './data/images/horses.jpg' --img 640 --device cpu --weights './yolov9-c-converted.pt' --name yolov9_c_c_640_detect
```
模型会输出运行耗时以及输出路径，并输出小马的物体识别结果
```text
gelan-c summary: 387 layers, 25288768 parameters, 64944 gradients, 102.1 GFLOPs
image 1/1 Your-Base-Dir/Multimodal-Toolkit/object_detection/yolov9/data/images/horses.jpg: 448x640 5 horses, 97.4ms
Speed: 0.3ms pre-process, 97.4ms inference, 0.5ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/yolov9_c_c_640_detect
```
![object_detect_ret](runs/detect/yolov9_c_c_640_detect/horses.jpg)

## 真实数据调用

# 踩坑记录