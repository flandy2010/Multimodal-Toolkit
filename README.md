# Multimodal-Toolkit

多模块大模型数据清洗工具集合！

本项目设立初衷为，希望在学习各类多模态大模型的过程中，将多模态大模型的数据清洗流程和工具进行一个总结。
考虑到部分多模态大模型的数据清洗也依赖于微调后的模型，只能尽力而为吧！

# 通用工具
TODO

# LPM 1.0: Video-based Character Performance Model

发布日期：2026-04-09 \
论文链接：[arvix](https://arxiv.org/pdf/2604.07823)

## 数据清洗流程

TODO

## 数据清洗工具

### 视频筛选

[2026-06-09] 场景切割工具TransNetV2，自带模型参数。原始仓库地址: [github](https://github.com/soCzech/TransNetV2/)，本仓库使用实录: [script](https://github.com/flandy2010/Multimodal-Toolkit/blob/main/scene_detection/TransNetV2/README_ch.md) \
[2026-06-10] 物体检测工具Yolov9，自带模型参数。原始仓库地址: [github](https://github.com/WongKinYiu/yolov9)，本仓库使用实录: [script](https://github.com/flandy2010/Multimodal-Toolkit/blob/main/object_detection/yolov9/README_ch.md) \
[2026-06-12] 质量检测工具FineVQ，HF公开模型参数。原始仓库地址: [github](https://github.com/IntMeGroup/FineVQ)，本仓库使用实录: [script](https://github.com/flandy2010/Multimodal-Toolkit/blob/main/quality_test/FineVQ/README_ch.md) \

### 说话人检测
[TODO] 说话人检测工具Light-ASD，HF存在公开模型参数。原始仓库地址：[github](https://github.com/Junhua-Liao/Light-ASD)，本仓库使用实录: [script](https://github.com/flandy2010/Multimodal-Toolkit/blob/main/speaker_detection/Light-ASD/README_ch.md) \
[TODO] 说话人检测工具TalkNet-ASD，HF存在公开模型参数。原始仓库地址：[github](https://github.com/TaoRuijie/TalkNet-ASD/)
