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
偷懒使用公共文件夹下面Data/images凑合一下吧！

# 使用介绍

## 模型参数下载
```shell
# 优先使用huggingface自带的命令行
huggingface-cli download IntMeGroup/FineVQ_score --local-dir ./IntMeGroup/FineVQ_score

# 如果huggingface抽风的话，也可以使用模型python
python scripts/download_ckpt.py
```

## 使用方法
```shell
python scripts/run.py --model_path ./ckpt/FineVQ_score --image_path ../../data/images/horses.jpg
# 在mac环境指定使用mps
# python scripts/run.py --model_path ./ckpt/FineVQ_score --image_path ../../data/images/horses.jpg --device mps
```
Demo采用的实际输入内容为模型参数地址和图片地址，模型工作流如下：
- 处理数据，把文本和图片数据分门别类处理好。
- 拼接成了固定的conversation模板
```python
# 拼接一下prompt用于评价图片（视频）质量
special_tokens = "\n".join([f"Frame{i+1}: <image>" for i in range(num_patches)])
special_tokens = special_tokens + "\nMotion Feature: <image>"
question = f"{special_tokens}\nHow would you rate the overall quality of this image?"
answer = "The overall quality of the image is excellent."

# 原因是模型实际上是选择了这段内容的某个token对应的logits经过MLP后作为分数的。
conversations = [
    {"from": "human", "value": question},
    {"from": "ai", "value": answer},
]
```
- 调用模型进行推理，并选择从后往前数固定位置的token对应的logits经过MLP后得到score
```python

class MLP(nn.Module):
    def __init__(self, input_dim=4096):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self._initialize_weights()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return x

class InternVLChatModel(PreTrainedModel):
        
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            pixel_values2: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            ...
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        ...

        last_hidden_states = outputs.hidden_states[-1]
        
        # 从last_hidden_states的固定位置选择input_tensor
        input_tensor = last_hidden_states[:,-4,:]
        if torch.isnan(last_hidden_states[:,-4,:]).any():
            print("Input contains NaN values!")
            input_tensor = torch.nan_to_num(last_hidden_states[:,-4,:], nan=0.0, posinf=1e9, neginf=-1e9)
        
        # 使用mlpscore根据input_tensor得到最终分数
        score1 = self.mlpscore(input_tensor)
        score1 = score1.squeeze(1)

        return {'score1':score1,
                'label': shift_labels,
                'logit': torch.argmax(shift_logits,dim=1)}
```

# 踩坑记录
由于没有linux环境，而mac环境下的mps/cpu不太支持avg_pool3d_out_frame的BFloat16，只能临时伪造一个motion_feature的结果了。
修改了`internvl/model/internvl_chat_infer/InternVLChatModel.py`内的相关代码。
```python
# 由于mac环境下的mps/cpu不太支持avg_pool3d_out_frame的BFloat16，暂时注释掉
# motion_feature = self.slowfast_model(inputs)
# motion_feature = motion_feature.view(B, -1)

motion_feature = torch.zeros((B, 2304)).to(input_embeds)
```

# 扩展阅读

##  图片信息是怎么被塞入LLM的输入中

核心：用图片信息替换占位符号

### 模型输入

- pixel_values，shape=[num_patches, 3, 448, 448]，表示这张图片全部的信息，其中根据这张图片本身的分辨率不同，拆份出的num_patches不同。
也可以认为是这张图片的切片，喂给ViT得到视觉embedding
- pixel_values2，shape=[32, 3, 448, 448]，是同一张图重复 32 次形成的时间序列。喂给SlowFast分支，来提取全局质量/运动特征。
- input_ids，扩展后的ids列表，已经包含了留给图片和运动特征的占位符
  - 最原始的结构是：`[{'from': 'human', 'value': '...'}, {'from': 'ai', 'value': '...'}]`
  - 转换成string：`<|im_start|>user\nFrame1: \<image>...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
  - 对视频special_token进行扩展，即：每个`\<image>`都会扩展成256个`\<image>`，然后tokenizer转化为ids序列。
```python
if not text_only:
    for conversation in conversations:
        for i in range(num_image):
            # 核心：根据列表里的数量生成一长串上下文 Token
            image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
            # 将第一个出现的 <image> 替换掉
            conversation = conversation.replace('<image>', image_tokens, 1)
```

### 模型处理

#### ViT处理pixel_values

输入： [num_patches, 3, 448, 448]

- ViT特征提取: ViT 内部按步长 14 将每个 448×448 的切片切分为 32×32=1024 个空间小块，输出形状为 [num_patches, 1024, vit_hidden_size] 的纯空间特征序列。
- 降采样与2D重组： 将 1024 个序列 Token 还原为 32×32 的 2D 矩阵结构，并利用 scale_factor 为 0.5 的 Pixel Shuffle（像素洗牌）逻辑将空间邻域内 2×2 的 4 个特征块在通道维度进行拼接，降低序列长度同时扩大隐层维度，输出形状为 [num_patches, 256, 4096]。
- 特征映射： 通过两层 MLP（mlp1）将压缩后的 4096 维视觉特征投影至大语言模型的隐层维度C，并按切片顺序平铺展开，最终生成形状为 [num_patches * 256, C] 的视觉嵌入序列 vit_embeds。

最终输出vit_embeds, shape=[num_patches * 256, C]。

#### SlowFast处理pixel_values2

输入： [32, 3, 448, 448]

- 时空路径重组层：将重复 32 次的静态图像序列通过 view 和 permute 转换为视频格式的五维张量 [B, 3, 32, 448, 448]，随后通过 pack_pathway_output 函数将其拆分为 Slow（低帧率采样）和 Fast（高帧率采样）两条平行的时空路径，以适应 SlowFast 模型对视频流输入的结构要求。
- SlowFast 特征提取层： 3D 视觉骨干网络通过多层 3D 卷积提取序列中的时空特征，即使帧内容完全相同，模型仍会将其作为时间体进行建模，并在网络末端执行全局平均池化（Global Average Pooling），将整个序列的信息高度浓缩为一个全局特征向量，输出形状为 [1, 2304]。
- 运动特征映射层：通过专门设计的 motion_mlp（包含 LayerNorm、两层线性映射及 GELU 激活函数）将 2304 维的全局摘要特征投影至大语言模型的隐层维度 C 。
- 
- 最终输出motion_embeds，shape=[1, C]。

#### 填入input_ids序列中

填槽方案就是在input_embeds对应下标位置加上对应的内容：
```python
input_embeds[selected1] = input_embeds[selected1] * 0.0 + vit_embeds.reshape(-1, C)
input_embeds[selected2] = input_embeds[selected2] * 0.0 + motion_embeds.reshape(-1, C)
```

下标的计算方式如下：
```python
# 1. 识别所有图像占位符：在 input_ids 中找到所有特殊的 img_context_token_id 位置
# selected 是一个布尔矩阵，形状为 [B, N]，为 True 的地方表示这里是一个视觉坑位
selected = (input_ids == self.img_context_token_id)

# 2. 统计序号：在序列维度（dim=1）上计算累加和
# 假设有 513 个视觉 Token，则该行数值会从 1 增长到 513
selected_cumsum = torch.cumsum(selected, dim=1)

# 3. 确定总数：找到每行（每个样本）中累加和的最大值
# max_cumsum 代表该样本中总共有多少个视觉坑位
max_cumsum = selected_cumsum.max(dim=1, keepdim=True)[0]

# 4. 锁定“最后一个”坑位：只有当当前累加值等于最大值，且该位置本身是视觉 Token 时才为 True
# 在本模型预处理逻辑中，最后一个 <image> 标签固定代表 Motion Feature
last_true_mask = (selected_cumsum == max_cumsum) & selected

# 5. 提取静态特征坑位 (selected1)：
# 先克隆所有坑位掩码，然后将最后一个坑位（运动特征位）剔除（设为 False）
# 剩下的 selected1 对应 Frame1, Frame2... 展开后的 num_patches * 256 个坑位
selected1 = selected.clone()
selected1[last_true_mask] = False

# 6. 提取运动特征坑位 (selected2)：
# 仅保留最后一个坑位，对应字符串中 "Motion Feature: <image>" 展开后的那 1 个 Token 位
selected2 = last_true_mask
```

