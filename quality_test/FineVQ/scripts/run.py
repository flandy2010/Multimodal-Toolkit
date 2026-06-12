"""
FineVQ single-image quality score inference.

Usage:
    cd quality_test/FineVQ
    python scripts/run.py --image_path /path/to/image.jpg \
                          --model_name_or_path IntMeGroup/FineVQ_score

Dependencies: torch, transformers, Pillow, torchvision, peft, pytorchvideo, flash-attn
"""

import argparse
import sys
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings

import torch
from PIL import Image
from copy import deepcopy

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internvl.model.internvl_chat_infer import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
from internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internlm
from internvl.train.constants import (
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
    BOX_END_TOKEN, BOX_START_TOKEN,
    QUAD_END_TOKEN, QUAD_START_TOKEN,
    REF_END_TOKEN, REF_START_TOKEN,
)
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


def load_model(model_path: str, device: str = "cuda"):
    """Load FineVQ_score model and tokenizer."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = 4096

    # Add special tokens
    token_list = [
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN, QUAD_END_TOKEN,
        REF_START_TOKEN, REF_END_TOKEN,
        BOX_START_TOKEN, BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Model config
    config = InternVLChatConfig.from_pretrained(model_path)
    config.vision_config.drop_path_rate = 0.1
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = "flash_attention_2"
    else:
        config.llm_config._attn_implementation = "flash_attention_2"
    config.template = "internlm2-chat"
    config.select_layer = -1
    config.dynamic_image_size = True
    config.use_thumbnail = True
    config.ps_version = "v2"
    config.min_dynamic_patch = 1
    config.max_dynamic_patch = 6

    # Load model
    model = InternVLChatModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, config=config
    )
    model.img_context_token_id = img_context_token_id

    # Resize embeddings if needed
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # Compute num_image_token
    patch_size = config.vision_config.patch_size
    force_image_size = config.force_image_size or config.vision_config.image_size
    model.num_image_token = int((force_image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))

    model.language_model.config.use_cache = False
    model.eval()
    model.to(device)

    return model, tokenizer


def prepare_image_inputs(image_path: str, model, tokenizer, device: str = "cuda"):
    """
    Prepare pixel_values, pixel_values2, input_ids, attention_mask, image_flags
    for a single image.
    """
    image_size = 448
    max_dynamic_patch = 6
    num_frames_for_slowfast = 32  # slowfast needs multiple frames; replicate the image

    # Load image
    image = Image.open(image_path).convert("RGB")

    # --- pixel_values: ViT dynamic patches ---
    transform = build_transform(is_train=False, input_size=image_size, normalize_type="imagenet")
    images = dynamic_preprocess(
        image, min_num=1, max_num=max_dynamic_patch,
        image_size=image_size, use_thumbnail=True
    )
    pixel_values = torch.stack([transform(img) for img in images])  # [num_patches, 3, 448, 448]

    # --- pixel_values2: slowfast frames (replicate single image as static "video") ---
    # 通过拼接画面让图片变成一个静态视频
    single_frame_img = dynamic_preprocess(
        image, image_size=image_size, use_thumbnail=True, max_num=1
    )
    frame_tensor = transform(single_frame_img[0])  # [3, 448, 448]
    pixel_values2 = frame_tensor.unsqueeze(0).repeat(num_frames_for_slowfast, 1, 1, 1)  # [32, 3, 448, 448]

    # --- Build conversation and tokenize ---
    num_patches = pixel_values.shape[0]
    # The model expects num_patches frames + 1 motion token
    num_patches_total = num_patches + 1

    # Construct conversation (mimics video_get_item but for a single image)
    # 拼接一下prompt用于评价图片（视频）质量
    special_tokens = "\n".join([f"Frame{i+1}: <image>" for i in range(num_patches)])
    special_tokens = special_tokens + "\nMotion Feature: <image>"
    question = f"{special_tokens}\nHow would you rate the overall quality of this image?"
    answer = "The overall quality of the image is excellent."

    # 关于这里为什么要拼接上“The overall quality of the image is excellent.”
    # 原因是模型实际上是选择了这段内容的某个token对应的logits经过MLP后作为分数的。
    conversations = [
        {"from": "human", "value": question},
        {"from": "ai", "value": answer},
    ]

    # Tokenize using preprocess_internlm
    num_image_token = model.num_image_token
    num_image_tokens = [num_image_token] * num_patches_total
    num_image_tokens[-1] = 1  # motion feature placeholder is 1 token

    ret = preprocess_internlm(
        "internlm2-chat",
        [deepcopy(conversations)],
        tokenizer,
        num_image_tokens,
        group_by_length=False,
        ds_name="inference",
        num_image=num_patches_total,
    )

    input_ids = ret["input_ids"]          # [1, seq_len]
    attention_mask = ret["attention_mask"] # [1, seq_len]
    labels = ret["labels"]                # [1, seq_len]
    image_flags = torch.tensor([1] * num_patches, dtype=torch.long)  # only ViT patches flagged

    # Move to device
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    pixel_values2 = pixel_values2.to(device=device, dtype=torch.bfloat16)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    image_flags = image_flags.to(device)

    return pixel_values, pixel_values2, input_ids, attention_mask, labels, image_flags


@torch.no_grad()
def predict_quality(model, tokenizer, image_path: str, device: str = "cuda") -> float:
    """Run inference and return quality score for a single image."""
    pixel_values, pixel_values2, input_ids, attention_mask, labels, image_flags = \
        prepare_image_inputs(image_path, model, tokenizer, device)

    output = model(
        pixel_values=pixel_values,
        pixel_values2=pixel_values2,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        labels=labels,
    )
    score = output["score1"].item()
    return score


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../../data/images/horses.jpg", help="输入图像的路径")
    parser.add_argument("--model_path", type=str, default="./ckpt/FineVQ_score", help="模型权重(checkpoint)的路径")
    parser.add_argument("--device", type=str, default="cpu", help="使用的设备")
    args = parser.parse_args()

    device = torch.device(args.device)

    model, tokenizer = load_model(args.model_path, device=device)

    print(f"Predicting quality for: {args.image_path}")
    score = predict_quality(model, tokenizer, args.image_path, device=device)
    print(f"Quality Score: {score:.4f}")


if __name__ == "__main__":
    main()
