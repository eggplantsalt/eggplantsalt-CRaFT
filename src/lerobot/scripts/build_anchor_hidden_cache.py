#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
离线 Hidden Feature Cache 生成脚本（Pooled Features）
====================================================

【功能说明】
为 CRaFT 训练生成离线 Hidden Feature Cache，包含：
- 图像帧（从 LeRobot dataset 采样）
- 固定的 prompt 模板
- Teacher 模型的 pooled hidden features（单个向量）

【与 build_anchor_cache.py 的区别】
- build_anchor_cache.py: 保存多层多向量 hidden states [B, n_layers, n_vecs, hidden_dim]
- build_anchor_hidden_cache.py: 保存单个 pooled feature vector [hidden_dim]（本脚本）

【优势】
1. Cache 更小：只保存一个 pooled vector，而非多层多向量
2. 训练更快：不需要在训练时做 pooling
3. 内存友好：适合大规模 anchor cache

【输出格式】
生成多个 .pt shard 文件，每个包含：
{
    "pixel_values": Tensor[B, C, H, W],      # 图像，float32，已归一化到 [-1, 1]
    "input_ids": Tensor[B, seq_len],         # 完整输入序列（prompt + BOS）
    "attention_mask": Tensor[B, seq_len],    # 注意力掩码
    "target_features": Tensor[B, hidden_dim], # Pooled hidden features (float16)
    "meta": {
        "hidden_layer": int,                  # 使用的层索引
        "pooling": str,                       # Pooling 策略
        "dtype": str,                         # 数据类型
    }
}

【Pooling 策略】
- mean_image_tokens: 对图像 tokens 取平均（推荐）
- mean_masked: 对所有非 padding tokens 取平均
- last_token: 取最后一个 token
- cls_token: 取 CLS token（如果有）

【使用示例】
```bash
# 基础用法
python src/lerobot/scripts/build_anchor_hidden_cache.py \\
    --teacher_policy_path=physical-intelligence/pi0-fast \\
    --dataset_repo_id=lerobot/aloha_sim_insertion_human \\
    --output_dir=data/anchor_hidden_cache \\
    --num_samples=1000

# 自定义配置
python src/lerobot/scripts/build_anchor_hidden_cache.py \\
    --teacher_policy_path=physical-intelligence/pi0-fast \\
    --dataset_repo_id=lerobot/aloha_sim_insertion_human \\
    --output_dir=data/anchor_hidden_cache \\
    --num_samples=1000 \\
    --prompts_file=prompts.json \\
    --hidden_layer=-2 \\
    --pooling=mean_image_tokens \\
    --dtype=float16 \\
    --shard_size=100
```
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import init_logging
from lerobot.utils.random_utils import set_seed


# 默认 prompt 模板
DEFAULT_PROMPTS = [
    "Pick up the object",
    "Place the object in the container",
    "Move to the target position",
    "Grasp the item",
    "Release the object",
]


def load_prompts(prompts_file: Path | None) -> list[str]:
    """加载 prompt 模板列表"""
    if prompts_file is None or not prompts_file.exists():
        logging.info(f"使用默认 prompts: {DEFAULT_PROMPTS}")
        return DEFAULT_PROMPTS
    
    with open(prompts_file) as f:
        data = json.load(f)
    
    prompts = data.get("prompts", [])
    if not prompts:
        logging.warning(f"Prompts 文件 {prompts_file} 为空，使用默认 prompts")
        return DEFAULT_PROMPTS
    
    logging.info(f"从 {prompts_file} 加载了 {len(prompts)} 个 prompts")
    return prompts


def detect_image_keys(dataset) -> list[str]:
    """自动探测 dataset 中的图像 keys"""
    image_keys = []
    
    # 方法 1: 使用 dataset.meta.camera_keys
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'camera_keys'):
        image_keys = dataset.meta.camera_keys
        logging.info(f"从 dataset.meta.camera_keys 探测到图像 keys: {image_keys}")
        return image_keys
    
    # 方法 2: 检查 features 中的 image/video 类型
    if hasattr(dataset, 'features'):
        for key, feature in dataset.features.items():
            if feature.get('dtype') in ['image', 'video']:
                image_keys.append(key)
        if image_keys:
            logging.info(f"从 features 探测到图像 keys: {image_keys}")
            return image_keys
    
    # 方法 3: 从第一个样本中探测
    if len(dataset) > 0:
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] in [1, 3]:
                image_keys.append(key)
        if image_keys:
            logging.info(f"从样本探测到图像 keys: {image_keys}")
            return image_keys
    
    raise ValueError("无法探测到任何图像 keys，请检查 dataset 格式")


def sample_frames_from_dataset(dataset, num_samples: int, seed: int = 42) -> list[dict]:
    """从 dataset 中随机采样帧"""
    rng = random.Random(seed)
    total_frames = len(dataset)
    
    if num_samples > total_frames:
        logging.warning(f"请求采样 {num_samples} 帧，但 dataset 只有 {total_frames} 帧，将采样全部")
        num_samples = total_frames
    
    indices = rng.sample(range(total_frames), num_samples)
    indices.sort()
    
    logging.info(f"从 {total_frames} 帧中采样 {num_samples} 帧")
    
    frames = []
    for idx in tqdm(indices, desc="采样帧"):
        frame = dataset[idx]
        frames.append(frame)
    
    return frames


def prepare_teacher_inputs(
    frames: list[dict],
    prompts: list[str],
    image_keys: list[str],
    policy,
    device: torch.device,
) -> dict:
    """准备 teacher 模型的输入"""
    batch_size = len(frames)
    
    # 为每个帧随机分配一个 prompt
    rng = random.Random(42)
    frame_prompts = [rng.choice(prompts) for _ in range(batch_size)]
    
    # 提取图像
    primary_image_key = image_keys[0]
    logging.info(f"使用图像 key: {primary_image_key}")
    
    images = []
    for frame in frames:
        img = frame[primary_image_key]
        if img.ndim == 4:
            img = img.squeeze(0)
        images.append(img)
    
    images = torch.stack(images).to(device)
    
    # Tokenize prompts
    tokenizer = policy._paligemma_tokenizer
    tokenized = tokenizer(
        frame_prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # 预处理图像
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    images = images * 2.0 - 1.0
    
    # Resize 图像
    from lerobot.policies.pi0_fast.modeling_pi0_fast import resize_with_pad_torch
    
    target_h, target_w = policy.config.image_resolution
    images_hwc = images.permute(0, 2, 3, 1)
    images_resized = resize_with_pad_torch(images_hwc, target_h, target_w)
    pixel_values = images_resized.permute(0, 3, 1, 2)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": frame_prompts,
    }


def identify_image_tokens(
    policy,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[int, int]:
    """
    识别图像 tokens 的范围
    
    对于 PaliGemma/Pi0 架构：
    - 图像 tokens 在序列开头
    - 数量 = (image_h // patch_size) * (image_w // patch_size)
    
    Returns:
        (start_idx, end_idx): 图像 tokens 的范围 [start_idx, end_idx)
    """
    # 方法 1: 从 policy config 获取
    if hasattr(policy.config, 'image_seq_length'):
        num_image_tokens = policy.config.image_seq_length
        logging.info(f"从 config 获取图像 token 数量: {num_image_tokens}")
        return 0, num_image_tokens
    
    # 方法 2: 计算
    if hasattr(policy.config, 'image_resolution') and hasattr(policy.config, 'patch_size'):
        h, w = policy.config.image_resolution
        patch_size = policy.config.patch_size
        num_image_tokens = (h // patch_size) * (w // patch_size)
        logging.info(f"计算得到图像 token 数量: {num_image_tokens} ({h}x{w} / {patch_size})")
        return 0, num_image_tokens
    
    # 方法 3: 默认值（PaliGemma 224x224, patch_size=16）
    num_image_tokens = 196  # (224 // 16) ** 2
    logging.warning(f"使用默认图像 token 数量: {num_image_tokens}")
    return 0, num_image_tokens


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
    policy,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    对 hidden states 进行 pooling
    
    Args:
        hidden_states: [B, seq_len, hidden_dim]
        attention_mask: [B, seq_len]
        pooling: Pooling 策略
        policy: Policy 实例（用于识别图像 tokens）
        input_ids: [B, seq_len]
    
    Returns:
        pooled_features: [B, hidden_dim]
    """
    B, seq_len, hidden_dim = hidden_states.shape
    
    if pooling == "mean_image_tokens":
        # 对图像 tokens 取平均
        start_idx, end_idx = identify_image_tokens(policy, input_ids, attention_mask)
        image_hidden = hidden_states[:, start_idx:end_idx, :]  # [B, num_image_tokens, hidden_dim]
        pooled = image_hidden.mean(dim=1)  # [B, hidden_dim]
        logging.info(f"使用 mean_image_tokens pooling: tokens [{start_idx}:{end_idx}]")
        
    elif pooling == "mean_masked":
        # 对所有非 padding tokens 取平均
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_hidden = hidden_states * mask
        pooled = masked_hidden.sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [B, hidden_dim]
        logging.info(f"使用 mean_masked pooling")
        
    elif pooling == "last_token":
        # 取最后一个非 padding token
        lengths = attention_mask.sum(dim=1) - 1  # [B]
        pooled = hidden_states[torch.arange(B), lengths]  # [B, hidden_dim]
        logging.info(f"使用 last_token pooling")
        
    elif pooling == "cls_token":
        # 取第一个 token（假设是 CLS）
        pooled = hidden_states[:, 0, :]  # [B, hidden_dim]
        logging.info(f"使用 cls_token pooling")
        
    else:
        raise ValueError(f"未知的 pooling 策略: {pooling}")
    
    return pooled


def extract_pooled_features(
    teacher_inputs: dict,
    policy,
    hidden_layer: int,
    pooling: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    提取 teacher 模型的 pooled hidden features
    
    Args:
        teacher_inputs: 包含 pixel_values, input_ids, attention_mask 的字典
        policy: Teacher policy 实例
        hidden_layer: 要提取的层索引（负数表示从后往前数）
        pooling: Pooling 策略
        device: 设备
        dtype: 目标数据类型
    
    Returns:
        target_features: [B, hidden_dim] 的 pooled features
    """
    pixel_values = teacher_inputs["pixel_values"]
    input_ids = teacher_inputs["input_ids"]
    attention_mask = teacher_inputs["attention_mask"]
    
    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = policy._paligemma_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    
    # 提取指定层的 hidden states
    all_hidden_states = outputs.hidden_states  # Tuple of [B, seq_len, hidden_dim]
    hidden_states = all_hidden_states[hidden_layer]  # [B, seq_len, hidden_dim]
    
    logging.info(f"提取第 {hidden_layer} 层 hidden states: {hidden_states.shape}")
    
    # Pooling
    pooled_features = pool_hidden_states(
        hidden_states,
        attention_mask,
        pooling,
        policy,
        input_ids,
    )
    
    # 转换数据类型
    pooled_features = pooled_features.to(dtype)
    
    logging.info(f"Pooled features shape: {pooled_features.shape}, dtype: {pooled_features.dtype}")
    
    return pooled_features


def build_hidden_cache(
    teacher_policy_path: str,
    dataset_repo_id: str,
    output_dir: Path,
    num_samples: int,
    prompts_file: Path | None = None,
    hidden_layer: int = -2,
    pooling: str = "mean_image_tokens",
    dtype: str = "float16",
    shard_size: int = 100,
    seed: int = 42,
):
    """
    构建 Hidden Feature Cache
    
    Args:
        teacher_policy_path: Teacher policy 路径
        dataset_repo_id: Dataset repo ID
        output_dir: 输出目录
        num_samples: 采样数量
        prompts_file: Prompts 文件路径
        hidden_layer: 要提取的层索引
        pooling: Pooling 策略
        dtype: 数据类型
        shard_size: 每个 shard 的大小
        seed: 随机种子
    """
    # 设置随机种子
    set_seed(seed)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 prompts
    prompts = load_prompts(prompts_file)
    
    # 加载 dataset
    logging.info(f"加载 dataset: {dataset_repo_id}")
    from lerobot.configs.train import TrainPipelineConfig
    cfg = TrainPipelineConfig()
    cfg.dataset.repo_id = dataset_repo_id
    dataset = make_dataset(cfg)
    
    # 探测图像 keys
    image_keys = detect_image_keys(dataset)
    
    # 采样帧
    frames = sample_frames_from_dataset(dataset, num_samples, seed)
    
    # 加载 teacher policy
    logging.info(f"加载 teacher policy: {teacher_policy_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from lerobot.configs.policy import PolicyConfig
    policy_cfg = PolicyConfig()
    policy_cfg.pretrained_path = teacher_policy_path
    policy = make_policy(policy_cfg, dataset.meta)
    policy = policy.to(device)
    policy.eval()
    
    # 转换 dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, torch.float16)
    
    # 分 shard 处理
    num_shards = (len(frames) + shard_size - 1) // shard_size
    logging.info(f"将生成 {num_shards} 个 shards，每个最多 {shard_size} 样本")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(frames))
        shard_frames = frames[start_idx:end_idx]
        
        logging.info(f"处理 shard {shard_idx + 1}/{num_shards} ({len(shard_frames)} 样本)")
        
        # 准备输入
        teacher_inputs = prepare_teacher_inputs(
            shard_frames,
            prompts,
            image_keys,
            policy,
            device,
        )
        
        # 提取 pooled features
        target_features = extract_pooled_features(
            teacher_inputs,
            policy,
            hidden_layer,
            pooling,
            device,
            target_dtype,
        )
        
        # 保存 shard
        shard_data = {
            "pixel_values": teacher_inputs["pixel_values"].cpu(),
            "input_ids": teacher_inputs["input_ids"].cpu(),
            "attention_mask": teacher_inputs["attention_mask"].cpu(),
            "target_features": target_features.cpu(),
            "meta": {
                "hidden_layer": hidden_layer,
                "pooling": pooling,
                "dtype": dtype,
            },
        }
        
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)
        logging.info(f"保存 shard: {shard_path}")
    
    # 保存 metadata
    metadata = {
        "teacher_policy_path": teacher_policy_path,
        "dataset_repo_id": dataset_repo_id,
        "num_samples": len(frames),
        "num_shards": num_shards,
        "shard_size": shard_size,
        "hidden_layer": hidden_layer,
        "pooling": pooling,
        "dtype": dtype,
        "prompts": prompts,
        "image_keys": image_keys,
        "seed": seed,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"保存 metadata: {metadata_path}")
    logging.info(f"✓ Hidden Feature Cache 构建完成: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="构建离线 Hidden Feature Cache")
    
    parser.add_argument(
        "--teacher_policy_path",
        type=str,
        required=True,
        help="Teacher policy 路径（例如: physical-intelligence/pi0-fast）",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repo ID（例如: lerobot/aloha_sim_insertion_human）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="采样数量",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Prompts 文件路径（JSON 格式）",
    )
    parser.add_argument(
        "--hidden_layer",
        type=int,
        default=-2,
        help="要提取的层索引（负数表示从后往前数，默认 -2）",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_image_tokens",
        choices=["mean_image_tokens", "mean_masked", "last_token", "cls_token"],
        help="Pooling 策略（默认 mean_image_tokens）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="数据类型（默认 float16）",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100,
        help="每个 shard 的大小（默认 100）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    init_logging()
    register_third_party_plugins()
    
    # 构建 cache
    build_hidden_cache(
        teacher_policy_path=args.teacher_policy_path,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        hidden_layer=args.hidden_layer,
        pooling=args.pooling,
        dtype=args.dtype,
        shard_size=args.shard_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

