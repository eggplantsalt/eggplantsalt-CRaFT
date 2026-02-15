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
离线 AnchorCache 生成脚本
========================

【功能说明】
为 CRaFT 训练生成离线 AnchorCache，包含：
- 图像帧（从 LeRobot dataset 采样）
- 固定的 prompt 模板
- Teacher 模型生成的 suffix tokens（确定性生成）

【输出格式】
生成多个 .pt shard 文件，每个包含：
{
    "pixel_values": Tensor[B, C, H, W],  # 图像，float32，已归一化到 [-1, 1]
    "input_ids": Tensor[B, seq_len],     # 完整输入序列（prompt + BOS）
    "attention_mask": Tensor[B, seq_len], # 注意力掩码
    "labels": Tensor[B, seq_len],        # 标签（prompt 部分为 -100，suffix 为 token ids，EOS 后为 -100）
}

【使用示例】
```bash
# 基础用法
python src/lerobot/scripts/build_anchor_cache.py \\
    --policy.pretrained_path=physical-intelligence/pi0-fast \\
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \\
    --out_dir=data/anchor_cache \\
    --num_anchors=1000

# 自定义 prompts
python src/lerobot/scripts/build_anchor_cache.py \\
    --policy.pretrained_path=physical-intelligence/pi0-fast \\
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \\
    --out_dir=data/anchor_cache \\
    --num_anchors=1000 \\
    --prompts_file=prompts.json \\
    --max_new_tokens=256 \\
    --shard_size=100
```

【Prompts 文件格式】
JSON 格式，包含 prompt 模板列表：
```json
{
    "prompts": [
        "Pick up the object",
        "Place the object in the box",
        "Move to the target position"
    ]
}
```

【注意事项】
1. Teacher 生成必须确定性：do_sample=False, temperature=0
2. 图像自动探测 dataset 中的 camera keys（支持多种命名）
3. 输出 labels 的 mask 规则：
   - Prompt tokens: -100（不计算损失）
   - Teacher suffix tokens: 实际 token ids（计算损失）
   - EOS 之后: -100（不计算损失）
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import init_logging
from lerobot.utils.random_utils import set_seed


# 默认 prompt 模板（如果用户未提供）
DEFAULT_PROMPTS = [
    "Pick up the object",
    "Place the object in the container",
    "Move to the target position",
    "Grasp the item",
    "Release the object",
]


def load_prompts(prompts_file: Path | None) -> list[str]:
    """
    加载 prompt 模板列表
    
    Args:
        prompts_file: JSON 文件路径，包含 {"prompts": [...]} 格式
    
    Returns:
        Prompt 字符串列表
    """
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
    """
    自动探测 dataset 中的图像 keys
    
    LeRobot dataset 的图像 key 可能有多种命名方式：
    - observation.images.{camera_name}
    - observation.image
    - pixels.{camera_name}
    
    Args:
        dataset: LeRobot dataset 实例
    
    Returns:
        图像 key 列表
    """
    image_keys = []
    
    # 方法 1: 使用 dataset.meta.camera_keys（最可靠）
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
    
    # 方法 3: 从第一个样本中探测（fallback）
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
    """
    从 dataset 中随机采样帧
    
    Args:
        dataset: LeRobot dataset
        num_samples: 采样数量
        seed: 随机种子
    
    Returns:
        采样的帧列表
    """
    rng = random.Random(seed)
    total_frames = len(dataset)
    
    if num_samples > total_frames:
        logging.warning(f"请求采样 {num_samples} 帧，但 dataset 只有 {total_frames} 帧，将采样全部")
        num_samples = total_frames
    
    # 随机采样索引
    indices = rng.sample(range(total_frames), num_samples)
    indices.sort()  # 排序以提高访问效率
    
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
    """
    准备 teacher 模型的输入
    
    Args:
        frames: 采样的帧列表
        prompts: Prompt 模板列表
        image_keys: 图像 key 列表
        policy: Policy 实例（用于访问 processor）
        device: 设备
    
    Returns:
        包含 pixel_values, input_ids, attention_mask 的字典
    """
    batch_size = len(frames)
    
    # 为每个帧随机分配一个 prompt
    rng = random.Random(42)
    frame_prompts = [rng.choice(prompts) for _ in range(batch_size)]
    
    # 提取图像（使用第一个 image key）
    # 注意：pi0_fast 支持多相机，但这里简化为使用第一个相机
    primary_image_key = image_keys[0]
    logging.info(f"使用图像 key: {primary_image_key}")
    
    images = []
    for frame in frames:
        img = frame[primary_image_key]
        # 确保图像格式正确 [C, H, W]
        if img.ndim == 4:  # [1, C, H, W]
            img = img.squeeze(0)
        images.append(img)
    
    images = torch.stack(images).to(device)  # [B, C, H, W]
    
    # 使用 policy 的 tokenizer 处理 prompts
    tokenizer = policy._paligemma_tokenizer
    
    # Tokenize prompts
    tokenized = tokenizer(
        frame_prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # 预处理图像（归一化到 [-1, 1]）
    # pi0_fast 期望图像在 [-1, 1] 范围
    if images.dtype == torch.uint8:
        images = images.float() / 255.0  # [0, 1]
    images = images * 2.0 - 1.0  # [-1, 1]
    
    # Resize 图像到 policy 期望的分辨率
    from lerobot.policies.pi0_fast.modeling_pi0_fast import resize_with_pad_torch
    
    target_h, target_w = policy.config.image_resolution
    # 转换为 [B, H, W, C] 格式（resize_with_pad_torch 期望）
    images_hwc = images.permute(0, 2, 3, 1)
    images_resized = resize_with_pad_torch(images_hwc, target_h, target_w)
    # 转回 [B, C, H, W]
    pixel_values = images_resized.permute(0, 3, 1, 2)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": frame_prompts,
    }


def generate_teacher_outputs(
    teacher_inputs: dict,
    policy,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    使用 teacher 模型生成 suffix tokens（确定性生成）
    
    Args:
        teacher_inputs: 包含 pixel_values, input_ids, attention_mask 的字典
        policy: Teacher policy 实例
        max_new_tokens: 最大生成 token 数
        device: 设备
    
    Returns:
        生成的 token ids [B, max_new_tokens]
    """
    policy.eval()
    
    with torch.no_grad():
        # 准备输入（模拟 pi0_fast 的输入格式）
        images = [teacher_inputs["pixel_values"]]  # List of images
        img_masks = [torch.ones(len(images[0]), dtype=torch.bool, device=device)]
        tokens = teacher_inputs["input_ids"]
        masks = teacher_inputs["attention_mask"]
        
        # 使用 policy 的 sample_actions_fast 方法生成 tokens
        # 注意：这里我们需要生成 action tokens，但实际上我们想要的是 language tokens
        # 对于 pi0_fast，我们直接调用底层的生成方法
        
        # 使用确定性生成：temperature=0
        if policy.config.use_kv_cache:
            generated_tokens = policy.model.sample_actions_fast_kv_cache(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                max_decoding_steps=max_new_tokens,
                temperature=0.0,  # 确定性生成
            )
        else:
            generated_tokens = policy.model.sample_actions_fast(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                max_decoding_steps=max_new_tokens,
                temperature=0.0,  # 确定性生成
            )
    
    return generated_tokens


def create_labels_with_mask(
    input_ids: torch.Tensor,
    generated_tokens: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    创建 labels 张量，正确设置 mask
    
    规则：
    - Prompt tokens: -100（不计算损失）
    - Teacher suffix tokens: 实际 token ids（计算损失）
    - EOS 之后: -100（不计算损失）
    - Padding: -100（不计算损失）
    
    Args:
        input_ids: 输入 token ids [B, prompt_len]
        generated_tokens: 生成的 token ids [B, gen_len]
        tokenizer: Tokenizer 实例
    
    Returns:
        Labels 张量 [B, total_len]
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    gen_len = generated_tokens.shape[1]
    total_len = prompt_len + gen_len
    
    # 初始化 labels 为 -100
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long)
    
    # Prompt 部分保持 -100（不计算损失）
    # Suffix 部分设置为实际 token ids
    labels[:, prompt_len:] = generated_tokens
    
    # 找到 EOS token 并将其后的 tokens 设置为 -100
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        for i in range(batch_size):
            # 在 generated_tokens 中找到第一个 EOS
            eos_positions = (generated_tokens[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # EOS 之后的所有 tokens 设置为 -100
                labels[i, prompt_len + first_eos + 1:] = -100
    
    return labels


def save_anchor_cache_shard(
    shard_data: dict,
    out_dir: Path,
    shard_idx: int,
) -> None:
    """
    保存一个 anchor cache shard
    
    Args:
        shard_data: 包含 pixel_values, input_ids, attention_mask, labels 的字典
        out_dir: 输出目录
        shard_idx: Shard 索引
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
    
    # 转换为 CPU 并保存
    shard_data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in shard_data.items()}
    
    torch.save(shard_data_cpu, shard_path)
    logging.info(f"保存 shard {shard_idx} 到 {shard_path}")


def build_anchor_cache(
    policy_pretrained_path: str,
    dataset_repo_id: str,
    out_dir: Path,
    num_anchors: int,
    prompts_file: Path | None = None,
    max_new_tokens: int = 256,
    shard_size: int = 100,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """
    构建 AnchorCache
    
    Args:
        policy_pretrained_path: Teacher 模型路径
        dataset_repo_id: LeRobot dataset repo id
        out_dir: 输出目录
        num_anchors: 总 anchor 数量
        prompts_file: Prompts 文件路径（可选）
        max_new_tokens: Teacher 生成的最大 token 数
        shard_size: 每个 shard 的样本数
        seed: 随机种子
        device: 设备
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    logging.info("=" * 80)
    logging.info("开始构建 AnchorCache")
    logging.info("=" * 80)
    logging.info(f"Teacher 模型: {policy_pretrained_path}")
    logging.info(f"Dataset: {dataset_repo_id}")
    logging.info(f"输出目录: {out_dir}")
    logging.info(f"Anchor 数量: {num_anchors}")
    logging.info(f"Shard 大小: {shard_size}")
    logging.info(f"设备: {device}")
    
    # 1. 加载 prompts
    prompts = load_prompts(prompts_file)
    
    # 2. 加载 dataset
    logging.info(f"加载 dataset: {dataset_repo_id}")
    from lerobot.configs.train import DatasetConfig
    dataset_cfg = DatasetConfig(repo_id=dataset_repo_id)
    
    # 创建一个临时的 TrainPipelineConfig 用于 make_dataset
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(policy_pretrained_path)
    
    # 创建临时配置
    class TempConfig:
        def __init__(self):
            self.dataset = dataset_cfg
            self.policy = policy_cfg
            self.tolerance_s = 1e-4
    
    temp_cfg = TempConfig()
    dataset = make_dataset(temp_cfg)
    
    # 3. 探测图像 keys
    image_keys = detect_image_keys(dataset)
    
    # 4. 加载 teacher policy
    logging.info(f"加载 teacher policy: {policy_pretrained_path}")
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset.meta,
        rename_map={},
    )
    policy.to(device)
    policy.eval()
    
    # 5. 采样帧
    frames = sample_frames_from_dataset(dataset, num_anchors, seed)
    
    # 6. 分批处理并生成 shards
    num_shards = (num_anchors + shard_size - 1) // shard_size
    logging.info(f"将生成 {num_shards} 个 shards")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, num_anchors)
        shard_frames = frames[start_idx:end_idx]
        
        logging.info(f"处理 shard {shard_idx + 1}/{num_shards} (样本 {start_idx}-{end_idx})")
        
        # 准备输入
        teacher_inputs = prepare_teacher_inputs(
            shard_frames,
            prompts,
            image_keys,
            policy,
            device,
        )
        
        # 生成 teacher outputs
        generated_tokens = generate_teacher_outputs(
            teacher_inputs,
            policy,
            max_new_tokens,
            device,
        )
        
        # 创建 labels
        labels = create_labels_with_mask(
            teacher_inputs["input_ids"],
            generated_tokens,
            policy._paligemma_tokenizer,
        )
        
        # 拼接 input_ids 和 generated_tokens
        full_input_ids = torch.cat([teacher_inputs["input_ids"], generated_tokens], dim=1)
        
        # 扩展 attention_mask
        gen_mask = torch.ones_like(generated_tokens, dtype=torch.long)
        full_attention_mask = torch.cat([teacher_inputs["attention_mask"], gen_mask], dim=1)
        
        # 构建 shard 数据
        shard_data = {
            "pixel_values": teacher_inputs["pixel_values"],
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels,
            "prompts": teacher_inputs["prompts"],  # 保存 prompts 用于调试
        }
        
        # 保存 shard
        save_anchor_cache_shard(shard_data, out_dir, shard_idx)
    
    # 7. 保存元数据
    metadata = {
        "num_anchors": num_anchors,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "max_new_tokens": max_new_tokens,
        "prompts": prompts,
        "image_keys": image_keys,
        "policy_pretrained_path": policy_pretrained_path,
        "dataset_repo_id": dataset_repo_id,
        "seed": seed,
    }
    
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"保存元数据到 {metadata_path}")
    logging.info("=" * 80)
    logging.info("AnchorCache 构建完成！")
    logging.info("=" * 80)


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    """
    主函数：解析命令行参数并构建 AnchorCache
    """
    # 从配置中提取参数
    policy_pretrained_path = cfg.policy.pretrained_path
    dataset_repo_id = cfg.dataset.repo_id
    
    # 解析额外的命令行参数
    import sys
    parser_extra = argparse.ArgumentParser()
    parser_extra.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser_extra.add_argument("--num_anchors", type=int, required=True, help="Anchor 数量")
    parser_extra.add_argument("--prompts_file", type=str, default=None, help="Prompts JSON 文件路径")
    parser_extra.add_argument("--max_new_tokens", type=int, default=256, help="Teacher 生成的最大 token 数")
    parser_extra.add_argument("--shard_size", type=int, default=100, help="每个 shard 的样本数")
    parser_extra.add_argument("--seed", type=int, default=42, help="随机种子")
    parser_extra.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    
    args, _ = parser_extra.parse_known_args(sys.argv[1:])
    
    out_dir = Path(args.out_dir)
    prompts_file = Path(args.prompts_file) if args.prompts_file else None
    
    build_anchor_cache(
        policy_pretrained_path=policy_pretrained_path,
        dataset_repo_id=dataset_repo_id,
        out_dir=out_dir,
        num_anchors=args.num_anchors,
        prompts_file=prompts_file,
        max_new_tokens=args.max_new_tokens,
        shard_size=args.shard_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()

