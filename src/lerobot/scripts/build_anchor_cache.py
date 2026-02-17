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
离线 AnchorCache 生成脚本（Hidden State Anchoring）
===================================================

【功能说明】
为 CRaFT 训练生成离线 AnchorCache，包含：
- 图像帧（从 LeRobot dataset 采样）
- 固定的 prompt 模板
- Teacher 模型的 hidden states（表征蒸馏）

【与 Token-level 版本的区别】
- 旧版本：保存 teacher 生成的 suffix tokens/labels（用于 token-level distillation）
- 新版本：保存 teacher 的 hidden states（用于 representation distillation）

【优势】
1. 避免 π0_fast 不稳定输出自然语言的问题
2. Cache 更小（只保存少量 hidden vectors）
3. 训练更稳定（hidden states 比 tokens 更平滑）

【输出格式】
生成多个 .pt shard 文件，每个包含：
{
    "pixel_values": Tensor[B, C, H, W],      # 图像，float32，已归一化到 [-1, 1]
    "input_ids": Tensor[B, seq_len],         # 完整输入序列（prompt + BOS）
    "attention_mask": Tensor[B, seq_len],    # 注意力掩码
    "teacher_hidden": Tensor[B, n_layers, n_vecs, hidden_dim],  # Teacher hidden states
    "meta": dict,  # 元数据：layers_to_save, pooling 方式, token 范围等
}

【Hidden States 提取策略】
- layers_to_save: 默认 [-2, -1]（最后两层）
- token_pooling:
    a) vision_token_mean: 对视觉 token 取 mean pooling，得到 1 个向量
    b) text_token_last: 对文本 token 取最后一个 token 向量

【使用示例】
```bash
# 基础用法
python src/lerobot/scripts/build_anchor_cache.py \\
    --policy.pretrained_path=physical-intelligence/pi0-fast \\
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \\
    --out_dir=data/anchor_cache_hidden \\
    --num_anchors=1000 \\
    --layers_to_save=-2,-1

# 自定义 prompts
python src/lerobot/scripts/build_anchor_cache.py \\
    --policy.pretrained_path=physical-intelligence/pi0-fast \\
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \\
    --out_dir=data/anchor_cache_hidden \\
    --num_anchors=1000 \\
    --prompts_file=prompts.json \\
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

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
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


def extract_hidden_states_with_pooling(
    teacher_inputs: dict,
    policy,
    layers_to_save: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    提取 teacher 模型的 hidden states 并进行 pooling
    
    Args:
        teacher_inputs: 包含 pixel_values, input_ids, attention_mask 的字典
        policy: Teacher policy 实例
        layers_to_save: 要保存的层索引（负数表示从后往前数）
        device: 设备
    
    Returns:
        teacher_hidden: [B, n_layers, n_vecs, hidden_dim]
        meta: 元数据字典
    """
    policy.eval()
    
    with torch.no_grad():
        # 准备输入
        images = [teacher_inputs["pixel_values"]]
        img_masks = [torch.ones(len(images[0]), dtype=torch.bool, device=device)]
        tokens = teacher_inputs["input_ids"]
        masks = teacher_inputs["attention_mask"]
        
        # 添加 BOS token
        bsize = tokens.shape[0]
        bos_token = torch.full(
            (bsize, 1), policy._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        
        # Embed prefix（不生成 action tokens）
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = (
            policy.model.embed_prefix_fast(
                images, img_masks, tokens, masks,
                fast_action_tokens=None,
                fast_action_masks=None,
            )
        )
        
        if policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        # Forward pass 并提取 hidden states
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = policy.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        
        # 使用 output_hidden_states=True
        language_model = policy.model.paligemma_with_expert.paligemma.language_model
        outputs = language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # outputs.hidden_states 是一个 tuple，包含每一层的 hidden states
        # hidden_states[0] 是 embedding layer 的输出
        # hidden_states[1] 到 hidden_states[n] 是各层的输出
        all_hidden_states = outputs.hidden_states  # tuple of [B, seq_len, hidden_dim]
        
        # 选择要保存的层
        total_layers = len(all_hidden_states) - 1  # 减去 embedding layer
        selected_hidden_states = []
        actual_layer_indices = []
        
        for layer_idx in layers_to_save:
            # 负数索引转换为正数
            if layer_idx < 0:
                actual_idx = total_layers + layer_idx + 1  # +1 因为 hidden_states[0] 是 embedding
            else:
                actual_idx = layer_idx + 1
            
            if 0 <= actual_idx < len(all_hidden_states):
                selected_hidden_states.append(all_hidden_states[actual_idx])
                actual_layer_indices.append(actual_idx - 1)  # 记录实际层号（不包括 embedding）
            else:
                logging.warning(f"层索引 {layer_idx} 超出范围，跳过")
        
        if not selected_hidden_states:
            raise ValueError(f"没有有效的层索引: {layers_to_save}")
        
        # Pooling 策略
        # 1. Vision tokens: mean pooling
        # 2. Text tokens: last token
        
        num_vision_tokens = total_t_images
        num_text_tokens = tokens.shape[1]
        
        pooled_hidden_states = []
        
        for hidden_state in selected_hidden_states:
            # hidden_state: [B, seq_len, hidden_dim]
            # seq_len = num_vision_tokens + num_text_tokens
            
            # Vision tokens: [B, num_vision_tokens, hidden_dim]
            vision_hidden = hidden_state[:, :num_vision_tokens, :]
            vision_pooled = vision_hidden.mean(dim=1)  # [B, hidden_dim]
            
            # Text tokens: [B, num_text_tokens, hidden_dim]
            text_hidden = hidden_state[:, num_vision_tokens:, :]
            # 找到每个样本的最后一个有效 text token
            text_masks = masks  # [B, num_text_tokens]
            last_text_indices = text_masks.sum(dim=1) - 1  # [B]
            text_pooled = text_hidden[torch.arange(bsize), last_text_indices]  # [B, hidden_dim]
            
            # 拼接: [B, 2, hidden_dim]
            layer_pooled = torch.stack([vision_pooled, text_pooled], dim=1)
            pooled_hidden_states.append(layer_pooled)
        
        # 拼接所有层: [B, n_layers, 2, hidden_dim]
        teacher_hidden = torch.stack(pooled_hidden_states, dim=1)
        
        # 元数据
        meta = {
            "layers_to_save": actual_layer_indices,
            "num_vision_tokens": num_vision_tokens,
            "num_text_tokens": num_text_tokens,
            "pooling_strategy": {
                "vision": "mean",
                "text": "last",
            },
            "hidden_dim": teacher_hidden.shape[-1],
            "n_vecs": 2,  # vision_pooled + text_pooled
        }
        
        return teacher_hidden.cpu().float(), meta


def save_anchor_cache_shard(
    shard_data: dict,
    out_dir: Path,
    shard_idx: int,
) -> None:
    """保存一个 anchor cache shard"""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
    
    # 转换为 CPU
    shard_data_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v 
        for k, v in shard_data.items()
    }
    
    torch.save(shard_data_cpu, shard_path)
    logging.info(f"保存 shard {shard_idx} 到 {shard_path}")


def build_anchor_cache(
    policy_pretrained_path: str,
    dataset_repo_id: str,
    out_dir: Path,
    num_anchors: int,
    prompts_file: Path | None = None,
    layers_to_save: list[int] = None,
    shard_size: int = 100,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """构建 AnchorCache（Hidden State Anchoring）"""
    if layers_to_save is None:
        layers_to_save = [-2, -1]  # 默认最后两层
    
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    logging.info("=" * 80)
    logging.info("开始构建 AnchorCache（Hidden State Anchoring）")
    logging.info("=" * 80)
    logging.info(f"Teacher 模型: {policy_pretrained_path}")
    logging.info(f"Dataset: {dataset_repo_id}")
    logging.info(f"输出目录: {out_dir}")
    logging.info(f"Anchor 数量: {num_anchors}")
    logging.info(f"保存层: {layers_to_save}")
    logging.info(f"Shard 大小: {shard_size}")
    logging.info(f"设备: {device}")
    
    # 1. 加载 prompts
    prompts = load_prompts(prompts_file)
    
    # 2. 加载 dataset
    logging.info(f"加载 dataset: {dataset_repo_id}")
    from lerobot.configs.train import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    
    dataset_cfg = DatasetConfig(repo_id=dataset_repo_id)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_pretrained_path)
    
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
        
        # 提取 hidden states
        teacher_hidden, meta = extract_hidden_states_with_pooling(
            teacher_inputs,
            policy,
            layers_to_save,
            device,
        )
        
        # 构建 shard 数据
        shard_data = {
            "pixel_values": teacher_inputs["pixel_values"],
            "input_ids": teacher_inputs["input_ids"],
            "attention_mask": teacher_inputs["attention_mask"],
            "teacher_hidden": teacher_hidden,  # [B, n_layers, n_vecs, hidden_dim]
            "meta": meta,
            "prompts": teacher_inputs["prompts"],
        }
        
        # 保存 shard
        save_anchor_cache_shard(shard_data, out_dir, shard_idx)
    
    # 7. 保存全局元数据
    metadata = {
        "num_anchors": num_anchors,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "layers_to_save": layers_to_save,
        "prompts": prompts,
        "image_keys": image_keys,
        "policy_pretrained_path": policy_pretrained_path,
        "dataset_repo_id": dataset_repo_id,
        "seed": seed,
        "cache_type": "hidden_state_anchoring",  # 标记为 hidden state 版本
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
    """主函数"""
    policy_pretrained_path = cfg.policy.pretrained_path
    dataset_repo_id = cfg.dataset.repo_id
    
    import sys
    parser_extra = argparse.ArgumentParser()
    parser_extra.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser_extra.add_argument("--num_anchors", type=int, required=True, help="Anchor 数量")
    parser_extra.add_argument("--prompts_file", type=str, default=None, help="Prompts JSON 文件路径")
    parser_extra.add_argument("--layers_to_save", type=str, default="-2,-1", help="要保存的层索引（逗号分隔）")
    parser_extra.add_argument("--shard_size", type=int, default=100, help="每个 shard 的样本数")
    parser_extra.add_argument("--seed", type=int, default=42, help="随机种子")
    parser_extra.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    
    args, _ = parser_extra.parse_known_args(sys.argv[1:])
    
    out_dir = Path(args.out_dir)
    prompts_file = Path(args.prompts_file) if args.prompts_file else None
    layers_to_save = [int(x.strip()) for x in args.layers_to_save.split(',')]
    
    build_anchor_cache(
        policy_pretrained_path=policy_pretrained_path,
        dataset_repo_id=dataset_repo_id,
        out_dir=out_dir,
        num_anchors=args.num_anchors,
        prompts_file=prompts_file,
        layers_to_save=layers_to_save,
        shard_size=args.shard_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()
