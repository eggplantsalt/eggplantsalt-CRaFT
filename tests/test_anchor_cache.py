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
AnchorCache 格式和 Labels Mask 规则测试
======================================

【测试目标】
验证 AnchorCache 的数据格式和 labels mask 规则的正确性：
1. Labels 的 mask 规则正确（prompt token 为 -100；suffix token 不为 -100；EOS 后为 -100）
2. 数据格式符合预期（pixel_values, input_ids, attention_mask, labels）
3. AnchorCacheDataset 能正确加载和返回数据

【测试策略】
使用 mock 数据而不是真实模型，重点验证：
- Labels mask 的逻辑正确性
- 数据结构的完整性
- Dataset 和 DataLoader 的功能

【运行方法】
```bash
# 运行测试
pytest tests/test_anchor_cache.py -v

# 运行特定测试
pytest tests/test_anchor_cache.py::test_labels_mask_rules -v
```
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from lerobot.craft.anchor_cache import AnchorCacheDataset, build_anchor_dataloader


def create_mock_anchor_cache(cache_dir: Path, num_samples: int = 10, shard_size: int = 5):
    """
    创建 mock AnchorCache 用于测试
    
    Args:
        cache_dir: 缓存目录
        num_samples: 总样本数
        shard_size: 每个 shard 的样本数
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建元数据
    num_shards = (num_samples + shard_size - 1) // shard_size
    metadata = {
        "num_anchors": num_samples,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "max_new_tokens": 20,
        "prompts": ["Test prompt 1", "Test prompt 2"],
        "image_keys": ["observation.images.camera"],
        "policy_pretrained_path": "mock/policy",
        "dataset_repo_id": "mock/dataset",
        "seed": 42,
    }
    
    with open(cache_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    # 创建 mock shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, num_samples)
        batch_size = end_idx - start_idx
        
        # Mock 数据
        # 图像: [B, 3, 224, 224], float32, [-1, 1]
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # 输入序列: [B, seq_len]
        # 假设 prompt 长度为 10，suffix 长度为 20
        prompt_len = 10
        suffix_len = 20
        seq_len = prompt_len + suffix_len
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # Labels: prompt 部分为 -100，suffix 部分为实际 token ids
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[:, prompt_len:] = input_ids[:, prompt_len:]  # Suffix 部分
        
        # 模拟 EOS token（假设在 suffix 的中间位置）
        eos_token_id = 2  # Mock EOS token id
        for i in range(batch_size):
            # 在 suffix 中随机位置放置 EOS
            eos_pos = prompt_len + torch.randint(5, suffix_len - 5, (1,)).item()
            input_ids[i, eos_pos] = eos_token_id
            labels[i, eos_pos] = eos_token_id
            # EOS 之后的 tokens 设置为 -100
            labels[i, eos_pos + 1:] = -100
        
        prompts = [f"Mock prompt {i}" for i in range(batch_size)]
        
        shard_data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts,
        }
        
        shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)


def test_labels_mask_rules():
    """
    测试：Labels mask 规则正确
    
    验证：
    1. Prompt tokens 为 -100（不计算损失）
    2. Suffix tokens 为实际 token ids（计算损失）
    3. EOS 之后的 tokens 为 -100（不计算损失）
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "anchor_cache"
        create_mock_anchor_cache(cache_dir, num_samples=10, shard_size=5)
        
        # 加载数据集
        dataset = AnchorCacheDataset(cache_dir)
        
        # 获取一个样本
        sample = dataset[0]
        
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        
        # 假设 prompt 长度为 10
        prompt_len = 10
        
        # 验证 1: Prompt 部分全部为 -100
        assert torch.all(labels[:prompt_len] == -100), "Prompt tokens 应该全部为 -100"
        
        # 验证 2: Suffix 部分至少有一些非 -100 的 tokens
        suffix_labels = labels[prompt_len:]
        assert torch.any(suffix_labels != -100), "Suffix 应该包含非 -100 的 tokens"
        
        # 验证 3: 找到 EOS token（id=2），验证其后的 tokens 为 -100
        eos_token_id = 2
        eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
        
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            # EOS 之后的所有 tokens 应该为 -100
            if first_eos + 1 < len(labels):
                assert torch.all(labels[first_eos + 1:] == -100), "EOS 之后的 tokens 应该全部为 -100"
        
        print("✓ Labels mask 规则验证通过")


def test_anchor_cache_dataset_format():
    """
    测试：AnchorCache 数据格式正确
    
    验证：
    1. 包含所有必需的字段
    2. 数据类型正确
    3. 形状符合预期
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "anchor_cache"
        create_mock_anchor_cache(cache_dir, num_samples=10, shard_size=5)
        
        # 加载数据集
        dataset = AnchorCacheDataset(cache_dir)
        
        # 验证数据集大小
        assert len(dataset) == 10, f"数据集大小应为 10，实际为 {len(dataset)}"
        
        # 获取一个样本
        sample = dataset[0]
        
        # 验证字段存在
        required_keys = {"pixel_values", "input_ids", "attention_mask", "labels"}
        assert required_keys.issubset(sample.keys()), f"缺少必需字段，期望 {required_keys}，实际 {sample.keys()}"
        
        # 验证数据类型
        assert isinstance(sample["pixel_values"], torch.Tensor), "pixel_values 应为 Tensor"
        assert isinstance(sample["input_ids"], torch.Tensor), "input_ids 应为 Tensor"
        assert isinstance(sample["attention_mask"], torch.Tensor), "attention_mask 应为 Tensor"
        assert isinstance(sample["labels"], torch.Tensor), "labels 应为 Tensor"
        
        # 验证形状
        assert sample["pixel_values"].ndim == 3, "pixel_values 应为 3D (C, H, W)"
        assert sample["pixel_values"].shape[0] == 3, "pixel_values 应有 3 个通道"
        
        assert sample["input_ids"].ndim == 1, "input_ids 应为 1D"
        assert sample["attention_mask"].ndim == 1, "attention_mask 应为 1D"
        assert sample["labels"].ndim == 1, "labels 应为 1D"
        
        # 验证序列长度一致
        seq_len = sample["input_ids"].shape[0]
        assert sample["attention_mask"].shape[0] == seq_len, "attention_mask 长度应与 input_ids 一致"
        assert sample["labels"].shape[0] == seq_len, "labels 长度应与 input_ids 一致"
        
        print("✓ 数据格式验证通过")


def test_anchor_cache_dataloader():
    """
    测试：AnchorCache DataLoader 功能正常
    
    验证：
    1. DataLoader 能正确创建
    2. 能正确迭代和返回批次
    3. 批次大小正确
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "anchor_cache"
        create_mock_anchor_cache(cache_dir, num_samples=10, shard_size=5)
        
        # 创建 DataLoader
        dataloader = build_anchor_dataloader(
            cache_dir=cache_dir,
            batch_size=4,
            num_workers=0,  # 使用单进程避免测试复杂性
            shuffle=False,
        )
        
        # 验证能迭代
        batches = list(dataloader)
        assert len(batches) > 0, "DataLoader 应返回至少一个批次"
        
        # 验证第一个批次
        first_batch = batches[0]
        
        # 验证批次大小
        assert first_batch["pixel_values"].shape[0] == 4, "批次大小应为 4"
        
        # 验证批次格式
        assert first_batch["pixel_values"].ndim == 4, "批次 pixel_values 应为 4D (B, C, H, W)"
        assert first_batch["input_ids"].ndim == 2, "批次 input_ids 应为 2D (B, seq_len)"
        assert first_batch["attention_mask"].ndim == 2, "批次 attention_mask 应为 2D (B, seq_len)"
        assert first_batch["labels"].ndim == 2, "批次 labels 应为 2D (B, seq_len)"
        
        print("✓ DataLoader 功能验证通过")


def test_anchor_cache_cross_shard_access():
    """
    测试：跨 shard 访问正确
    
    验证：
    1. 能正确访问不同 shard 中的样本
    2. Shard 缓存机制工作正常
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "anchor_cache"
        create_mock_anchor_cache(cache_dir, num_samples=10, shard_size=3)  # 3 个样本/shard
        
        # 加载数据集
        dataset = AnchorCacheDataset(cache_dir)
        
        # 访问第一个 shard 的样本
        sample_0 = dataset[0]
        assert sample_0 is not None
        
        # 访问第二个 shard 的样本
        sample_3 = dataset[3]
        assert sample_3 is not None
        
        # 访问第三个 shard 的样本
        sample_6 = dataset[6]
        assert sample_6 is not None
        
        # 验证不同 shard 的样本不同（通过 input_ids）
        assert not torch.equal(sample_0["input_ids"], sample_3["input_ids"]), "不同样本应有不同的 input_ids"
        
        print("✓ 跨 shard 访问验证通过")


def test_labels_no_loss_on_padding():
    """
    测试：Padding tokens 不计算损失
    
    验证：
    1. Attention mask 为 0 的位置，labels 应为 -100
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "anchor_cache"
        
        # 创建带 padding 的 mock cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "num_anchors": 2,
            "num_shards": 1,
            "shard_size": 2,
            "max_new_tokens": 20,
            "prompts": ["Test"],
            "image_keys": ["camera"],
            "policy_pretrained_path": "mock",
            "dataset_repo_id": "mock",
            "seed": 42,
        }
        
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # 创建带 padding 的数据
        seq_len = 30
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, seq_len))
        
        # 第一个样本：无 padding
        attention_mask = torch.ones(2, seq_len, dtype=torch.long)
        # 第二个样本：后 10 个 tokens 为 padding
        attention_mask[1, 20:] = 0
        
        labels = torch.randint(0, 1000, (2, seq_len), dtype=torch.long)
        # Padding 位置的 labels 应为 -100
        labels[1, 20:] = -100
        
        shard_data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": ["Test 1", "Test 2"],
        }
        
        torch.save(shard_data, cache_dir / "shard_0000.pt")
        
        # 加载并验证
        dataset = AnchorCacheDataset(cache_dir)
        sample = dataset[1]  # 第二个样本有 padding
        
        # 验证 padding 位置的 labels 为 -100
        padding_positions = sample["attention_mask"] == 0
        if padding_positions.any():
            assert torch.all(sample["labels"][padding_positions] == -100), \
                "Padding 位置的 labels 应为 -100"
        
        print("✓ Padding 不计算损失验证通过")


if __name__ == "__main__":
    # 运行所有测试
    print("=" * 80)
    print("运行 AnchorCache 测试")
    print("=" * 80)
    
    test_labels_mask_rules()
    test_anchor_cache_dataset_format()
    test_anchor_cache_dataloader()
    test_anchor_cache_cross_shard_access()
    test_labels_no_loss_on_padding()
    
    print("=" * 80)
    print("所有测试通过！✓")
    print("=" * 80)

