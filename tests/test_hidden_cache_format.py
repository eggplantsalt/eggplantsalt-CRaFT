#!/usr/bin/env python

"""
Hidden Feature Cache 格式测试
==============================

测试 hidden feature cache 的加载、格式验证和 batch collation。
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch


def test_hidden_cache_dataset_loading(tmp_path):
    """测试 hidden feature cache 的加载"""
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    # 创建 fake cache
    cache_dir = tmp_path / "hidden_cache"
    cache_dir.mkdir()
    
    # 参数
    num_samples = 10
    shard_size = 5
    num_shards = 2
    hidden_dim = 128
    
    # 创建 shards
    for shard_idx in range(num_shards):
        shard_data = {
            "pixel_values": torch.randn(shard_size, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (shard_size, 50)),
            "attention_mask": torch.ones(shard_size, 50),
            "target_features": torch.randn(shard_size, hidden_dim).half(),  # float16
            "meta": {
                "hidden_layer": -2,
                "pooling": "mean_image_tokens",
                "dtype": "float16",
            },
        }
        
        shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)
    
    # 创建 metadata
    metadata = {
        "num_anchors": num_samples,
        "num_shards": num_shards,
        "shard_size": shard_size,
    }
    
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # 加载 dataset
    dataset = AnchorCacheDataset(cache_dir)
    
    # 验证
    assert len(dataset) == num_samples, f"Dataset 大小错误: {len(dataset)} != {num_samples}"
    
    # 获取样本
    sample = dataset[0]
    
    # 验证字段
    assert "pixel_values" in sample, "缺少 pixel_values"
    assert "input_ids" in sample, "缺少 input_ids"
    assert "attention_mask" in sample, "缺少 attention_mask"
    assert "target_features" in sample, "缺少 target_features"
    assert "meta" in sample, "缺少 meta"
    
    # 验证 shape
    assert sample["pixel_values"].shape == (3, 224, 224), f"pixel_values shape 错误: {sample['pixel_values'].shape}"
    assert sample["input_ids"].shape == (50,), f"input_ids shape 错误: {sample['input_ids'].shape}"
    assert sample["attention_mask"].shape == (50,), f"attention_mask shape 错误: {sample['attention_mask'].shape}"
    assert sample["target_features"].shape == (hidden_dim,), f"target_features shape 错误: {sample['target_features'].shape}"
    
    # 验证 dtype
    assert sample["target_features"].dtype == torch.float16, f"target_features dtype 错误: {sample['target_features'].dtype}"
    
    # 验证 meta
    assert sample["meta"]["hidden_layer"] == -2
    assert sample["meta"]["pooling"] == "mean_image_tokens"
    assert sample["meta"]["dtype"] == "float16"
    
    print("✓ Hidden cache dataset 加载测试通过")


def test_hidden_cache_cross_shard_access(tmp_path):
    """测试跨 shard 访问"""
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    # 创建 fake cache
    cache_dir = tmp_path / "hidden_cache"
    cache_dir.mkdir()
    
    # 参数
    shard_size = 3
    num_shards = 3
    num_samples = 8  # 不是 shard_size 的整数倍
    hidden_dim = 64
    
    # 创建 shards
    for shard_idx in range(num_shards):
        actual_size = min(shard_size, num_samples - shard_idx * shard_size)
        
        shard_data = {
            "pixel_values": torch.randn(actual_size, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (actual_size, 50)),
            "attention_mask": torch.ones(actual_size, 50),
            "target_features": torch.randn(actual_size, hidden_dim).half(),
            "meta": {
                "hidden_layer": -1,
                "pooling": "mean_masked",
                "dtype": "float16",
            },
        }
        
        shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)
    
    # 创建 metadata
    metadata = {
        "num_anchors": num_samples,
        "num_shards": num_shards,
        "shard_size": shard_size,
    }
    
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # 加载 dataset
    dataset = AnchorCacheDataset(cache_dir)
    
    # 测试跨 shard 访问
    sample_0 = dataset[0]  # shard 0
    sample_3 = dataset[3]  # shard 1
    sample_7 = dataset[7]  # shard 2
    
    assert sample_0["target_features"].shape == (hidden_dim,)
    assert sample_3["target_features"].shape == (hidden_dim,)
    assert sample_7["target_features"].shape == (hidden_dim,)
    
    print("✓ 跨 shard 访问测试通过")


def test_hidden_cache_dataloader_collation(tmp_path):
    """测试 DataLoader 的 batch collation"""
    from lerobot.craft.anchor_cache import build_anchor_dataloader
    
    # 创建 fake cache
    cache_dir = tmp_path / "hidden_cache"
    cache_dir.mkdir()
    
    # 参数
    num_samples = 20
    shard_size = 10
    num_shards = 2
    hidden_dim = 256
    batch_size = 4
    
    # 创建 shards
    for shard_idx in range(num_shards):
        shard_data = {
            "pixel_values": torch.randn(shard_size, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (shard_size, 50)),
            "attention_mask": torch.ones(shard_size, 50),
            "target_features": torch.randn(shard_size, hidden_dim).half(),
            "meta": {
                "hidden_layer": -2,
                "pooling": "mean_image_tokens",
                "dtype": "float16",
            },
        }
        
        shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)
    
    # 创建 metadata
    metadata = {
        "num_anchors": num_samples,
        "num_shards": num_shards,
        "shard_size": shard_size,
    }
    
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # 创建 DataLoader
    dataloader = build_anchor_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=0,  # 单进程测试
        shuffle=False,
        pin_memory=False,
    )
    
    # 获取一个 batch
    batch = next(iter(dataloader))
    
    # 验证 batch shape
    assert batch["pixel_values"].shape == (batch_size, 3, 224, 224), f"Batch pixel_values shape 错误: {batch['pixel_values'].shape}"
    assert batch["input_ids"].shape == (batch_size, 50), f"Batch input_ids shape 错误: {batch['input_ids'].shape}"
    assert batch["attention_mask"].shape == (batch_size, 50), f"Batch attention_mask shape 错误: {batch['attention_mask'].shape}"
    assert batch["target_features"].shape == (batch_size, hidden_dim), f"Batch target_features shape 错误: {batch['target_features'].shape}"
    
    # 验证 dtype
    assert batch["target_features"].dtype == torch.float16, f"Batch target_features dtype 错误: {batch['target_features'].dtype}"
    
    # 验证 meta（应该是单个 dict，不是 batch）
    assert isinstance(batch["meta"], dict), f"Meta 应该是 dict，而不是 {type(batch['meta'])}"
    
    print("✓ DataLoader collation 测试通过")


def test_hidden_cache_backward_compatibility(tmp_path):
    """测试与旧版本 cache 的兼容性"""
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    # 创建三种类型的 cache
    cache_types = [
        ("token_level", {"labels": torch.randint(0, 1000, (5, 50))}),
        ("hidden_state", {
            "teacher_hidden": torch.randn(5, 2, 10, 128),
            "meta": {"layers_to_save": [-2, -1], "pooling": "vision_token_mean"}
        }),
        ("hidden_feature", {
            "target_features": torch.randn(5, 128).half(),
            "meta": {"hidden_layer": -2, "pooling": "mean_image_tokens", "dtype": "float16"}
        }),
    ]
    
    for cache_type, extra_fields in cache_types:
        # 创建 cache
        cache_dir = tmp_path / f"{cache_type}_cache"
        cache_dir.mkdir()
        
        # 创建 shard
        shard_data = {
            "pixel_values": torch.randn(5, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (5, 50)),
            "attention_mask": torch.ones(5, 50),
            **extra_fields,
        }
        
        shard_path = cache_dir / "shard_0000.pt"
        torch.save(shard_data, shard_path)
        
        # 创建 metadata
        metadata = {
            "num_anchors": 5,
            "num_shards": 1,
            "shard_size": 5,
        }
        
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # 加载 dataset
        dataset = AnchorCacheDataset(cache_dir)
        sample = dataset[0]
        
        # 验证基础字段
        assert "pixel_values" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        
        # 验证类型特定字段
        if cache_type == "token_level":
            assert "labels" in sample, f"{cache_type}: 缺少 labels"
            assert "target_features" not in sample
            assert "teacher_hidden" not in sample
        elif cache_type == "hidden_state":
            assert "teacher_hidden" in sample, f"{cache_type}: 缺少 teacher_hidden"
            assert "meta" in sample
            assert "target_features" not in sample
        elif cache_type == "hidden_feature":
            assert "target_features" in sample, f"{cache_type}: 缺少 target_features"
            assert "meta" in sample
            assert "teacher_hidden" not in sample
        
        print(f"✓ {cache_type} cache 兼容性测试通过")


def test_target_features_dtype_conversion(tmp_path):
    """测试 target_features 的 dtype 转换"""
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    # 测试不同 dtype
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    
    for dtype in dtypes:
        # 创建 cache
        cache_dir = tmp_path / f"cache_{dtype}"
        cache_dir.mkdir()
        
        # 创建 shard
        shard_data = {
            "pixel_values": torch.randn(5, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (5, 50)),
            "attention_mask": torch.ones(5, 50),
            "target_features": torch.randn(5, 128).to(dtype),
            "meta": {
                "hidden_layer": -2,
                "pooling": "mean_image_tokens",
                "dtype": str(dtype),
            },
        }
        
        shard_path = cache_dir / "shard_0000.pt"
        torch.save(shard_data, shard_path)
        
        # 创建 metadata
        metadata = {
            "num_anchors": 5,
            "num_shards": 1,
            "shard_size": 5,
        }
        
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # 加载 dataset
        dataset = AnchorCacheDataset(cache_dir)
        sample = dataset[0]
        
        # 验证 dtype
        assert sample["target_features"].dtype == dtype, f"dtype 不匹配: {sample['target_features'].dtype} != {dtype}"
        
        print(f"✓ {dtype} dtype 测试通过")


if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        print("运行 hidden cache 格式测试...")
        test_hidden_cache_dataset_loading(tmp_path)
        test_hidden_cache_cross_shard_access(tmp_path)
        test_hidden_cache_dataloader_collation(tmp_path)
        test_hidden_cache_backward_compatibility(tmp_path)
        test_target_features_dtype_conversion(tmp_path)
        
        print("\n✓ 所有测试通过！")

