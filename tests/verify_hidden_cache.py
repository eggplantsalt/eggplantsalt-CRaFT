#!/usr/bin/env python

"""
Hidden Feature Cache 格式验证脚本（独立运行）
==========================================

验证 hidden feature cache 的核心功能，不依赖完整的 lerobot 环境。
"""

import json
import sys
import tempfile
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 1: 基本功能")
    print("=" * 60)
    
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "hidden_cache"
        cache_dir.mkdir()
        
        # 创建 fake shard
        hidden_dim = 128
        shard_data = {
            "pixel_values": torch.randn(5, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (5, 50)),
            "attention_mask": torch.ones(5, 50),
            "target_features": torch.randn(5, hidden_dim).half(),
            "meta": {
                "hidden_layer": -2,
                "pooling": "mean_image_tokens",
                "dtype": "float16",
            },
        }
        
        torch.save(shard_data, cache_dir / "shard_0000.pt")
        
        # 创建 metadata
        metadata = {
            "num_anchors": 5,
            "num_shards": 1,
            "shard_size": 5,
        }
        
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # 加载 dataset
        dataset = AnchorCacheDataset(cache_dir)
        
        # 验证
        assert len(dataset) == 5
        sample = dataset[0]
        
        assert "pixel_values" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "target_features" in sample
        assert "meta" in sample
        
        assert sample["target_features"].shape == (hidden_dim,)
        assert sample["target_features"].dtype == torch.float16
        
        print(f"[OK] Dataset size: {len(dataset)}")
        print(f"[OK] target_features shape: {sample['target_features'].shape}")
        print(f"[OK] target_features dtype: {sample['target_features'].dtype}")
        print(f"[OK] meta: {sample['meta']}")
        print()


def test_three_cache_types():
    """测试三种 cache 类型的兼容性"""
    print("=" * 60)
    print("测试 2: 三种 cache 类型兼容性")
    print("=" * 60)
    
    from lerobot.craft.anchor_cache import AnchorCacheDataset
    
    cache_configs = [
        ("token_level", {"labels": torch.randint(0, 1000, (5, 50))}),
        ("hidden_state", {
            "teacher_hidden": torch.randn(5, 2, 10, 128),
            "meta": {"layers_to_save": [-2, -1]}
        }),
        ("hidden_feature", {
            "target_features": torch.randn(5, 128).half(),
            "meta": {"hidden_layer": -2, "pooling": "mean_image_tokens"}
        }),
    ]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for cache_type, extra_fields in cache_configs:
            cache_dir = Path(tmp_dir) / cache_type
            cache_dir.mkdir()
            
            # 创建 shard
            shard_data = {
                "pixel_values": torch.randn(5, 3, 224, 224),
                "input_ids": torch.randint(0, 1000, (5, 50)),
                "attention_mask": torch.ones(5, 50),
                **extra_fields,
            }
            
            torch.save(shard_data, cache_dir / "shard_0000.pt")
            
            # 创建 metadata
            metadata = {
                "num_anchors": 5,
                "num_shards": 1,
                "shard_size": 5,
            }
            
            with open(cache_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # 加载并验证
            dataset = AnchorCacheDataset(cache_dir)
            sample = dataset[0]
            
            # 验证类型特定字段
            if cache_type == "token_level":
                assert "labels" in sample
                assert "target_features" not in sample
                assert "teacher_hidden" not in sample
                print(f"[OK] {cache_type}: labels shape = {sample['labels'].shape}")
                
            elif cache_type == "hidden_state":
                assert "teacher_hidden" in sample
                assert "target_features" not in sample
                print(f"[OK] {cache_type}: teacher_hidden shape = {sample['teacher_hidden'].shape}")
                
            elif cache_type == "hidden_feature":
                assert "target_features" in sample
                assert "teacher_hidden" not in sample
                print(f"[OK] {cache_type}: target_features shape = {sample['target_features'].shape}")
    
    print()


def test_dataloader_batch():
    """测试 DataLoader batch collation"""
    print("=" * 60)
    print("测试 3: DataLoader Batch Collation")
    print("=" * 60)
    
    from lerobot.craft.anchor_cache import build_anchor_dataloader
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "hidden_cache"
        cache_dir.mkdir()
        
        # 创建 shard
        hidden_dim = 256
        batch_size = 4
        
        shard_data = {
            "pixel_values": torch.randn(10, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (10, 50)),
            "attention_mask": torch.ones(10, 50),
            "target_features": torch.randn(10, hidden_dim).half(),
            "meta": {
                "hidden_layer": -2,
                "pooling": "mean_image_tokens",
                "dtype": "float16",
            },
        }
        
        torch.save(shard_data, cache_dir / "shard_0000.pt")
        
        # 创建 metadata
        metadata = {
            "num_anchors": 10,
            "num_shards": 1,
            "shard_size": 10,
        }
        
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # 创建 DataLoader
        dataloader = build_anchor_dataloader(
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )
        
        # 获取 batch
        batch = next(iter(dataloader))
        
        # 验证
        assert batch["pixel_values"].shape == (batch_size, 3, 224, 224)
        assert batch["input_ids"].shape == (batch_size, 50)
        assert batch["target_features"].shape == (batch_size, hidden_dim)
        assert batch["target_features"].dtype == torch.float16
        
        print(f"[OK] Batch pixel_values shape: {batch['pixel_values'].shape}")
        print(f"[OK] Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"[OK] Batch target_features shape: {batch['target_features'].shape}")
        print(f"[OK] Batch target_features dtype: {batch['target_features'].dtype}")
        print()


def main():
    print("\n" + "=" * 60)
    print("Hidden Feature Cache 格式验证")
    print("=" * 60)
    print()
    
    try:
        test_basic_functionality()
        test_three_cache_types()
        test_dataloader_batch()
        
        print("=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

