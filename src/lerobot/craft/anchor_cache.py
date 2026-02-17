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
锚点数据缓存模块 (Anchor Cache - Hidden State Anchoring)
========================================================

【模块功能】
管理离线生成的 AnchorCache 的加载和采样，用于 CRaFT 训练中的保留损失计算。

【AnchorCache 格式】
支持三种格式的 cache：

1. Token-level cache（旧版本）：
{
    "pixel_values": Tensor[B, C, H, W],
    "input_ids": Tensor[B, seq_len],
    "attention_mask": Tensor[B, seq_len],
    "labels": Tensor[B, seq_len],  # Teacher 生成的 tokens
}

2. Hidden state cache（多层多向量）：
{
    "pixel_values": Tensor[B, C, H, W],
    "input_ids": Tensor[B, seq_len],
    "attention_mask": Tensor[B, seq_len],
    "teacher_hidden": Tensor[B, n_layers, n_vecs, hidden_dim],
    "meta": dict,  # layers, pooling 策略等
}

3. Hidden feature cache（单个 pooled vector）：
{
    "pixel_values": Tensor[B, C, H, W],
    "input_ids": Tensor[B, seq_len],
    "attention_mask": Tensor[B, seq_len],
    "target_features": Tensor[B, hidden_dim],  # Pooled features
    "meta": dict,  # hidden_layer, pooling, dtype
}

【使用示例】
```python
from lerobot.craft.anchor_cache import AnchorCacheDataset, build_anchor_dataloader

# 创建 DataLoader
anchor_dataloader = build_anchor_dataloader(
    cache_dir="data/anchor_cache_hidden",
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# 在训练循环中使用
from lerobot.datasets.utils import cycle
anchor_dl_iter = cycle(anchor_dataloader)

for step in range(total_steps):
    anchor_batch = next(anchor_dl_iter)
    # anchor_batch 包含: pixel_values, input_ids, attention_mask, teacher_hidden, meta
    retention_loss = compute_retention_loss_hidden(student_hidden, anchor_batch["teacher_hidden"])
```

【设计考虑】
1. 离线生成: Teacher hidden states 在训练前完成，避免在线调用开销
2. 表征蒸馏: 使用 hidden states 而非 tokens，更稳定
3. 内存效率: 只保存少量 pooled vectors，cache 很小
4. 向后兼容: 自动检测 cache 类型（token-level 或 hidden-state）
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class AnchorCacheDataset(Dataset):
    """
    AnchorCache 数据集加载器
    
    【功能说明】
    从离线生成的 AnchorCache shards 中加载数据，用于 CRaFT 训练的保留损失计算。
    
    【数据格式】
    根据 cache 类型，每个样本包含不同字段：
    
    基础字段（所有类型都有）：
    - pixel_values: Tensor[C, H, W], 图像，float32，[-1, 1]
    - input_ids: Tensor[seq_len], 完整输入序列
    - attention_mask: Tensor[seq_len], 注意力掩码
    
    类型特定字段：
    - Token-level: labels (Tensor[seq_len])
    - Hidden state: teacher_hidden (Tensor[n_layers, n_vecs, hidden_dim]), meta (dict)
    - Hidden feature: target_features (Tensor[hidden_dim]), meta (dict)
    
    【参数】
    cache_dir: str | Path
        AnchorCache 目录路径，包含 shard_*.pt 文件和 metadata.json
    
    【使用示例】
    >>> from lerobot.craft.anchor_cache import AnchorCacheDataset
    >>> 
    >>> # 创建数据集
    >>> anchor_dataset = AnchorCacheDataset(cache_dir="data/anchor_cache")
    >>> 
    >>> # 访问样本
    >>> sample = anchor_dataset[0]
    >>> print(sample.keys())  # dict_keys(['pixel_values', 'input_ids', 'attention_mask', 'labels'])
    >>> 
    >>> # 数据集大小
    >>> print(len(anchor_dataset))  # 1000
    """
    
    def __init__(self, cache_dir: str | Path):
        """
        初始化 AnchorCache 数据集
        
        【参数】
        cache_dir: AnchorCache 目录路径
        """
        self.cache_dir = Path(cache_dir)
        
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"AnchorCache 目录不存在: {self.cache_dir}")
        
        # 加载元数据
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_anchors = self.metadata["num_anchors"]
        self.num_shards = self.metadata["num_shards"]
        self.shard_size = self.metadata["shard_size"]
        
        # 构建 shard 文件列表
        self.shard_files = []
        for shard_idx in range(self.num_shards):
            shard_path = self.cache_dir / f"shard_{shard_idx:04d}.pt"
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard 文件不存在: {shard_path}")
            self.shard_files.append(shard_path)
        
        # 缓存当前加载的 shard（避免重复加载）
        self._current_shard_idx = None
        self._current_shard_data = None
        
        logging.info(f"加载 AnchorCache: {self.cache_dir}")
        logging.info(f"  - 总样本数: {self.num_anchors}")
        logging.info(f"  - Shard 数量: {self.num_shards}")
        logging.info(f"  - Shard 大小: {self.shard_size}")
    
    def __len__(self):
        """
        返回数据集大小
        
        【返回值】
        int: 锚点数据集中的样本总数
        """
        return self.num_anchors
    
    def _load_shard(self, shard_idx: int) -> dict:
        """
        加载指定的 shard
        
        【参数】
        shard_idx: Shard 索引
        
        【返回值】
        dict: Shard 数据
        """
        if self._current_shard_idx == shard_idx:
            return self._current_shard_data
        
        shard_path = self.shard_files[shard_idx]
        shard_data = torch.load(shard_path, map_location="cpu")
        
        self._current_shard_idx = shard_idx
        self._current_shard_data = shard_data
        
        return shard_data
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取指定索引的样本
        
        【参数】
        idx: int, 样本索引
        
        【返回值】
        dict: 包含 pixel_values, input_ids, attention_mask, teacher_hidden, meta 的样本字典
              （如果是旧版本 cache，则包含 labels 而非 teacher_hidden）
        """
        if idx < 0 or idx >= self.num_anchors:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_anchors})")
        
        # 确定样本所在的 shard
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        
        # 加载 shard
        shard_data = self._load_shard(shard_idx)
        
        # 提取样本（基础字段）
        sample = {
            "pixel_values": shard_data["pixel_values"][local_idx],
            "input_ids": shard_data["input_ids"][local_idx],
            "attention_mask": shard_data["attention_mask"][local_idx],
        }
        
        # 检测 cache 类型并添加相应字段
        if "target_features" in shard_data:
            # Hidden feature cache（pooled vector）
            sample["target_features"] = shard_data["target_features"][local_idx]
            sample["meta"] = shard_data["meta"]
        elif "teacher_hidden" in shard_data:
            # Hidden state cache（多层多向量）
            sample["teacher_hidden"] = shard_data["teacher_hidden"][local_idx]
            sample["meta"] = shard_data["meta"]
        elif "labels" in shard_data:
            # Token-level cache（旧版本，向后兼容）
            sample["labels"] = shard_data["labels"][local_idx]
        else:
            raise ValueError("AnchorCache 格式错误：没有 target_features、teacher_hidden 或 labels")
        
        return sample


def build_anchor_dataloader(
    cache_dir: str | Path,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建 AnchorCache 的 DataLoader（推荐使用此函数）
    
    【功能说明】
    封装 AnchorCacheDataset 和 DataLoader 的创建逻辑，提供一站式接口。
    
    【参数】
    cache_dir: str | Path
        AnchorCache 目录路径，包含 shard_*.pt 文件和 metadata.json
        
    batch_size: int
        批次大小
        - 建议设置为任务数据批次大小的 50%-100%
        - 例如: 任务批次 32，锚点批次可设为 16
        
    num_workers: int = 4
        数据加载的工作进程数
        - 0: 主进程加载（简单但慢）
        - 4-8: 多进程加载（推荐）
        - 过多可能导致内存占用过高
        
    shuffle: bool = True
        是否随机打乱数据
        - True: 每个 epoch 随机采样（推荐）
        - False: 顺序采样（用于调试）
        
    pin_memory: bool = True
        是否将数据固定在内存中（加速 GPU 传输）
        - True: 使用 CUDA 时推荐
        - False: CPU 训练时使用
    
    【返回值】
    DataLoader: PyTorch DataLoader 实例
        - 可迭代对象，每次返回一个批次
        - 支持多进程加载和预取
    
    【使用示例】
    >>> from lerobot.craft.anchor_cache import build_anchor_dataloader
    >>> from lerobot.datasets.utils import cycle
    >>> 
    >>> # 创建 DataLoader
    >>> anchor_dataloader = build_anchor_dataloader(
    ...     cache_dir="data/anchor_cache",
    ...     batch_size=16,
    ...     num_workers=4,
    ...     shuffle=True
    ... )
    >>> 
    >>> # 创建无限迭代器（训练循环常用）
    >>> anchor_dl_iter = cycle(anchor_dataloader)
    >>> 
    >>> # 在训练循环中使用
    >>> for step in range(10000):
    ...     anchor_batch = next(anchor_dl_iter)
    ...     # anchor_batch 包含: pixel_values, input_ids, attention_mask, labels
    ...     print(anchor_batch["pixel_values"].shape)  # torch.Size([16, 3, 224, 224])
    
    【性能优化】
    1. 批次大小: 根据 GPU 内存调整，通常为任务批次的 50%-100%
    2. 工作进程: 4-8 个通常足够，过多会增加内存开销
    3. 预取因子: prefetch_factor=2 可以隐藏数据加载延迟
    4. 固定内存: pin_memory=True 加速 CPU->GPU 传输
    
    【与任务 DataLoader 的对比】
    ```python
    # 任务数据 DataLoader（新任务）
    task_dataloader = DataLoader(
        task_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # 锚点数据 DataLoader（旧任务）
    anchor_dataloader = build_anchor_dataloader(
        cache_dir="data/anchor_cache",
        batch_size=16,  # 通常较小
        num_workers=4,
        shuffle=True
    )
    
    # 两者格式不同：anchor 包含预生成的 teacher outputs
    task_batch = next(iter(task_dataloader))
    anchor_batch = next(iter(anchor_dataloader))
    # anchor_batch: {"pixel_values", "input_ids", "attention_mask", "labels"}
    ```
    """
    # 创建数据集
    anchor_dataset = AnchorCacheDataset(cache_dir)
    
    # 创建 DataLoader
    dataloader = DataLoader(
        anchor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 保留最后不完整的批次
        prefetch_factor=2 if num_workers > 0 else None,  # 预取 2 个批次
    )
    
    logging.info(f"创建 AnchorCache DataLoader:")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Num workers: {num_workers}")
    logging.info(f"  - Shuffle: {shuffle}")
    logging.info(f"  - Pin memory: {pin_memory}")
    
    return dataloader

