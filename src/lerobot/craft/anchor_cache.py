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
锚点数据缓存模块 (Anchor Cache)
================================

【模块功能】
管理锚点/保留数据集的加载和采样，用于 CRaFT 训练中的保留损失计算。

【核心概念】
锚点数据 (Anchor Data):
- 旧任务的代表性样本，用于防止灾难性遗忘
- 通常是从之前训练过的任务中采样的子集
- 数量通常是新任务数据的 10%-50%

【数据来源】
1. 历史训练数据: 从之前的训练集中随机采样
2. 验证集: 使用验证集作为锚点数据
3. 合成数据: 通过数据增强生成的样本
4. 核心集 (Coreset): 使用聚类等方法选择的代表性样本

【使用示例】
```python
from lerobot.craft.anchor_cache import AnchorCacheDataset, create_anchor_dataloader

# 方法 1: 创建数据集
anchor_dataset = AnchorCacheDataset(
    dataset_path="lerobot/aloha_sim_insertion_human",
    transform=preprocessor
)

# 方法 2: 直接创建 DataLoader（推荐）
anchor_dataloader = create_anchor_dataloader(
    dataset_path="lerobot/aloha_sim_insertion_human",
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# 在训练循环中使用
from lerobot.datasets.utils import cycle
anchor_dl_iter = cycle(anchor_dataloader)

for step in range(total_steps):
    anchor_batch = next(anchor_dl_iter)
    retention_loss = compute_retention_loss(policy, anchor_batch)
```

【设计考虑】
1. 内存效率: 支持流式加载，不需要全部加载到内存
2. 采样策略: 支持随机采样和重要性采样
3. 数据格式: 与训练数据格式完全一致
4. 预处理: 复用训练数据的预处理流程
"""

import torch
from torch.utils.data import Dataset, DataLoader


class AnchorCacheDataset(Dataset):
    """
    锚点数据集包装器
    
    【功能说明】
    封装锚点/保留数据的加载和访问逻辑，提供与 PyTorch Dataset 一致的接口。
    
    【数据格式】
    锚点数据应与训练数据格式完全一致，包含相同的键：
    - observation.images: 图像观测
    - observation.state: 状态向量
    - action: 动作标签
    - episode_index: 轨迹索引（可选）
    - frame_index: 帧索引（可选）
    
    【实现策略】
    1. 简单封装: 直接使用 LeRobot 的 make_dataset() 加载
    2. 子集采样: 从完整数据集中采样固定比例
    3. 缓存优化: 预加载常用样本到内存
    
    【参数】
    dataset_path: str
        锚点数据集路径
        - HuggingFace dataset: "lerobot/aloha_sim_insertion_human"
        - 本地路径: "/path/to/anchor/dataset"
        
    transform: callable, optional
        数据预处理函数（通常是 preprocessor）
        - 应与训练数据使用相同的预处理
        - 包括归一化、图像变换等
    
    【使用示例】
    >>> from lerobot.craft.anchor_cache import AnchorCacheDataset
    >>> 
    >>> # 创建数据集
    >>> anchor_dataset = AnchorCacheDataset(
    ...     dataset_path="lerobot/aloha_sim_insertion_human",
    ...     transform=None  # 预处理在 DataLoader 外部进行
    ... )
    >>> 
    >>> # 访问样本
    >>> sample = anchor_dataset[0]
    >>> print(sample.keys())  # dict_keys(['observation.images', 'observation.state', 'action'])
    >>> 
    >>> # 数据集大小
    >>> print(len(anchor_dataset))  # 1000
    
    【实现提示】
    ```python
    def __init__(self, dataset_path: str, transform=None):
        from lerobot.datasets.factory import make_dataset
        from lerobot.configs.train import DatasetConfig
        
        # 创建数据集配置
        dataset_cfg = DatasetConfig(repo_id=dataset_path)
        
        # 加载数据集
        self.dataset = make_dataset(dataset_cfg)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    ```
    
    TODO: 在下一阶段实现此类
    """
    
    def __init__(self, dataset_path: str, transform=None):
        """
        初始化锚点数据集
        
        【参数】
        dataset_path: 数据集路径（HuggingFace repo_id 或本地路径）
        transform: 可选的数据预处理函数
        """
        self.dataset_path = dataset_path
        self.transform = transform
        # TODO: 在下一阶段加载数据集
        raise NotImplementedError("AnchorCacheDataset.__init__: 待在下一阶段实现")
    
    def __len__(self):
        """
        返回数据集大小
        
        【返回值】
        int: 锚点数据集中的样本总数
        """
        raise NotImplementedError("AnchorCacheDataset.__len__: 待在下一阶段实现")
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        【参数】
        idx: int, 样本索引
        
        【返回值】
        dict: 包含观测、动作等的样本字典
        """
        raise NotImplementedError("AnchorCacheDataset.__getitem__: 待在下一阶段实现")


def create_anchor_dataloader(
    dataset_path: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建锚点数据的 DataLoader（推荐使用此函数）
    
    【功能说明】
    封装 AnchorCacheDataset 和 DataLoader 的创建逻辑，提供一站式接口。
    
    【参数】
    dataset_path: str
        锚点数据集路径
        - HuggingFace dataset: "lerobot/aloha_sim_insertion_human"
        - 本地路径: "/path/to/anchor/dataset"
        
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
    >>> from lerobot.craft.anchor_cache import create_anchor_dataloader
    >>> from lerobot.datasets.utils import cycle
    >>> 
    >>> # 创建 DataLoader
    >>> anchor_dataloader = create_anchor_dataloader(
    ...     dataset_path="lerobot/aloha_sim_insertion_human",
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
    ...     # anchor_batch 是一个字典，包含批次数据
    ...     print(anchor_batch["action"].shape)  # torch.Size([16, 50, 7])
    
    【实现提示】
    ```python
    # 1. 创建数据集
    anchor_dataset = AnchorCacheDataset(dataset_path, transform=None)
    
    # 2. 创建 DataLoader
    dataloader = DataLoader(
        anchor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 保留最后不完整的批次
        prefetch_factor=2 if num_workers > 0 else None,  # 预取 2 个批次
    )
    
    return dataloader
    ```
    
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
    anchor_dataloader = create_anchor_dataloader(
        dataset_path="path/to/anchor",
        batch_size=16,  # 通常较小
        num_workers=4,
        shuffle=True
    )
    
    # 两者格式完全一致，可以用相同的方式处理
    task_batch = next(iter(task_dataloader))
    anchor_batch = next(iter(anchor_dataloader))
    assert task_batch.keys() == anchor_batch.keys()
    ```
    
    TODO: 在下一阶段实现此函数
    """
    raise NotImplementedError("create_anchor_dataloader: 待在下一阶段实现")

