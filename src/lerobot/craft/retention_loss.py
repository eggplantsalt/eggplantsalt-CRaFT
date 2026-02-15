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
保留损失计算模块 (Retention Loss)
=================================

【模块功能】
在锚点/旧任务数据上计算保留损失，衡量模型对已学习知识的记忆程度。

【核心思想】
保留损失 (Retention Loss) 用于评估模型在学习新任务后，是否仍能保持
对旧任务的性能。它与任务损失使用相同的损失函数，但在不同的数据集上计算：
- 任务损失: 在新任务数据上计算（前向学习）
- 保留损失: 在锚点数据上计算（后向记忆）

【数学定义】
L_retain = E_{(x,y)~D_anchor} [Loss(policy(x), y)]

其中：
- D_anchor: 锚点数据集（旧任务的代表性样本）
- Loss: 与任务损失相同的损失函数（如 MSE、CrossEntropy）
- policy(x): 模型在输入 x 上的预测

【使用场景】
1. 持续学习 (Continual Learning): 防止灾难性遗忘
2. 领域适应 (Domain Adaptation): 保持源域性能
3. 微调 (Fine-tuning): 避免过度拟合新数据

【使用示例】
```python
from lerobot.craft.retention_loss import compute_retention_loss

# 在训练循环中
for step in range(total_steps):
    # 1. 计算任务损失（新数据）
    task_batch = next(task_dataloader)
    task_loss, _ = policy.forward(task_batch)
    
    # 2. 计算保留损失（锚点数据）
    anchor_batch = next(anchor_dataloader)
    retention_loss = compute_retention_loss(policy, anchor_batch)
    
    # 3. 检查约束是否满足
    if retention_loss > epsilon:
        print(f"警告: 保留损失 {retention_loss:.3f} 超过阈值 {epsilon:.3f}")
```

【设计原则】
- 简单性: 复用 policy.forward() 的损失计算逻辑
- 一致性: 与任务损失使用相同的度量标准
- 高效性: 无需额外的模型或计算开销
"""

import torch
from torch import Tensor


def compute_retention_loss(
    policy,
    anchor_batch: dict,
    reduction: str = "mean",
) -> Tensor:
    """
    在锚点数据批次上计算保留损失
    
    【功能说明】
    保留损失衡量模型在旧任务/锚点数据上的性能。它使用与任务损失
    相同的损失函数，但在不同的数据集（锚点数据）上计算。
    
    【实现原理】
    直接调用 policy.forward() 方法，该方法已经实现了完整的损失计算逻辑：
    1. 前向传播: 模型预测
    2. 损失计算: 预测与标签的差异
    3. 返回损失值
    
    因此，本函数只是一个简单的封装，确保在锚点数据上调用相同的损失计算。
    
    【参数】
    policy: PreTrainedPolicy
        策略模型（必须有 forward 方法返回损失）
        - 例如: PI0FastPolicy, ACTPolicy, DiffusionPolicy
        - forward() 签名: forward(batch) -> (loss, output_dict)
        
    anchor_batch: dict
        锚点数据批次（格式与训练批次相同）
        - 必须包含模型所需的所有输入键
        - 例如: {"observation.images": ..., "observation.state": ..., "action": ...}
        
    reduction: str = "mean"
        损失归约模式
        - "mean": 返回批次平均损失（推荐）
        - "sum": 返回批次总损失
        - "none": 返回每个样本的损失（用于加权）
    
    【返回值】
    Tensor: 保留损失标量张量
        - 标量值（如果 reduction="mean" 或 "sum"）
        - 向量值（如果 reduction="none"）
    
    【实现提示】
    这个函数的实现非常简单，因为 policy.forward() 已经完成了所有工作：
    
    ```python
    # 方法 1: 直接调用 forward（推荐）
    loss, _ = policy.forward(anchor_batch)
    return loss
    
    # 方法 2: 如果需要 per-sample loss
    if reduction == "none":
        loss, _ = policy.forward(anchor_batch, reduction="none")
    else:
        loss, _ = policy.forward(anchor_batch, reduction=reduction)
    return loss
    ```
    
    【注意事项】
    1. 确保 policy 处于训练模式: policy.train()
       - 虽然是在锚点数据上评估，但仍需要梯度用于反向传播
       
    2. 锚点批次格式必须与训练批次一致
       - 相同的键名和张量形状
       - 已经过预处理（preprocessor）
       
    3. 不要在此函数内调用 backward()
       - 损失计算和反向传播分离
       - 反向传播在训练循环中统一处理
    
    【示例】
    >>> # 假设 policy 和 anchor_batch 已准备好
    >>> policy.train()  # 确保处于训练模式
    >>> 
    >>> # 计算保留损失
    >>> retention_loss = compute_retention_loss(policy, anchor_batch)
    >>> print(f"保留损失: {retention_loss.item():.4f}")
    >>> 
    >>> # 用于梯度计算
    >>> retention_loss.backward()  # 在训练循环中调用
    >>> 
    >>> # Per-sample loss（用于加权）
    >>> per_sample_loss = compute_retention_loss(policy, anchor_batch, reduction="none")
    >>> print(f"批次中每个样本的损失: {per_sample_loss}")
    
    【与任务损失的关系】
    ```python
    # 任务损失（新数据）
    task_loss, _ = policy.forward(task_batch)
    
    # 保留损失（锚点数据）- 使用相同的 forward 方法
    retention_loss = compute_retention_loss(policy, anchor_batch)
    
    # 两者使用相同的损失函数，只是数据来源不同
    assert task_loss.shape == retention_loss.shape  # 都是标量
    ```
    
    【调试技巧】
    如果保留损失异常（过高或过低），检查：
    1. 锚点数据是否正确加载和预处理
    2. 锚点数据分布是否与训练数据一致
    3. 模型是否在训练模式（policy.train()）
    4. 批次大小是否合理（太小可能不稳定）
    
    TODO: 在下一阶段实现此函数（实现非常简单，只需调用 policy.forward）
    """
    raise NotImplementedError("compute_retention_loss: 待在下一阶段实现")

