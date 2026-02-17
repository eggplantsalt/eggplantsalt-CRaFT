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
梯度手术模块 (Gradient Surgery)
================================

【模块功能】
实现多目标优化中的梯度投影和冲突解决算法，基于 PCGrad 和相关梯度手术技术。

【核心算法】
当任务梯度 g_task 和保留梯度 g_retain 发生冲突（负点积）时：
1. 检测冲突: cos(g_task, g_retain) = dot(g_task, g_retain) / (||g_task|| * ||g_retain||)
2. 投影修正: g_task_proj = g_task - proj_{g_retain}(g_task)
3. 合并梯度: g_final = g_task_proj + λ * g_retain

【数学原理】
投影公式（将 g_task 投影到 g_retain 的法平面）：
    g_task_proj = g_task - (dot(g_task, g_retain) / ||g_retain||²) * g_retain

这样可以确保：
- g_task_proj 与 g_retain 正交（无冲突）
- 保留 g_task 在法平面上的分量（尽可能保持任务学习方向）

【使用示例】
```python
from lerobot.craft.grad_surgery import compute_dot, project_if_conflict, merge_grads

# 1. 计算梯度点积
dot_product = compute_dot(grad_task, grad_retain)

# 2. 如果冲突则投影
if dot_product < -0.1:
    grad_task_proj, grad_retain_proj, conflict = project_if_conflict(
        grad_task, grad_retain, conflict_threshold=-0.1
    )
else:
    grad_task_proj, grad_retain_proj = grad_task, grad_retain

# 3. 合并梯度
final_grad = merge_grads(grad_task_proj, grad_retain_proj, lambda_weight=2.0)
```

【参考文献】
- PCGrad: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
- CAGrad: Liu et al. "Conflict-Averse Gradient Descent for Multi-task Learning" (NeurIPS 2021)
"""

import torch
from torch import Tensor


def compute_dot(grad1: list[Tensor], grad2: list[Tensor]) -> Tensor:
    """
    计算两个梯度向量的点积（用于判断梯度方向是否冲突）
    
    【功能说明】
    将两个梯度列表（每个参数一个张量）同位置配对并计算点积。
    点积的符号表示梯度方向的关系：
    - 正值: 梯度方向一致（协同优化）
    - 负值: 梯度方向冲突（需要投影）
    - 接近零: 梯度正交（互不干扰）
    
    【参数】
    grad1: list[Tensor]
        目标 1 的梯度列表（每个模型参数对应一个梯度张量）
        例如: [grad_layer1, grad_layer2, ...]
        
    grad2: list[Tensor]
        目标 2 的梯度列表（结构与 grad1 相同）
    
    【返回值】
    Tensor: 标量张量，表示两个梯度向量的点积
        - 正值表示方向一致
        - 负值表示方向冲突
    
    【实现细节】
    使用 zip 同位置配对，跳过任一为 None 的项，避免错位问题。
    
    【示例】
    >>> grad_task = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    >>> grad_retain = [torch.tensor([1.0, -1.0]), torch.tensor([0.5])]
    >>> dot = compute_dot(grad_task, grad_retain)
    >>> print(dot)  # 1*1 + 2*(-1) + 3*0.5 = 0.5
    """
    # 使用 zip 同位置配对，累加点积（跳过任一为 None 的项）
    dot_sum = 0.0
    for g1, g2 in zip(grad1, grad2):
        if g1 is not None and g2 is not None:
            dot_sum += torch.sum(g1 * g2)
    
    return torch.tensor(dot_sum, device=grad1[0].device if grad1[0] is not None else "cpu")


def project_if_conflict(
    grad_task: list[Tensor],
    grad_retain: list[Tensor],
    conflict_threshold: float = -0.1,
) -> tuple[list[Tensor], list[Tensor], bool]:
    """
    当梯度冲突时进行投影修正（基于 PCGrad 算法）
    
    【功能说明】
    检测任务梯度和保留梯度是否冲突，如果冲突则将任务梯度投影到
    保留梯度的法平面上，消除冲突方向的分量。
    
    【算法流程】
    1. 计算点积: dot = <g_task, g_retain>
    2. 判断冲突: if dot < conflict_threshold
    3. 如果冲突:
       - 计算投影: proj = (dot / ||g_retain||²) * g_retain
       - 修正梯度: g_task_new = g_task - proj
    4. 如果不冲突: 保持原梯度不变
    
    【参数】
    grad_task: list[Tensor]
        任务损失的梯度（新任务学习方向）
        
    grad_retain: list[Tensor]
        保留损失的梯度（旧任务记忆方向）
        
    conflict_threshold: float = -0.1
        冲突检测阈值（余弦相似度）
        - 典型值: -0.1 到 -0.3
        - 越负表示越严格的冲突判定
    
    【返回值】
    tuple[list[Tensor], list[Tensor], bool]:
        - projected_grad_task: 投影后的任务梯度
        - projected_grad_retain: 投影后的保留梯度（通常不变）
        - conflict_detected: 是否检测到冲突
    
    【数学公式】
    投影公式（将 a 投影到 b 的法平面）：
        a_proj = a - (<a, b> / ||b||²) * b
    
    其中：
        <a, b> = dot(a, b)  # 点积
        ||b||² = dot(b, b)  # 范数平方
    
    【实现提示】
    1. 调用 compute_dot() 计算点积
    2. 如果 dot < conflict_threshold:
       - 计算 ||g_retain||²
       - 计算投影系数: coef = dot / ||g_retain||²
       - 逐参数投影: g_task[i] = g_task[i] - coef * g_retain[i]
    3. 返回投影后的梯度和冲突标志
    
    【示例】
    >>> # 冲突情况（负点积）
    >>> grad_task = [torch.tensor([1.0, 0.0])]
    >>> grad_retain = [torch.tensor([-1.0, 0.0])]
    >>> proj_task, proj_retain, conflict = project_if_conflict(
    ...     grad_task, grad_retain, conflict_threshold=-0.1
    ... )
    >>> print(conflict)  # True
    >>> print(proj_task)  # [tensor([0.0, 0.0])] - 完全投影掉
    """
    # 计算点积
    dot_product = compute_dot(grad_task, grad_retain)
    
    # 检测冲突（负点积表示方向相反）
    if dot_product < conflict_threshold:
        # 计算 ||g_retain||²
        norm_squared = compute_dot(grad_retain, grad_retain)
        
        # 避免除零
        if norm_squared < 1e-12:
            return grad_task, grad_retain, False
        
        # 计算投影系数
        projection_coef = dot_product / norm_squared
        
        # 投影：g_task_proj = g_task - (dot / ||g_retain||²) * g_retain
        projected_grad_task = [
            g_t - projection_coef * g_r if g_t is not None and g_r is not None else g_t
            for g_t, g_r in zip(grad_task, grad_retain)
        ]
        
        return projected_grad_task, grad_retain, True
    else:
        # 无冲突，保持原梯度
        return grad_task, grad_retain, False


def merge_grads(
    grad_task: list[Tensor],
    grad_retain: list[Tensor],
    lambda_weight: float,
    mode: str = "weighted",
) -> list[Tensor]:
    """
    合并任务梯度和保留梯度（根据指定的合并策略）
    
    【功能说明】
    将投影后的任务梯度和保留梯度按照指定模式合并为最终梯度，
    用于优化器更新。
    
    【合并模式】
    1. "weighted" (推荐):
       g_final = g_task + λ * g_retain
       - λ 由原对偶优化动态调整
       - 当保留约束被违反时，λ 增大，更重视保留
       
    2. "equal":
       g_final = 0.5 * (g_task + g_retain)
       - 平等对待两个目标
       - 适用于两个任务同等重要的场景
       
    3. "task_priority":
       g_final = g_task + min(λ, 1.0) * g_retain
       - 限制保留梯度的权重不超过任务梯度
       - 确保新任务学习不被过度抑制
    
    【参数】
    grad_task: list[Tensor]
        任务损失的梯度（已投影）
        
    grad_retain: list[Tensor]
        保留损失的梯度（已投影）
        
    lambda_weight: float
        Lagrangian 乘子 λ（保留损失权重）
        - 由 primal_dual.update_lambda() 动态更新
        - 典型范围: 0.1 到 10.0
        
    mode: str = "weighted"
        合并策略，可选值：
        - "weighted": 加权合并（推荐）
        - "equal": 平均合并
        - "task_priority": 任务优先合并
    
    【返回值】
    list[Tensor]: 合并后的最终梯度列表
        - 结构与输入梯度相同
        - 可直接用于 optimizer.step()
    
    【实现提示】
    根据 mode 选择合并公式：
    ```python
    if mode == "weighted":
        merged = [g_t + lambda_weight * g_r for g_t, g_r in zip(grad_task, grad_retain)]
    elif mode == "equal":
        merged = [0.5 * (g_t + g_r) for g_t, g_r in zip(grad_task, grad_retain)]
    elif mode == "task_priority":
        capped_lambda = min(lambda_weight, 1.0)
        merged = [g_t + capped_lambda * g_r for g_t, g_r in zip(grad_task, grad_retain)]
    ```
    
    【示例】
    >>> grad_task = [torch.tensor([1.0, 2.0])]
    >>> grad_retain = [torch.tensor([0.5, -0.5])]
    >>> 
    >>> # Weighted 模式
    >>> merged = merge_grads(grad_task, grad_retain, lambda_weight=2.0, mode="weighted")
    >>> print(merged)  # [tensor([2.0, 1.0])] = [1.0, 2.0] + 2.0 * [0.5, -0.5]
    >>> 
    >>> # Equal 模式
    >>> merged = merge_grads(grad_task, grad_retain, lambda_weight=2.0, mode="equal")
    >>> print(merged)  # [tensor([0.75, 0.75])] = 0.5 * ([1.0, 2.0] + [0.5, -0.5])
    """
    if mode == "weighted":
        # g_final = g_task + λ * g_retain
        merged = [
            g_t + lambda_weight * g_r if g_t is not None and g_r is not None else g_t
            for g_t, g_r in zip(grad_task, grad_retain)
        ]
    elif mode == "equal":
        # g_final = 0.5 * (g_task + g_retain)
        merged = [
            0.5 * (g_t + g_r) if g_t is not None and g_r is not None else g_t
            for g_t, g_r in zip(grad_task, grad_retain)
        ]
    elif mode == "task_priority":
        # g_final = g_task + min(λ, 1.0) * g_retain
        capped_lambda = min(lambda_weight, 1.0)
        merged = [
            g_t + capped_lambda * g_r if g_t is not None and g_r is not None else g_t
            for g_t, g_r in zip(grad_task, grad_retain)
        ]
    else:
        raise ValueError(f"Unknown merge mode: {mode}. Must be one of ['weighted', 'equal', 'task_priority']")
    
    return merged

