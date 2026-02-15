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
原对偶优化模块 (Primal-Dual Optimization)
=========================================

【模块功能】
实现 Lagrangian 乘子更新和约束阈值调度，用于约束优化问题的求解。

【优化问题】
原问题 (Primal):
    min L_task(θ)
    s.t. L_retain(θ) ≤ ε(t)

对偶问题 (Dual):
    max_λ min_θ L_task(θ) + λ * (L_retain(θ) - ε(t))

【算法原理】
1. 原变量更新（θ）: 通过梯度下降最小化 Lagrangian
   θ ← θ - lr * ∇[L_task + λ * (L_retain - ε)]

2. 对偶变量更新（λ）: 通过梯度上升最大化对偶函数
   λ ← clip(λ + λ_lr * (L_retain - ε), 0, λ_max)

【直观理解】
- 如果 L_retain > ε: 违反约束 → 增大 λ → 更重视保留损失
- 如果 L_retain < ε: 满足约束 → 减小 λ → 更重视任务损失
- λ 自动调节两个目标的平衡点

【使用示例】
```python
from lerobot.craft.primal_dual import epsilon_schedule, update_lambda

# 初始化
lambda_weight = 1.0
epsilon_start, epsilon_end = 1.0, 0.1
decay_steps = 10000

# 训练循环
for step in range(total_steps):
    # 1. 计算当前 epsilon
    epsilon = epsilon_schedule(
        step, epsilon_start, epsilon_end, decay_steps, schedule_type="linear"
    )
    
    # 2. 前向传播和损失计算
    task_loss = compute_task_loss(...)
    retention_loss = compute_retention_loss(...)
    
    # 3. 更新 lambda
    lambda_weight = update_lambda(
        lambda_weight, retention_loss, epsilon, lambda_lr=0.01, lambda_max=10.0
    )
```

【参考文献】
- Boyd & Vandenberghe, "Convex Optimization" (2004), Chapter 5
- Bertsekas, "Constrained Optimization and Lagrange Multiplier Methods" (1982)
"""

import torch


def epsilon_schedule(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_steps: int,
    schedule_type: str = "linear",
) -> float:
    """
    计算当前训练步的保留损失阈值 ε(t)
    
    【功能说明】
    阈值从 epsilon_start（宽松）退火到 epsilon_end（严格），
    允许模型在训练初期专注于新任务学习，后期逐渐加强保留约束。
    
    【退火策略】
    1. Linear（线性）:
       ε(t) = ε_start - (ε_start - ε_end) * (t / T)
       - 均匀递减
       - 简单稳定
       
    2. Cosine（余弦）:
       ε(t) = ε_end + 0.5 * (ε_start - ε_end) * (1 + cos(π * t / T))
       - 初期快速下降，后期缓慢
       - 平滑过渡
       
    3. Exponential（指数）:
       ε(t) = ε_end + (ε_start - ε_end) * exp(-5 * t / T)
       - 初期极快下降
       - 适合快速收紧约束
    
    【参数】
    step: int
        当前训练步数（从 0 开始）
        
    epsilon_start: float
        起始阈值（训练初期，宽松约束）
        - 典型值: 1.0 到 2.0
        - 允许较大的保留损失
        
    epsilon_end: float
        最终阈值（训练后期，严格约束）
        - 典型值: 0.05 到 0.2
        - 要求严格的性能保持
        
    decay_steps: int
        完整退火所需的步数
        - 通常设置为总训练步数的 50%-100%
        - 例如: 总步数 20000，decay_steps 可设为 15000
        
    schedule_type: str = "linear"
        退火策略类型
        - "linear": 线性退火（推荐，稳定）
        - "cosine": 余弦退火（平滑）
        - "exponential": 指数退火（激进）
    
    【返回值】
    float: 当前步的 epsilon 值
        - 范围: [epsilon_end, epsilon_start]
        - 随训练步数单调递减
    
    【实现提示】
    ```python
    # 计算进度比例
    progress = min(step / decay_steps, 1.0)  # 限制在 [0, 1]
    
    if schedule_type == "linear":
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress
    elif schedule_type == "cosine":
        import math
        epsilon = epsilon_end + 0.5 * (epsilon_start - epsilon_end) * (1 + math.cos(math.pi * progress))
    elif schedule_type == "exponential":
        import math
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-5 * progress)
    
    return epsilon
    ```
    
    【示例】
    >>> # 线性退火
    >>> eps_0 = epsilon_schedule(0, 1.0, 0.1, 1000, "linear")
    >>> print(eps_0)  # 1.0 (起始值)
    >>> 
    >>> eps_500 = epsilon_schedule(500, 1.0, 0.1, 1000, "linear")
    >>> print(eps_500)  # 0.55 (中间值)
    >>> 
    >>> eps_1000 = epsilon_schedule(1000, 1.0, 0.1, 1000, "linear")
    >>> print(eps_1000)  # 0.1 (最终值)
    
    TODO: 在下一阶段实现此函数
    """
    raise NotImplementedError("epsilon_schedule: 待在下一阶段实现")


def update_lambda(
    current_lambda: float,
    retention_loss: float,
    epsilon: float,
    lambda_lr: float,
    lambda_max: float,
) -> float:
    """
    更新 Lagrangian 乘子 λ（对偶变量的梯度上升）
    
    【功能说明】
    根据保留损失是否违反约束，动态调整 λ 的值。
    λ 控制保留损失在总损失中的权重。
    
    【更新规则】
    λ_{t+1} = clip(λ_t + λ_lr * (L_retain - ε), 0, λ_max)
    
    【直观理解】
    - L_retain > ε (违反约束):
      * (L_retain - ε) > 0
      * λ 增大 → 下一步更重视保留损失
      * 模型被"惩罚"，需要改善旧任务性能
      
    - L_retain < ε (满足约束):
      * (L_retain - ε) < 0
      * λ 减小 → 下一步更重视任务损失
      * 模型可以更专注于新任务学习
      
    - L_retain ≈ ε (约束边界):
      * λ 保持稳定
      * 达到理想的平衡状态
    
    【参数】
    current_lambda: float
        当前的 λ 值
        - 初始值通常设为 1.0
        - 训练过程中动态调整
        
    retention_loss: float
        当前步的保留损失值
        - 在锚点数据上计算得到
        - 衡量对旧任务的记忆程度
        
    epsilon: float
        当前步的保留损失阈值
        - 由 epsilon_schedule() 计算得到
        - 定义"可接受"的保留损失上界
        
    lambda_lr: float
        λ 的学习率（更新步长）
        - 典型值: 0.001 到 0.1
        - 过大可能导致震荡，过小收敛慢
        
    lambda_max: float
        λ 的最大允许值（上界）
        - 典型值: 5.0 到 20.0
        - 防止 λ 无界增长导致训练不稳定
    
    【返回值】
    float: 更新后的 λ 值
        - 范围: [0, lambda_max]
        - 非负约束确保不会"负惩罚"
    
    【实现提示】
    ```python
    # 计算约束违反程度
    constraint_violation = retention_loss - epsilon
    
    # 梯度上升更新
    new_lambda = current_lambda + lambda_lr * constraint_violation
    
    # 裁剪到合法范围
    new_lambda = max(0.0, min(new_lambda, lambda_max))
    
    return new_lambda
    ```
    
    【示例】
    >>> # 情况 1: 违反约束（保留损失过高）
    >>> lambda_new = update_lambda(
    ...     current_lambda=1.0,
    ...     retention_loss=1.5,  # 高于阈值
    ...     epsilon=1.0,
    ...     lambda_lr=0.1,
    ...     lambda_max=10.0
    ... )
    >>> print(lambda_new)  # 1.05 = 1.0 + 0.1 * (1.5 - 1.0)
    >>> # λ 增大，下一步更重视保留
    >>> 
    >>> # 情况 2: 满足约束（保留损失较低）
    >>> lambda_new = update_lambda(
    ...     current_lambda=1.0,
    ...     retention_loss=0.5,  # 低于阈值
    ...     epsilon=1.0,
    ...     lambda_lr=0.1,
    ...     lambda_max=10.0
    ... )
    >>> print(lambda_new)  # 0.95 = 1.0 + 0.1 * (0.5 - 1.0)
    >>> # λ 减小，下一步更重视任务学习
    >>> 
    >>> # 情况 3: 边界裁剪
    >>> lambda_new = update_lambda(
    ...     current_lambda=9.8,
    ...     retention_loss=3.0,
    ...     epsilon=1.0,
    ...     lambda_lr=0.5,
    ...     lambda_max=10.0
    ... )
    >>> print(lambda_new)  # 10.0 (裁剪到上界)
    
    TODO: 在下一阶段实现此函数
    """
    raise NotImplementedError("update_lambda: 待在下一阶段实现")

