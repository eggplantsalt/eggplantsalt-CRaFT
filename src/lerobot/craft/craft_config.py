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
CRaFT 配置模块
=============

【模块功能】
定义 CRaFT 训练的所有超参数配置，包括：
- 锚点数据集配置
- 损失权重参数（λ）
- 保留约束参数（ε）
- 梯度手术配置
- 日志记录设置

【使用示例】
```python
from lerobot.craft import CraftConfig

# 创建默认配置
config = CraftConfig()

# 自定义配置
config = CraftConfig(
    anchor_dataset_path="data/anchor_dataset",
    initial_lambda=2.0,
    epsilon_start=1.5,
    epsilon_end=0.05,
    use_grad_projection=True,
    conflict_threshold=-0.2
)
```
"""

from dataclasses import dataclass, field


@dataclass
class CraftConfig:
    """
    CRaFT (约束保留微调) 训练配置类
    
    【核心思想】
    通过双目标优化实现持续学习，在学习新任务的同时保持对旧任务的记忆：
    - 任务损失 (L_task): 在新任务数据上的标准监督学习
    - 保留损失 (L_retain): 在锚点/旧任务数据上的性能保持
    - 原对偶优化: 通过 Lagrangian 乘子 λ 动态平衡两个目标
    
    【优化问题】
    min L_task(θ)
    s.t. L_retain(θ) ≤ ε(t)
    
    通过 Lagrangian 方法转化为无约束优化：
    L_total = L_task + λ * max(0, L_retain - ε)
    
    【参数说明】
    
    ## 锚点数据集配置
    anchor_dataset_path: str
        锚点/保留数据集的路径（支持 HuggingFace dataset 或本地路径）
        锚点数据通常是旧任务的代表性样本，用于计算保留损失
        
    anchor_batch_size: int = 16
        锚点数据的批次大小
        建议设置为任务数据批次大小的 50%-100%
        
    anchor_sample_ratio: float = 0.5
        每个训练步采样锚点数据的概率 (0.0-1.0)
        0.5 表示 50% 的步骤会计算保留损失
        设置为 1.0 表示每步都计算（更强的保留约束）
    
    ## 损失权重（原对偶优化）
    initial_lambda: float = 1.0
        Lagrangian 乘子 λ 的初始值
        λ 控制保留损失的权重，越大表示越重视旧任务记忆
        
    lambda_lr: float = 0.01
        λ 的学习率（用于原对偶更新）
        更新规则: λ ← λ + lambda_lr * (L_retain - ε)
        
    lambda_max: float = 10.0
        λ 的最大允许值（防止无界增长）
        当保留损失持续违反约束时，λ 会被限制在此上界
    
    ## 保留约束（ε 调度）
    epsilon_start: float = 1.0
        保留损失阈值 ε 的起始值（训练初期较宽松）
        
    epsilon_end: float = 0.1
        保留损失阈值 ε 的最终值（训练后期较严格）
        
    epsilon_decay_steps: int = 10000
        ε 从 start 退火到 end 的步数
        退火策略允许模型在训练初期专注于新任务，后期逐渐加强保留约束
    
    ## 梯度手术
    use_grad_projection: bool = True
        是否启用梯度投影（当任务梯度和保留梯度冲突时）
        基于 PCGrad 算法：当两个梯度方向冲突（负点积）时，
        将任务梯度投影到保留梯度的法平面上
        
    conflict_threshold: float = -0.1
        梯度冲突检测阈值（余弦相似度）
        负值表示梯度方向相反（冲突）
        典型值: -0.1 到 -0.3
        
    projection_mode: str = "weighted"
        梯度合并模式，可选值：
        - "weighted": g_final = g_task + λ * g_retain（推荐）
        - "equal": g_final = 0.5 * (g_task + g_retain)
        - "task_priority": g_final = g_task + min(λ, 1.0) * g_retain
    
    ## 日志和调试
    log_craft_metrics_freq: int = 100
        记录 CRaFT 特定指标的频率（每 N 步）
        记录内容包括: λ, ε, L_retain, 梯度冲突次数等
        
    save_lambda_history: bool = True
        是否保存 λ 的完整轨迹用于分析
        有助于理解原对偶优化的收敛行为
    """
    
    # ========== 锚点数据集配置 ==========
    anchor_dataset_path: str = ""
    anchor_batch_size: int = 16
    anchor_sample_ratio: float = 0.5  # 50% 的步骤采样锚点数据
    
    # ========== 损失权重（原对偶优化）==========
    initial_lambda: float = 1.0
    lambda_lr: float = 0.01
    lambda_max: float = 10.0
    
    # ========== 保留约束（ε 调度）==========
    epsilon_start: float = 1.0  # 训练初期：宽松约束
    epsilon_end: float = 0.1    # 训练后期：严格约束
    epsilon_decay_steps: int = 10000
    
    # ========== 梯度手术 ==========
    use_grad_projection: bool = True
    conflict_threshold: float = -0.1  # 负余弦相似度表示冲突
    projection_mode: str = "weighted"  # 可选: "weighted", "equal", "task_priority"
    
    # ========== 日志记录 ==========
    log_craft_metrics_freq: int = 100
    save_lambda_history: bool = True
    
    def __post_init__(self):
        """
        配置参数验证
        
        在初始化后自动调用，检查参数的合法性：
        - anchor_sample_ratio 必须在 [0, 1] 范围内
        - lambda 相关参数必须为正数
        - epsilon_start 必须 >= epsilon_end（退火方向）
        - projection_mode 必须是支持的模式之一
        """
        if not 0.0 <= self.anchor_sample_ratio <= 1.0:
            raise ValueError(f"anchor_sample_ratio must be in [0, 1], got {self.anchor_sample_ratio}")
        
        if self.initial_lambda < 0:
            raise ValueError(f"initial_lambda must be non-negative, got {self.initial_lambda}")
        
        if self.lambda_max <= 0:
            raise ValueError(f"lambda_max must be positive, got {self.lambda_max}")
        
        if self.epsilon_start < self.epsilon_end:
            raise ValueError(
                f"epsilon_start ({self.epsilon_start}) must be >= epsilon_end ({self.epsilon_end})"
            )
        
        if self.projection_mode not in ["weighted", "equal", "task_priority"]:
            raise ValueError(
                f"projection_mode must be one of ['weighted', 'equal', 'task_priority'], "
                f"got {self.projection_mode}"
            )

