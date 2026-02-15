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
CRaFT (Constrained Retention Fine-Tuning) 训练框架
=================================================

【模块说明】
本包实现了 CRaFT 持续学习训练方法，通过保留约束来防止灾难性遗忘。

【核心功能】
1. 梯度手术 (Gradient Surgery): 解决任务梯度和保留梯度的冲突
2. 原对偶优化 (Primal-Dual): 动态调整保留损失权重 λ
3. 锚点损失计算 (Anchor Loss): 在旧任务数据上评估性能保持

【使用示例】
```python
from lerobot.craft import CraftConfig

# 创建 CRaFT 配置
craft_cfg = CraftConfig(
    anchor_dataset_path="path/to/anchor/data",
    initial_lambda=1.0,
    epsilon_start=1.0,
    epsilon_end=0.1,
    use_grad_projection=True
)
```

【文件结构】
- craft_config.py: 配置类定义
- grad_surgery.py: 梯度投影和合并算法
- primal_dual.py: λ 更新和 ε 调度
- retention_loss.py: 保留损失计算
- anchor_cache.py: 锚点数据集加载
"""

from lerobot.craft.craft_config import CraftConfig

__all__ = ["CraftConfig"]

