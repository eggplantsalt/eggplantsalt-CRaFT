# CRaFT 文件组织说明

## 📂 CRaFT 专属文件清单

### 核心模块（src/lerobot/craft/）
```
src/lerobot/craft/
├── README.md                    # CRaFT 模块总览和使用指南
├── __init__.py                  # 包初始化（导出 CraftConfig）
├── craft_config.py              # 配置类：所有超参数定义
├── grad_surgery.py              # 梯度手术：投影和合并算法
├── primal_dual.py               # 原对偶优化：λ 更新和 ε 调度
├── retention_loss.py            # 保留损失：在锚点数据上计算损失
└── anchor_cache.py              # 锚点缓存：数据集加载和采样
```

### 训练脚本（src/lerobot/scripts/）
```
src/lerobot/scripts/
├── lerobot_train.py             # ✅ Baseline 训练（不修改）
└── lerobot_train_craft.py       # 🆕 CRaFT 训练入口
```

### 测试文件（tests/）
```
tests/
└── test_grad_surgery_math.py    # 🆕 梯度手术单元测试
```

### 文档（根目录）
```
根目录/
├── progress.txt                 # 🆕 项目进度跟踪
└── tests.json                   # 🆕 测试计划定义
```

---

## 🎯 文件功能速查

### 1. craft_config.py（配置中心）
**作用**: 定义所有 CRaFT 训练超参数

**关键配置项**:
- `anchor_dataset_path`: 锚点数据集路径
- `initial_lambda`: 初始 λ 值（保留损失权重）
- `epsilon_start/end`: 保留损失阈值的起始/结束值
- `use_grad_projection`: 是否启用梯度投影
- `conflict_threshold`: 冲突检测阈值
- `projection_mode`: 梯度合并模式

**中文注释要点**:
- 每个参数都有详细的中文说明
- 包含典型值范围和推荐设置
- 解释了参数之间的关系

---

### 2. grad_surgery.py（梯度手术核心）
**作用**: 解决任务梯度和保留梯度的冲突

**核心函数**:
```python
compute_dot(grad1, grad2)
# 计算梯度点积，判断是否冲突
# 正值=协同，负值=冲突

project_if_conflict(grad_task, grad_retain, threshold)
# 当冲突时投影任务梯度到保留梯度的法平面
# 基于 PCGrad 算法

merge_grads(grad_task, grad_retain, lambda_weight, mode)
# 合并投影后的梯度
# 支持 weighted/equal/task_priority 三种模式
```

**中文注释要点**:
- 详细的数学公式和推导
- 直观的几何解释
- 完整的实现提示和示例代码
- 参考文献（PCGrad, CAGrad）

---

### 3. primal_dual.py（原对偶优化）
**作用**: 动态调整保留损失权重 λ

**核心函数**:
```python
epsilon_schedule(step, epsilon_start, epsilon_end, decay_steps, schedule_type)
# 计算当前步的保留损失阈值 ε(t)
# 支持 linear/cosine/exponential 三种退火策略

update_lambda(current_lambda, retention_loss, epsilon, lambda_lr, lambda_max)
# 更新 Lagrangian 乘子 λ
# 规则: λ ← clip(λ + λ_lr * (L_retain - ε), 0, λ_max)
```

**中文注释要点**:
- 优化问题的数学表述（原问题和对偶问题）
- 直观理解（违反约束→增大 λ，满足约束→减小 λ）
- 详细的更新规则和裁剪逻辑
- 三种退火策略的对比

---

### 4. retention_loss.py（保留损失）
**作用**: 在锚点数据上计算损失，衡量记忆程度

**核心函数**:
```python
compute_retention_loss(policy, anchor_batch, reduction)
# 在锚点数据上调用 policy.forward()
# 复用训练损失的计算逻辑
```

**中文注释要点**:
- 保留损失的定义和意义
- 与任务损失的关系（相同函数，不同数据）
- 实现非常简单（封装 policy.forward）
- 调试技巧和常见问题

---

### 5. anchor_cache.py（锚点数据管理）
**作用**: 加载和采样锚点/旧任务数据

**核心类/函数**:
```python
class AnchorCacheDataset(Dataset)
# PyTorch Dataset 包装器
# 封装锚点数据的加载和访问

create_anchor_dataloader(dataset_path, batch_size, ...)
# 一站式创建 DataLoader
# 推荐使用此函数
```

**中文注释要点**:
- 锚点数据的概念和来源
- 数据格式要求（与训练数据一致）
- 性能优化建议（批次大小、工作进程数）
- 与任务 DataLoader 的对比

---

### 6. lerobot_train_craft.py（训练入口）
**作用**: CRaFT 训练的主脚本，扩展自 baseline

**核心函数**:
```python
train_craft(cfg)
# 主训练循环
# 加载任务数据 + 锚点数据

update_policy_craft(...)
# 单步训练更新
# 双向后传播 + 梯度手术 + λ 更新
```

**训练流程**:
```
for step in range(steps):
    1. 前向传播（任务数据）→ L_task
    2. 反向传播 → ∇L_task
    3. 前向传播（锚点数据）→ L_retain
    4. 反向传播 → ∇L_retain
    5. 梯度手术（投影 + 合并）
    6. 优化器更新
    7. 更新 λ（原对偶）
```

**当前状态**: Dry-run 模式（只做 1 个 batch 的 forward）

---

### 7. test_grad_surgery_math.py（单元测试）
**作用**: 验证梯度手术的数学正确性

**测试用例**:
- `test_compute_dot_positive`: 对齐梯度的点积
- `test_compute_dot_negative`: 冲突梯度的点积
- `test_project_if_conflict_no_conflict`: 无冲突时不投影
- `test_project_if_conflict_with_conflict`: 冲突时投影
- `test_merge_grads_weighted`: 加权合并
- `test_merge_grads_equal`: 平均合并
- `test_gradient_surgery_end_to_end`: 端到端测试

**中文注释要点**:
- 每个测试的场景和预期结果
- 详细的实现示例代码
- 数学验证步骤

---

## 📋 文件依赖关系

```
lerobot_train_craft.py
    ├── craft_config.py (配置)
    ├── anchor_cache.py (数据加载)
    │   └── AnchorCacheDataset
    │   └── create_anchor_dataloader
    ├── retention_loss.py (损失计算)
    │   └── compute_retention_loss
    ├── grad_surgery.py (梯度手术)
    │   ├── compute_dot
    │   ├── project_if_conflict
    │   └── merge_grads
    └── primal_dual.py (优化)
        ├── epsilon_schedule
        └── update_lambda
```

---

## 🔍 快速定位指南

### 想修改超参数？
→ `craft_config.py` 的 `CraftConfig` 类

### 想理解梯度投影算法？
→ `grad_surgery.py` 的 `project_if_conflict` 函数

### 想调整 λ 更新策略？
→ `primal_dual.py` 的 `update_lambda` 函数

### 想修改训练循环？
→ `lerobot_train_craft.py` 的 `update_policy_craft` 函数

### 想添加新的测试？
→ `test_grad_surgery_math.py`

---

## ✅ 中文注释覆盖情况

| 文件 | 模块级注释 | 类/函数注释 | 参数说明 | 示例代码 | 实现提示 |
|------|-----------|------------|---------|---------|---------|
| craft_config.py | ✅ | ✅ | ✅ | ✅ | ✅ |
| grad_surgery.py | ✅ | ✅ | ✅ | ✅ | ✅ |
| primal_dual.py | ✅ | ✅ | ✅ | ✅ | ✅ |
| retention_loss.py | ✅ | ✅ | ✅ | ✅ | ✅ |
| anchor_cache.py | ✅ | ✅ | ✅ | ✅ | ✅ |
| test_grad_surgery_math.py | ✅ | ✅ | ✅ | ✅ | ✅ |

**注释特点**:
- 📖 模块级：功能概述、核心思想、使用示例
- 🔧 函数级：详细说明、参数解释、返回值、实现提示
- 💡 示例代码：完整可运行的代码片段
- 🎯 实现提示：伪代码和关键步骤
- 📚 参考文献：相关论文和算法

---

## 🚀 下一步工作

### 阶段 2: 核心算法实现
1. 实现 `grad_surgery.py` 的三个函数
2. 实现 `primal_dual.py` 的两个函数
3. 实现 `retention_loss.py`（最简单）
4. 编写对应的单元测试

### 阶段 3: 数据管道
1. 实现 `anchor_cache.py` 的数据加载
2. 在 `lerobot_train_craft.py` 中集成锚点数据

### 阶段 4: 训练循环
1. 在 `update_policy_craft` 中实现双向后传播
2. 集成梯度手术和 λ 更新
3. 取消注释 TODO 部分

---

## 📞 使用帮助

### 如何运行 Dry-run？
```bash
python src/lerobot/scripts/lerobot_train_craft.py \
    --policy.type=pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --batch_size=8 \
    --steps=1 \
    --output_dir=outputs/craft_dryrun
```

### 如何查看某个模块的文档？
```bash
# 在 Python 中
from lerobot.craft import CraftConfig
help(CraftConfig)

from lerobot.craft.grad_surgery import compute_dot
help(compute_dot)
```

### 如何运行测试？
```bash
# 运行所有测试（当前会跳过）
pytest tests/test_grad_surgery_math.py -v

# 实现后运行
pytest tests/test_grad_surgery_math.py -v -m "not skip"
```

---

## 📝 Git 提交记录

```bash
# 查看 CRaFT 相关提交
git log --oneline --grep="craft\|CRaFT" --all

# 查看文件修改历史
git log --follow src/lerobot/craft/grad_surgery.py
```

---

**最后更新**: 2026-02-15  
**维护者**: CRaFT 开发团队  
**状态**: 脚手架完成，核心算法待实现

