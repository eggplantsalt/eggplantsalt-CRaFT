# CRaFT 训练指南

## 概述

本指南介绍如何使用 CRaFT (Constrained Retention Fine-Tuning) 框架训练 LeRobot 策略模型。

## 前置条件

### 1. 准备 AnchorCache

在开始 CRaFT 训练之前，需要先生成 AnchorCache（锚点数据缓存）：

```bash
# 生成 AnchorCache
python -m lerobot.scripts.build_anchor_cache \
    --dataset_repo_id="lerobot/aloha_sim_insertion_human" \
    --policy_repo_id="lerobot/pi0_fast" \
    --output_dir="data/anchor_cache" \
    --num_samples=1000 \
    --batch_size=16 \
    --num_workers=4
```

详细说明请参考 `ANCHOR_CACHE_GUIDE.md`。

### 2. 验证环境

确保已安装所有依赖：

```bash
pip install -e .
```

## 快速开始

### Dry-Run 测试

首先运行 dry-run 验证训练流程（只运行 3 步）：

```bash
bash scripts/train_craft_dryrun.sh
```

预期输出：
- ✓ 数据集加载成功
- ✓ 策略模型加载成功
- ✓ 前向传播成功
- ✓ 反向传播成功
- ✓ 优化器更新成功

### 完整训练

#### 方式 1：使用脚本（推荐）

编辑 `scripts/train_craft.sh` 中的配置参数，然后运行：

```bash
bash scripts/train_craft.sh
```

#### 方式 2：直接命令行

```bash
python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id="lerobot/aloha_sim_insertion_human" \
    --policy.path="lerobot/pi0_fast" \
    --output_dir="outputs/craft_train" \
    --steps=10000 \
    --batch_size=8 \
    --eval_freq=1000 \
    --log_freq=100 \
    --save_freq=1000
```

## CRaFT 配置说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `False` | 是否启用 CRaFT（False 时退化为 baseline） |
| `anchor_cache_dir` | `""` | AnchorCache 目录路径 |
| `anchor_batch_size` | `16` | 锚点数据批次大小 |
| `retention_freq` | `5` | K-step 频率（每 K 步计算一次保留损失） |

### 原对偶优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_lambda` | `1.0` | Lagrangian 乘子 λ 的初始值 |
| `lambda_lr` | `0.01` | λ 的学习率 |
| `lambda_max` | `10.0` | λ 的最大值 |
| `epsilon_start` | `1.0` | 保留损失阈值 ε 的起始值 |
| `epsilon_end` | `0.1` | 保留损失阈值 ε 的最终值 |
| `epsilon_decay_steps` | `0` | ε 衰减步数（0 表示使用总训练步数） |

### 梯度手术参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_grad_projection` | `True` | 是否启用梯度投影 |
| `conflict_threshold` | `-0.1` | 冲突检测阈值（负余弦相似度） |
| `projection_mode` | `"weighted"` | 梯度合并模式（weighted/equal/task_priority） |

## 训练流程详解

### 1. 标准训练步骤（无 CRaFT）

```
for step in range(total_steps):
    batch = next(dataloader)
    loss = policy.forward(batch)
    loss.backward()
    optimizer.step()
```

### 2. CRaFT 训练步骤

```
for step in range(total_steps):
    # 任务损失
    task_batch = next(task_dataloader)
    task_loss = policy.forward(task_batch)
    task_loss.backward()
    task_grads = [p.grad.clone() for p in policy.parameters()]
    
    # 保留损失（每 K 步）
    if step % K == 0:
        anchor_batch = next(anchor_dataloader)
        retention_loss = policy.forward(anchor_batch)
        retention_loss.backward()
        retention_grads = [p.grad.clone() for p in policy.parameters()]
        
        # 梯度手术
        if conflict_detected(task_grads, retention_grads):
            task_grads = project(task_grads, retention_grads)
        
        # 合并梯度
        final_grads = task_grads + λ * retention_grads
        
        # 更新 λ
        λ = λ + λ_lr * (retention_loss - ε)
    else:
        final_grads = task_grads
    
    # 优化器更新
    set_grads(policy, final_grads)
    optimizer.step()
```

## 监控指标

### 训练日志

```
Step 100/10000 | loss=0.523 | grdn=1.234 | lr=3.0e-04 | L_ret=0.456 | λ=1.23 | ε=0.95 | conflict=✓ | cos=-0.15
```

指标说明：
- `loss`: 任务损失
- `grdn`: 梯度范数
- `lr`: 学习率
- `L_ret`: 保留损失
- `λ`: 当前 Lagrangian 乘子
- `ε`: 当前保留损失阈值
- `conflict`: 是否检测到梯度冲突
- `cos`: 梯度余弦相似度

### WandB 可视化

如果启用 WandB（`--wandb.enable=true`），可以查看：

- `train/loss`: 任务损失曲线
- `craft/retention_loss`: 保留损失曲线
- `craft/lambda`: λ 的演化轨迹
- `craft/epsilon`: ε 的衰减曲线
- `craft/grad_conflict`: 梯度冲突频率
- `craft/grad_dot`: 梯度点积（余弦相似度）

## 检查点管理

### 保存的文件

```
outputs/craft_train/
├── step_001000/
│   ├── model.safetensors       # 模型权重
│   ├── optimizer.pt            # 优化器状态
│   ├── scheduler.pt            # 学习率调度器状态
│   ├── train_config.json       # 训练配置
│   └── craft_state.pt          # CRaFT 状态（λ, ε, 历史）
├── step_002000/
│   └── ...
└── final/
    ├── model.safetensors
    ├── craft_state.pt
    └── lambda_history.csv      # λ 完整历史（CSV 格式）
```

### 恢复训练

```bash
python -m lerobot.scripts.lerobot_train_craft \
    --config_path="outputs/craft_train/step_001000/train_config.json" \
    --resume=true
```

## 对比实验

### Baseline vs CRaFT

运行对比实验：

```bash
# Baseline（无 CRaFT）
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="lerobot/aloha_sim_insertion_human" \
    --policy.path="lerobot/pi0_fast" \
    --output_dir="outputs/baseline" \
    --steps=10000

# CRaFT
bash scripts/train_craft.sh
```

对比指标：
- 新任务性能（task loss）
- 旧任务保留（retention loss）
- 训练稳定性（λ 收敛情况）
- 梯度冲突频率

## 常见问题

### Q1: AnchorCache 加载失败

**错误信息：**
```
⚠ AnchorCache 加载失败: [Errno 2] No such file or directory: 'data/anchor_cache'
```

**解决方案：**
先运行 `build_anchor_cache.py` 生成 AnchorCache。

### Q2: 训练时 λ 持续增长

**现象：** λ 从 1.0 增长到 10.0（上界）并保持不变。

**原因：** 保留损失持续违反约束（L_ret > ε）。

**解决方案：**
- 增大 `epsilon_start` 和 `epsilon_end`（放宽约束）
- 减小 `lambda_lr`（减缓 λ 增长速度）
- 增大 `anchor_batch_size`（提高保留损失估计准确性）
- 检查 AnchorCache 质量（是否包含代表性样本）

### Q3: 梯度冲突频繁

**现象：** 大部分步骤都检测到梯度冲突。

**原因：** 任务数据和锚点数据分布差异较大。

**解决方案：**
- 调整 `conflict_threshold`（例如从 -0.1 改为 -0.3，更严格的冲突判定）
- 减小 `retention_freq`（减少保留损失计算频率）
- 检查数据集是否合适（任务数据和锚点数据应有一定相关性）

### Q4: 训练速度慢

**现象：** CRaFT 训练比 baseline 慢 2 倍以上。

**原因：** 每步需要两次前向+反向传播。

**解决方案：**
- 增大 `retention_freq`（例如从 5 改为 10，减少保留损失计算频率）
- 减小 `anchor_batch_size`（减少锚点数据计算量）
- 禁用梯度投影（`use_grad_projection=false`，跳过冲突检测）

### Q5: 如何在服务器上训练

**本地操作：**
```bash
# 1. 本地编辑代码和配置
# 2. Git commit（不要 push）
git add .
git commit -m "feat: configure CRaFT training"
```

**服务器操作：**
```bash
# 1. SSH 到服务器
ssh user@server

# 2. 进入项目目录
cd /path/to/lerobot

# 3. Git pull（如果已 push）或手动同步代码
git pull

# 4. 生成 AnchorCache（如果还没有）
python -m lerobot.scripts.build_anchor_cache \
    --dataset_repo_id="lerobot/aloha_sim_insertion_human" \
    --policy_repo_id="lerobot/pi0_fast" \
    --output_dir="data/anchor_cache" \
    --num_samples=1000

# 5. 运行训练
bash scripts/train_craft.sh
```

## 高级用法

### 自定义 CRaFT 配置

在代码中创建自定义配置：

```python
from lerobot.craft import CraftConfig

craft_config = CraftConfig(
    enabled=True,
    anchor_cache_dir="data/anchor_cache",
    anchor_batch_size=16,
    retention_freq=5,
    initial_lambda=2.0,
    lambda_lr=0.02,
    lambda_max=15.0,
    epsilon_start=1.5,
    epsilon_end=0.05,
    epsilon_decay_steps=8000,
    use_grad_projection=True,
    conflict_threshold=-0.2,
    projection_mode="task_priority",
)

# 传递给训练函数
train_craft(cfg, craft_config=craft_config)
```

### 分析 λ 历史

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载 λ 历史
df = pd.read_csv("outputs/craft_train/final/lambda_history.csv")

# 绘制 λ 演化曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df["step"], df["lambda"])
plt.xlabel("Step")
plt.ylabel("λ")
plt.title("Lambda Evolution")

plt.subplot(1, 3, 2)
plt.plot(df["step"], df["epsilon"])
plt.xlabel("Step")
plt.ylabel("ε")
plt.title("Epsilon Decay")

plt.subplot(1, 3, 3)
plt.plot(df["step"], df["retention_loss"])
plt.axhline(y=df["epsilon"].iloc[-1], color='r', linestyle='--', label='Final ε')
plt.xlabel("Step")
plt.ylabel("Retention Loss")
plt.title("Retention Loss vs Constraint")
plt.legend()

plt.tight_layout()
plt.savefig("lambda_analysis.png")
```

## 参考资料

- [CRaFT 论文](https://arxiv.org/abs/xxxx.xxxxx)（待发布）
- [PCGrad 论文](https://arxiv.org/abs/2001.06782)
- [原对偶优化](https://web.stanford.edu/~boyd/cvxbook/)
- [LeRobot 文档](https://github.com/huggingface/lerobot)

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

