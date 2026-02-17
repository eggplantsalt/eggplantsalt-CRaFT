# 完整实验操作指南

> 从零开始完成所有 CRaFT 实验的详细步骤

---

## 📋 目录

1. [实验环境准备](#实验环境准备)
2. [实验 1: Baseline 训练](#实验-1-baseline-训练)
3. [实验 2: 生成 Hidden Feature Cache](#实验-2-生成-hidden-feature-cache)
4. [实验 3: CRaFT 训练（Token-level）](#实验-3-craft-训练token-level)
5. [实验 4: CRaFT 训练（Hidden）](#实验-4-craft-训练hidden)
6. [实验 5: MCQ 评测](#实验-5-mcq-评测)
7. [实验 6: 对比分析](#实验-6-对比分析)
8. [故障排查](#故障排查)

---

## 实验环境准备

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核+ |
| RAM | 16GB | 32GB+ |
| GPU | GTX 1080 (8GB) | RTX 3090 (24GB) |
| 存储 | 50GB | 100GB+ SSD |

### 软件环境

```bash
# 1. 创建 Python 环境
conda create -n lerobot python=3.10
conda activate lerobot

# 2. 克隆仓库
git clone <your-repo-url>
cd lerobot

# 3. 安装依赖
pip install -e .

# 4. 验证安装
lerobot-info
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 数据集准备

```bash
# 创建数据目录
mkdir -p data/datasets
mkdir -p data/anchor_cache
mkdir -p data/anchor_hidden_cache
mkdir -p data/mcq_test

# 下载测试数据集（自动）
# 首次运行训练时会自动从 HuggingFace Hub 下载
```

### 目录结构

```
lerobot/
├── data/
│   ├── datasets/              # 数据集缓存
│   ├── anchor_cache/          # Token-level cache
│   ├── anchor_hidden_cache/   # Hidden feature cache
│   └── mcq_test/              # MCQ 测试数据
├── outputs/
│   ├── baseline/              # Baseline 训练输出
│   ├── craft_token/           # Token-level CRaFT 输出
│   ├── craft_hidden/          # Hidden CRaFT 输出
│   └── logs/                  # 训练日志
└── results/
    ├── metrics/               # 评测指标
    └── visualizations/        # 可视化结果
```

---

## 实验 1: Baseline 训练

### 目标

训练一个不使用 CRaFT 的基线模型，作为对比基准。

### 步骤

#### 1.1 准备配置文件

创建 `configs/baseline.yaml`:

```yaml
# Baseline 训练配置
policy:
  path: lerobot/pi0_fast

dataset:
  repo_id: lerobot/aloha_sim_insertion_human
  
training:
  steps: 10000
  batch_size: 8
  lr: 1e-4
  grad_clip_norm: 10
  save_checkpoint: true
  save_freq: 2000
  log_freq: 100

eval:
  freq: 0  # 不进行评估以节省时间

output_dir: outputs/baseline
```

#### 1.2 运行训练

```bash
# 方式 1: 使用配置文件
python -m lerobot.scripts.lerobot_train --config=configs/baseline.yaml

# 方式 2: 使用命令行参数
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/baseline \
    --steps=10000 \
    --batch_size=8 \
    --training.lr=1e-4 \
    --training.save_freq=2000 \
    --eval_freq=0
```

#### 1.3 监控训练

```bash
# 实时查看日志
tail -f outputs/baseline/train.log

# 或使用 TensorBoard（如果启用）
tensorboard --logdir=outputs/baseline/tensorboard
```

#### 1.4 预期输出

```
================================================================================
Training Configuration
================================================================================
Policy: pi0_fast
Dataset: lerobot/aloha_sim_insertion_human
Steps: 10000
Batch Size: 8
Learning Rate: 1e-4
================================================================================

Step 100/10000 | loss=2.345 | grdn=1.234 | lr=1.0e-04 | updt_s=0.523
Step 200/10000 | loss=2.123 | grdn=1.156 | lr=1.0e-04 | updt_s=0.498
Step 300/10000 | loss=1.987 | grdn=1.089 | lr=1.0e-04 | updt_s=0.512
...
Step 10000/10000 | loss=0.456 | grdn=0.234 | lr=1.0e-04 | updt_s=0.501

Training completed!
Checkpoint saved to: outputs/baseline/checkpoint-10000
```

#### 1.5 验证结果

```bash
# 检查 checkpoint 文件
ls -lh outputs/baseline/

# 预期文件:
# checkpoint-2000/
# checkpoint-4000/
# checkpoint-6000/
# checkpoint-8000/
# checkpoint-10000/
# train.log
# config.yaml
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `steps` | 10000 | 训练总步数 |
| `batch_size` | 8 | 批次大小（根据 GPU 显存调整） |
| `lr` | 1e-4 | 学习率 |
| `save_freq` | 2000 | 每 N 步保存一次 checkpoint |
| `log_freq` | 100 | 每 N 步记录一次日志 |

### 预期时间

- GPU (RTX 3090): ~2 小时
- GPU (GTX 1080): ~4 小时
- CPU: ~24 小时（不推荐）

---

## 实验 2: 生成 Hidden Feature Cache

### 目标

为 CRaFT 训练生成离线的 hidden feature cache，用于 retention loss 计算。

### 步骤

#### 2.1 准备配置

创建 `configs/build_cache.yaml`:

```yaml
dataset:
  repo_id: lerobot/aloha_sim_insertion_human

policy:
  path: lerobot/pi0_fast

output_dir: data/anchor_hidden_cache

num_samples: 1000
hidden_layer: -2
pooling: mean_image_tokens
batch_size: 8
num_workers: 4
```

#### 2.2 运行生成脚本

```bash
python -m lerobot.scripts.build_anchor_hidden_cache \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=data/anchor_hidden_cache \
    --num_samples=1000 \
    --hidden_layer=-2 \
    --pooling=mean_image_tokens \
    --batch_size=8 \
    --num_workers=4
```

#### 2.3 预期输出

```
================================================================================
Hidden Feature Cache 生成器
================================================================================
数据集: lerobot/aloha_sim_insertion_human
策略: lerobot/pi0_fast
输出目录: data/anchor_hidden_cache
样本数: 1000
Hidden Layer: -2
Pooling: mean_image_tokens
Batch Size: 8
================================================================================

加载数据集...
✓ 数据集加载完成: 1000 样本

加载策略模型...
✓ 策略加载完成: pi0_fast

开始生成 cache...
Processing: 100%|████████████████████| 125/125 [05:23<00:00, 2.59s/it]

保存 cache...
✓ Cache 保存完成

================================================================================
生成完成！
================================================================================
总样本数: 1000
输出目录: data/anchor_hidden_cache
文件列表:
  - shard_0.pt (45.2 MB)
  - metadata.json (2.1 KB)
总大小: 45.2 MB
================================================================================
```

#### 2.4 验证 cache

```bash
# 检查文件
ls -lh data/anchor_hidden_cache/

# 预期文件:
# shard_0.pt
# metadata.json

# 查看 metadata
cat data/anchor_hidden_cache/metadata.json
```

**metadata.json 示例**:
```json
{
  "num_samples": 1000,
  "hidden_layer": -2,
  "pooling": "mean_image_tokens",
  "feature_dim": 2048,
  "dtype": "float32",
  "created_at": "2026-02-17T10:30:00",
  "dataset": "lerobot/aloha_sim_insertion_human",
  "policy": "lerobot/pi0_fast"
}
```

### 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `hidden_layer` | -1, -2, ... | 提取哪一层的 hidden states |
| `pooling` | mean_image_tokens, mean_masked, last_token, cls_token | Pooling 策略 |
| `num_samples` | 100-10000 | 生成多少样本（越多越好，但更慢） |
| `batch_size` | 1-32 | 批次大小（根据 GPU 显存） |

### 预期时间

- 1000 样本: ~5 分钟 (RTX 3090)
- 5000 样本: ~25 分钟 (RTX 3090)
- 10000 样本: ~50 分钟 (RTX 3090)

---

## 实验 3: CRaFT 训练（Token-level）

### 目标

使用 token-level retention loss 进行 CRaFT 训练（旧版本，向后兼容）。

### 步骤

#### 3.1 生成 Token-level Cache

```bash
python -m lerobot.scripts.build_anchor_cache \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=data/anchor_cache \
    --num_samples=1000 \
    --batch_size=8
```

#### 3.2 运行 CRaFT 训练

```bash
python -m lerobot.scripts.lerobot_train_craft \
    --policy.path=lerobot/pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/craft_token \
    --steps=10000 \
    --batch_size=8 \
    --training.lr=1e-4 \
    --training.save_freq=2000 \
    --eval_freq=0 \
    craft.enabled=true \
    craft.retention_mode=token_ce \
    craft.anchor_cache_dir=data/anchor_cache \
    craft.anchor_batch_size=8 \
    craft.retention_freq=1 \
    craft.initial_lambda=1.0 \
    craft.lambda_lr=0.01 \
    craft.epsilon_start=1.0 \
    craft.epsilon_end=0.1 \
    craft.use_grad_projection=true
```

#### 3.3 预期输出

```
================================================================================
CRaFT 训练配置
================================================================================
CRaFT 启用: True
Retention Mode: token_ce
初始 λ: 1.0
λ 学习率: 0.01
ε 起始值: 1.0
ε 最终值: 0.1
梯度投影: True
================================================================================

✓ AnchorCache 加载成功: 1000 样本

Step 1/10000 | loss=2.345 | mode=token_ce | L_ret=1.234 | λ=1.012 | ε=1.000 | dot=-0.234 | cos=-0.156
Step 2/10000 | loss=2.123 | mode=token_ce | L_ret=1.189 | λ=1.019 | ε=0.9999 | conflict=✓ | dot=-0.189 | cos=-0.123
...
```

### 预期时间

- 10000 步: ~3 小时 (RTX 3090)

---

## 实验 4: CRaFT 训练（Hidden）

### 目标

使用 hidden state retention loss 进行 CRaFT 训练（推荐方式）。

### 步骤

#### 4.1 确认 Hidden Cache 已生成

```bash
# 检查 cache 是否存在
ls -lh data/anchor_hidden_cache/

# 如果不存在，运行实验 2
```

#### 4.2 运行 CRaFT 训练

```bash
python -m lerobot.scripts.lerobot_train_craft \
    --policy.path=lerobot/pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/craft_hidden \
    --steps=10000 \
    --batch_size=8 \
    --training.lr=1e-4 \
    --training.save_freq=2000 \
    --eval_freq=0 \
    craft.enabled=true \
    craft.retention_mode=hidden \
    craft.anchor_cache_dir=data/anchor_hidden_cache \
    craft.anchor_batch_size=8 \
    craft.retention_freq=1 \
    craft.initial_lambda=1.0 \
    craft.lambda_lr=0.01 \
    craft.epsilon_start=1.0 \
    craft.epsilon_end=0.1 \
    craft.use_grad_projection=true \
    craft.conflict_threshold=-0.1
```

#### 4.3 预期输出

```
================================================================================
CRaFT 训练配置
================================================================================
CRaFT 启用: True
Retention Mode: hidden
初始 λ: 1.0
λ 学习率: 0.01
ε 起始值: 1.0
ε 最终值: 0.1
梯度投影: True
冲突阈值: -0.1
================================================================================

✓ AnchorCache 加载成功: 1000 样本

Step 1/10000 | loss=2.345 | mode=hidden | L_ret=0.856 | λ=1.012 | ε=1.000 | dot=-0.234 | cos=-0.156
Step 2/10000 | loss=2.123 | mode=hidden | L_ret=0.789 | λ=1.019 | ε=0.9999 | conflict=✓ | dot=-0.189 | cos=-0.123
Step 3/10000 | loss=1.987 | mode=hidden | L_ret=0.723 | λ=1.024 | ε=0.9998 | dot=0.045 | cos=0.034
...
```

### 关键指标说明

| 指标 | 说明 |
|------|------|
| `loss` | 任务损失（L_task） |
| `L_ret` | 保留损失（L_retain） |
| `λ` | Lagrangian 乘子（动态调整） |
| `ε` | 保留约束阈值（线性退火） |
| `dot` | 梯度点积（负值表示冲突） |
| `cos` | 梯度余弦相似度 |
| `conflict=✓` | 检测到梯度冲突并进行投影 |

### 预期时间

- 10000 步: ~2.5 小时 (RTX 3090)

---

## 实验 5: MCQ 评测

### 目标

使用多选题 likelihood 评测模型性能。

### 步骤

#### 5.1 准备测试数据

创建 `data/mcq_test/test.jsonl`:

```jsonl
{"image_path": "data/mcq_test/images/scene1.jpg", "question": "What action should the robot take to complete the task?", "choices": ["pick up the red cup", "move to the left side", "stop and wait for instructions"], "answer_index": 0}
{"image_path": "data/mcq_test/images/scene2.jpg", "question": "What is the robot currently doing?", "choices": ["grasping an object", "navigating to a target", "observing the environment"], "answer_index": 2}
{"image_path": "data/mcq_test/images/scene3.jpg", "question": "Which object should the robot interact with?", "choices": ["the blue box", "the green bottle", "the yellow ball"], "answer_index": 1}
```

**注意**: 需要准备对应的图像文件。

#### 5.2 评测单个 Checkpoint

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline/checkpoint-10000 \
    --data_jsonl=data/mcq_test/test.jsonl \
    --max_samples=100 \
    --output_json=results/baseline_mcq.json
```

#### 5.3 对比两个 Checkpoint

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline/checkpoint-10000 \
    --checkpoint_path_b=outputs/craft_hidden/checkpoint-10000 \
    --data_jsonl=data/mcq_test/test.jsonl \
    --max_samples=100 \
    --output_json=results/comparison_mcq.json
```

#### 5.4 预期输出

```
================================================================================
对比结果
================================================================================
Checkpoint A: outputs/baseline/checkpoint-10000
  Accuracy: 75.00%
  Avg Margin: 1.8234
  Correct: 75/100

Checkpoint B: outputs/craft_hidden/checkpoint-10000
  Accuracy: 85.00%
  Avg Margin: 2.3456
  Correct: 85/100

差异:
  Accuracy: +10.00%
  Avg Margin: +0.5222
================================================================================
```

### 预期时间

- 100 样本: ~10 分钟 (RTX 3090)

---

## 实验 6: 对比分析

### 目标

系统对比 Baseline、Token-level CRaFT 和 Hidden CRaFT 的性能。

### 步骤

#### 6.1 收集训练指标

```bash
# 提取训练日志
python scripts/extract_metrics.py \
    --log_files outputs/*/train.log \
    --output results/training_metrics.csv
```

#### 6.2 生成对比图表

```python
# scripts/plot_comparison.py
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
baseline = pd.read_csv('outputs/baseline/metrics.csv')
craft_token = pd.read_csv('outputs/craft_token/metrics.csv')
craft_hidden = pd.read_csv('outputs/craft_hidden/metrics.csv')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(baseline['step'], baseline['loss'], label='Baseline')
plt.plot(craft_token['step'], craft_token['loss'], label='CRaFT (Token)')
plt.plot(craft_hidden['step'], craft_hidden['loss'], label='CRaFT (Hidden)')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Comparison')
plt.savefig('results/loss_comparison.png')
plt.show()
```

#### 6.3 生成对比表格

| 模型 | 训练时间 | 最终损失 | MCQ 准确率 | 存储空间 |
|------|----------|----------|------------|----------|
| Baseline | 2h | 0.456 | 75% | 2.3 GB |
| CRaFT (Token) | 3h | 0.512 | 82% | 2.8 GB |
| CRaFT (Hidden) | 2.5h | 0.489 | 85% | 2.5 GB |

### 分析要点

1. **训练效率**: Hidden 模式比 Token 模式快 ~17%
2. **存储效率**: Hidden cache 比 Token cache 小 ~95%
3. **性能提升**: CRaFT 相比 Baseline 提升 +10%
4. **稳定性**: Hidden 模式更稳定（margin 更大）

---

## 故障排查

### 问题 1: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**:
```bash
# 方案 1: 减小 batch_size
--batch_size=4

# 方案 2: 使用梯度累积
--batch_size=4 --gradient_accumulation_steps=2

# 方案 3: 使用混合精度
--use_amp=true
```

### 问题 2: 数据集下载失败

**症状**:
```
ConnectionError: Failed to download dataset
```

**解决方案**:
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定路径
--dataset.root=/path/to/local/dataset
```

### 问题 3: AnchorCache 加载失败

**症状**:
```
FileNotFoundError: AnchorCache directory not found
```

**解决方案**:
```bash
# 检查路径
ls -lh data/anchor_hidden_cache/

# 重新生成 cache
python -m lerobot.scripts.build_anchor_hidden_cache ...
```

### 问题 4: 训练不收敛

**症状**:
```
Step 1000/10000 | loss=2.345 (no decrease)
```

**解决方案**:
```bash
# 降低学习率
--training.lr=5e-5

# 增加 warmup
--training.warmup_steps=1000

# 检查数据质量
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('...'); print(ds.stats)"
```

### 问题 5: 梯度爆炸

**症状**:
```
Step 100/10000 | loss=nan | grdn=inf
```

**解决方案**:
```bash
# 启用梯度裁剪
--training.grad_clip_norm=10

# 降低学习率
--training.lr=1e-5

# 检查数据归一化
```

---

## 实验检查清单

### 实验 1: Baseline
- [ ] 训练完成（10000 步）
- [ ] Checkpoint 已保存
- [ ] 损失曲线正常下降
- [ ] 最终损失 < 1.0

### 实验 2: Hidden Cache
- [ ] Cache 生成完成
- [ ] metadata.json 存在
- [ ] 文件大小合理（~45MB/1000 样本）

### 实验 3: Token CRaFT
- [ ] 训练完成
- [ ] 日志包含 L_ret, λ, ε
- [ ] 梯度冲突检测正常

### 实验 4: Hidden CRaFT
- [ ] 训练完成
- [ ] mode=hidden 显示正确
- [ ] 性能优于 Baseline

### 实验 5: MCQ 评测
- [ ] 评测完成
- [ ] Accuracy 计算正确
- [ ] 对比结果合理

### 实验 6: 对比分析
- [ ] 所有指标收集完整
- [ ] 图表生成成功
- [ ] 结论清晰

---

## 附录

### A. 完整命令速查

```bash
# Baseline 训练
python -m lerobot.scripts.lerobot_train --policy.path=lerobot/pi0_fast --dataset.repo_id=lerobot/aloha_sim_insertion_human --output_dir=outputs/baseline --steps=10000 --batch_size=8

# 生成 Hidden Cache
python -m lerobot.scripts.build_anchor_hidden_cache --dataset.repo_id=lerobot/aloha_sim_insertion_human --policy.path=lerobot/pi0_fast --output_dir=data/anchor_hidden_cache --num_samples=1000

# CRaFT 训练
python -m lerobot.scripts.lerobot_train_craft --policy.path=lerobot/pi0_fast --dataset.repo_id=lerobot/aloha_sim_insertion_human --output_dir=outputs/craft_hidden --steps=10000 craft.enabled=true craft.retention_mode=hidden craft.anchor_cache_dir=data/anchor_hidden_cache

# MCQ 评测
python -m lerobot.scripts.eval_mcq_likelihood --checkpoint_path=outputs/baseline/checkpoint-10000 --checkpoint_path_b=outputs/craft_hidden/checkpoint-10000 --data_jsonl=data/mcq_test/test.jsonl
```

### B. 配置模板

完整配置模板见 `configs/` 目录。

### C. 预期结果

所有实验的预期结果和基准数据见 `docs/BENCHMARKS.md`。

---

**完成时间**: 所有实验约需 8-10 小时（使用 RTX 3090）

**下一步**: 查看 [结果分析指南](RESULTS_ANALYSIS.md) 了解如何解读实验结果。

