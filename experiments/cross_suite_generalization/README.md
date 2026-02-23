# CRaFT 跨 Suite 泛化能力验证实验

> **实验目标**: 验证 CRaFT 相比 Naive SFT 在单 Suite 微调后，能够显著缓解灾难性遗忘，在未见过的其他 Suites 上取得更高的成功率。

---

## 📋 实验设计

### 核心假设

**论文主张**: 相比于 Naive SFT，CRaFT 能够在保持目标任务（ID）性能的同时，显著缓解灾难性遗忘，在未见过的其他 Suites（OOD）上取得更高的成功率。

### 实验协议

| 项目 | 配置 |
|------|------|
| **Base Model** | `lerobot/pi0_fast` |
| **ID 训练集** | `libero_spatial` (目标微调域) |
| **OOD 测试集** | `libero_object`, `libero_goal`, `libero_10` |
| **Baseline** | Naive SFT (标准行为克隆，无 CRaFT 约束) |
| **实验组** | CRaFT (带表征漂移约束的微调) |

### 期待结论

✅ **ID 性能**: 在 `libero_spatial` 评测中，两者成功率相近  
✅ **OOD 泛化**: 在 `libero_object`、`libero_goal`、`libero_10` 评测中，**CRaFT 成功率 >> Naive SFT**

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 激活环境
conda activate lerobot

# 2. 进入实验目录
cd experiments/cross_suite_generalization

# 3. 创建输出目录
mkdir -p outputs/baseline_spatial
mkdir -p outputs/craft_spatial
mkdir -p outputs/anchor_cache
mkdir -p results
```

### 一键运行完整实验

```bash
# 运行完整实验流程（约 6-8 小时，取决于硬件）
bash run_full_experiment.sh
```

### 分步运行（推荐用于调试）

```bash
# 步骤 1: 训练 Baseline
bash scripts/01_train_baseline.sh

# 步骤 2: 构建 Anchor Cache
bash scripts/02_build_anchor_cache.sh

# 步骤 3: 训练 CRaFT
bash scripts/03_train_craft.sh

# 步骤 4: 跨 Suite 评测
bash scripts/04_eval_cross_suite.sh

# 步骤 5: 生成对比报告
python scripts/05_generate_report.py
```

---

## 📂 目录结构

```
cross_suite_generalization/
├── README.md                          # 本文档
├── run_full_experiment.sh             # 一键运行脚本
├── scripts/                           # 实验脚本
│   ├── 01_train_baseline.sh           # Baseline 训练
│   ├── 02_build_anchor_cache.sh       # 构建 Anchor Cache
│   ├── 03_train_craft.sh              # CRaFT 训练
│   ├── 04_eval_cross_suite.sh         # 跨 Suite 评测
│   └── 05_generate_report.py          # 生成对比报告
├── configs/                           # 配置文件
│   ├── baseline_spatial.yaml          # Baseline 训练配置
│   └── craft_spatial.yaml             # CRaFT 训练配置
├── outputs/                           # 训练输出（自动生成）
│   ├── baseline_spatial/              # Baseline checkpoints
│   ├── craft_spatial/                 # CRaFT checkpoints
│   └── anchor_cache/                  # Anchor cache 数据
└── results/                           # 评测结果（自动生成）
    ├── baseline_eval_results.json     # Baseline 评测结果
    ├── craft_eval_results.json        # CRaFT 评测结果
    └── comparison_report.md           # 对比报告
```

---

## 📊 实验流程详解

### 阶段一：训练 Baseline (Naive SFT)

**目标**: 在 `libero_spatial` 上进行标准微调，作为对照组。

**命令**:
```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0_fast \
    --env.type=libero \
    --env.task=libero_spatial \
    --dataset.repo_id=lerobot/libero_spatial_no_noops \
    --output_dir=experiments/cross_suite_generalization/outputs/baseline_spatial \
    --steps=10000 \
    --batch_size=32 \
    --eval_freq=2000 \
    --save_freq=2000 \
    --log_freq=100 \
    --seed=42
```

**预期输出**:
- Checkpoint 保存在 `outputs/baseline_spatial/checkpoints/010000/`
- 训练日志显示在 `libero_spatial` 上的损失持续下降

**检查清单**:
- [ ] 训练完成，无错误
- [ ] 最终 checkpoint 存在
- [ ] 训练损失收敛（< 0.5）

---

### 阶段二：构建 Anchor Cache

**目标**: 为 CRaFT 训练生成离线特征锚点缓存。

**命令**:
```bash
python -m lerobot.scripts.build_anchor_hidden_cache \
    --teacher_policy_path=lerobot/pi0_fast \
    --dataset_repo_id=lerobot/libero_spatial_no_noops \
    --output_dir=experiments/cross_suite_generalization/outputs/anchor_cache \
    --num_samples=1000 \
    --hidden_layer=-2 \
    --pooling=mean_image_tokens \
    --dtype=float16 \
    --shard_size=100 \
    --seed=42
```

**预期输出**:
- 生成 10 个 shard 文件: `shard_0000.pt` ~ `shard_0009.pt`
- 生成 `metadata.json` 配置文件
- 总大小约 500MB-1GB

**检查清单**:
- [ ] 所有 shard 文件生成
- [ ] metadata.json 包含正确的配置
- [ ] 可以用 `torch.load()` 加载任意 shard

---

### 阶段三：训练 CRaFT

**目标**: 使用 CRaFT 约束在 `libero_spatial` 上微调，保持对原始任务的记忆。

**命令**:
```bash
python -m lerobot.scripts.lerobot_train_craft \
    --policy.path=lerobot/pi0_fast \
    --env.type=libero \
    --env.task=libero_spatial \
    --dataset.repo_id=lerobot/libero_spatial_no_noops \
    --output_dir=experiments/cross_suite_generalization/outputs/craft_spatial \
    --steps=10000 \
    --batch_size=32 \
    --eval_freq=2000 \
    --save_freq=2000 \
    --log_freq=100 \
    --seed=42 \
    craft.enabled=true \
    craft.anchor_cache_dir=experiments/cross_suite_generalization/outputs/anchor_cache \
    craft.retention_freq=5 \
    craft.initial_lambda=1.0 \
    craft.lambda_lr=0.001 \
    craft.epsilon_start=1.0 \
    craft.epsilon_end=0.05 \
    craft.use_grad_projection=true \
    craft.conflict_threshold=-0.1
```

**预期输出**:
- Checkpoint 保存在 `outputs/craft_spatial/checkpoints/010000/`
- 训练日志显示 `task_loss` 和 `retention_loss` 都在下降
- `lambda` 值逐渐调整（通常在 0.5-2.0 之间）

**检查清单**:
- [ ] 训练完成，无错误
- [ ] 最终 checkpoint 存在
- [ ] 训练日志包含 `retention_loss` 和 `lambda` 指标
- [ ] `retention_loss` 保持在 `epsilon` 阈值附近

---

### 阶段四：跨 Suite 评测

**目标**: 在 4 个不同的 Suites 上评测 Baseline 和 CRaFT，验证泛化能力。

#### 4.1 评测 Baseline

```bash
# ID: libero_spatial
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/baseline_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/baseline_spatial_on_spatial

# OOD: libero_object
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/baseline_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_object \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/baseline_spatial_on_object

# OOD: libero_goal
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/baseline_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_goal \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/baseline_spatial_on_goal

# OOD: libero_10
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/baseline_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/baseline_spatial_on_10
```

#### 4.2 评测 CRaFT

```bash
# ID: libero_spatial
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/craft_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/craft_spatial_on_spatial

# OOD: libero_object
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/craft_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_object \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/craft_spatial_on_object

# OOD: libero_goal
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/craft_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_goal \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/craft_spatial_on_goal

# OOD: libero_10
python -m lerobot.scripts.lerobot_eval \
    --policy.path=experiments/cross_suite_generalization/outputs/craft_spatial/checkpoints/010000/pretrained_model \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=experiments/cross_suite_generalization/results/craft_spatial_on_10
```

**预期输出**:
- 每个评测生成 `eval_info.json` 包含成功率等指标
- 视频保存在各自的 `videos/` 目录

**检查清单**:
- [ ] 所有 8 个评测任务完成
- [ ] 每个结果目录包含 `eval_info.json`
- [ ] 成功率数据可读取

---

### 阶段五：生成对比报告

**目标**: 汇总所有评测结果，生成对比表格和可视化图表。

**命令**:
```bash
python experiments/cross_suite_generalization/scripts/05_generate_report.py
```

**预期输出**:
```
results/comparison_report.md
results/comparison_table.csv
results/success_rate_comparison.png
```

**示例报告**:

| Suite | Baseline Success Rate | CRaFT Success Rate | Improvement |
|-------|----------------------|-------------------|-------------|
| libero_spatial (ID) | 85.2% | 84.8% | -0.4% |
| libero_object (OOD) | 12.4% | 45.6% | **+33.2%** |
| libero_goal (OOD) | 8.7% | 38.9% | **+30.2%** |
| libero_10 (OOD) | 5.3% | 28.4% | **+23.1%** |

**关键结论**:
✅ ID 性能保持：CRaFT 在 `libero_spatial` 上的性能与 Baseline 相当  
✅ OOD 泛化提升：CRaFT 在所有 OOD Suites 上的成功率显著高于 Baseline  
✅ 验证论文主张：CRaFT 有效缓解灾难性遗忘

---

## ⚙️ 配置说明

### Baseline 配置 (`configs/baseline_spatial.yaml`)

标准的 Pi0-fast 训练配置，无任何 CRaFT 约束。

### CRaFT 配置 (`configs/craft_spatial.yaml`)

关键超参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `craft.enabled` | `true` | 启用 CRaFT |
| `craft.anchor_cache_dir` | `outputs/anchor_cache` | Anchor cache 路径 |
| `craft.retention_freq` | `5` | 每 5 步计算一次保留损失 |
| `craft.initial_lambda` | `1.0` | 初始 Lagrangian 乘子 |
| `craft.lambda_lr` | `0.001` | λ 更新学习率 |
| `craft.epsilon_start` | `1.0` | 初始保留约束阈值 |
| `craft.epsilon_end` | `0.05` | 最终保留约束阈值 |
| `craft.use_grad_projection` | `true` | 启用梯度投影 |
| `craft.conflict_threshold` | `-0.1` | 梯度冲突检测阈值 |

---

## 🔧 故障排查

### 问题 1: LIBERO 环境未安装

**错误信息**:
```
ModuleNotFoundError: No module named 'libero'
```

**解决方案**:
```bash
pip install libero-robotics
```

### 问题 2: 数据集下载失败

**错误信息**:
```
HfHubHTTPError: 404 Client Error
```

**解决方案**:
```bash
# 手动下载数据集
huggingface-cli download lerobot/libero_spatial_no_noops --repo-type dataset
```

### 问题 3: GPU 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 减小批次大小
--batch_size=16  # 从 32 降到 16
craft.anchor_batch_size=8  # 从 16 降到 8
```

### 问题 4: Anchor Cache 加载失败

**错误信息**:
```
FileNotFoundError: metadata.json not found
```

**解决方案**:
```bash
# 重新生成 Anchor Cache
bash scripts/02_build_anchor_cache.sh
```

---

## 📈 预期实验时间

| 阶段 | 时间（RTX 3090） | 时间（V100） |
|------|-----------------|-------------|
| Baseline 训练 | 1.5 小时 | 2.5 小时 |
| Anchor Cache 生成 | 15 分钟 | 25 分钟 |
| CRaFT 训练 | 2 小时 | 3.5 小时 |
| 跨 Suite 评测 (8 个任务) | 2 小时 | 3 小时 |
| 报告生成 | 5 分钟 | 5 分钟 |
| **总计** | **约 6 小时** | **约 9.5 小时** |

---

## 📚 参考文档

- [CRaFT 训练指南](../../docs/craft/CRAFT_TRAINING_GUIDE.md)
- [Hidden Feature Cache 说明](../../docs/HIDDEN_FEATURE_CACHE_SUMMARY.md)
- [LIBERO 环境文档](../../docs/source/libero.mdx)
- [实验操作指南](../../docs/EXPERIMENT_GUIDE.md)

---

## 📝 引用

如果本实验对你的研究有帮助，请引用：

```bibtex
@article{craft2025,
  title={Constrained Retention Fine-Tuning for Continual Robot Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 🤝 贡献

如有问题或改进建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2025-02-23

