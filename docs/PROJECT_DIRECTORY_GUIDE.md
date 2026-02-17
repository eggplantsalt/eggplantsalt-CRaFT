# 项目目录完整指引

> LeRobot + CRaFT 项目的完整目录结构和文件说明

---

## 📂 项目根目录

```
lerobot/
├── README.md                    # LeRobot 原始 README
├── README_CRAFT.md              # CRaFT 项目 README（主入口）
├── CONTRIBUTING.md              # 贡献指南
├── LICENSE                      # Apache 2.0 许可证
├── pyproject.toml               # 项目配置和依赖
├── setup.py                     # 安装脚本
│
├── src/                         # 源代码
├── docs/                        # 文档
├── tests/                       # 测试
├── scripts/                     # 训练脚本
├── configs/                     # 配置文件
├── data/                        # 数据目录
├── outputs/                     # 训练输出
└── results/                     # 实验结果
```

---

## 📚 文档目录 (docs/)

### 核心文档

```
docs/
├── README.md                    # 文档导航（从这里开始）
├── QUICKSTART.md                # 快速开始指南（5分钟上手）
├── EXPERIMENT_GUIDE.md          # 完整实验操作指南（详细步骤）
├── API_REFERENCE.md             # API 参考文档（开发者必读）
├── OVERVIEW.md                  # 项目架构文档（技术详解）
├── TROUBLESHOOTING.md           # 故障排查指南（问题解决）
│
├── HIDDEN_FEATURE_CACHE_SUMMARY.md  # Hidden Cache 文档
├── MCQ_LIKELIHOOD_EVAL.md           # MCQ 评测工具文档
├── CONTEXT.md                       # 用户自定义上下文
│
└── craft/                       # CRaFT 专题文档
    └── CRAFT_TRAINING_GUIDE.md  # CRaFT 训练详解
```

### 文档阅读顺序

**新手用户**:
1. `README_CRAFT.md` (根目录)
2. `docs/QUICKSTART.md`
3. `docs/TROUBLESHOOTING.md`

**研究人员**:
1. `README_CRAFT.md`
2. `docs/QUICKSTART.md`
3. `docs/EXPERIMENT_GUIDE.md`
4. `docs/craft/CRAFT_TRAINING_GUIDE.md`

**开发者**:
1. `docs/OVERVIEW.md`
2. `docs/API_REFERENCE.md`
3. `src/lerobot/craft/README.md`
4. `CONTRIBUTING.md`

---

## 💻 源代码目录 (src/lerobot/)

### CRaFT 核心模块

```
src/lerobot/craft/
├── __init__.py                  # 包初始化，导出 CraftConfig
├── README.md                    # CRaFT 模块说明
├── craft_config.py              # CRaFT 配置类
├── grad_surgery.py              # 梯度手术（投影、合并）
├── primal_dual.py               # 原对偶优化（λ 更新、ε 调度）
├── retention_loss.py            # 保留损失计算
└── anchor_cache.py              # 锚点数据加载器
```

**关键文件说明**:

| 文件 | 功能 | 核心函数 |
|------|------|----------|
| `craft_config.py` | 配置管理 | `CraftConfig` 类 |
| `grad_surgery.py` | 梯度手术 | `compute_dot()`, `project_if_conflict()`, `merge_grads()` |
| `primal_dual.py` | 原对偶优化 | `epsilon_schedule()`, `update_lambda()` |
| `retention_loss.py` | 保留损失 | `compute_hidden_retention_loss()` |
| `anchor_cache.py` | 数据加载 | `AnchorCacheDataset` 类 |

### 训练脚本

```
src/lerobot/scripts/
├── lerobot_train.py             # Baseline 训练脚本
├── lerobot_train_craft.py       # CRaFT 训练脚本（核心）
├── build_anchor_cache.py        # Token-level cache 生成
├── build_anchor_hidden_cache.py # Hidden feature cache 生成（推荐）
├── eval_mcq_likelihood.py       # MCQ 评测脚本
└── lerobot_eval.py              # 标准评估脚本
```

**脚本使用说明**:

| 脚本 | 用途 | 示例命令 |
|------|------|----------|
| `lerobot_train.py` | Baseline 训练 | `python -m lerobot.scripts.lerobot_train ...` |
| `lerobot_train_craft.py` | CRaFT 训练 | `python -m lerobot.scripts.lerobot_train_craft ...` |
| `build_anchor_hidden_cache.py` | 生成 cache | `python -m lerobot.scripts.build_anchor_hidden_cache ...` |
| `eval_mcq_likelihood.py` | MCQ 评测 | `python -m lerobot.scripts.eval_mcq_likelihood ...` |

### 策略模型

```
src/lerobot/policies/
├── pretrained.py                # 策略基类
├── pi0_fast/                    # Pi0Fast 模型
│   ├── modeling_pi0_fast.py     # 模型实现
│   ├── configuration_pi0_fast.py # 配置
│   └── processor_pi0_fast.py    # 数据处理
├── act/                         # ACT 模型
├── diffusion/                   # Diffusion 模型
└── ...                          # 其他模型
```

---

## 🧪 测试目录 (tests/)

```
tests/
├── test_grad_surgery_math.py    # 梯度手术单元测试
├── test_primal_dual.py          # 原对偶优化测试
├── test_hidden_retention_loss_math.py  # 保留损失测试
├── test_hidden_cache_format.py  # Hidden cache 格式测试
├── test_mcq_likelihood_smoke.py # MCQ 评测 smoke test
└── verify_hidden_cache.py       # Hidden cache 验证脚本
```

**测试运行**:
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_grad_surgery_math.py -v

# 运行 smoke test
python tests/test_mcq_likelihood_smoke.py
```

---

## 📜 脚本目录 (scripts/)

```
scripts/
├── train_craft.sh               # CRaFT 训练脚本（完整）
└── train_craft_hidden_dryrun.sh # Dry-run 测试脚本（3步）
```

**使用方法**:
```bash
# 完整训练
bash scripts/train_craft.sh

# Dry-run 测试
bash scripts/train_craft_hidden_dryrun.sh
```

---

## ⚙️ 配置目录 (configs/)

```
configs/
├── baseline.yaml                # Baseline 训练配置
├── craft_token.yaml             # Token-level CRaFT 配置
├── craft_hidden.yaml            # Hidden CRaFT 配置（推荐）
└── build_cache.yaml             # Cache 生成配置
```

**配置文件示例**:
```yaml
# configs/craft_hidden.yaml
policy:
  path: lerobot/pi0_fast

dataset:
  repo_id: lerobot/aloha_sim_insertion_human

training:
  steps: 10000
  batch_size: 8
  lr: 1e-4

craft:
  enabled: true
  retention_mode: hidden
  anchor_cache_dir: data/anchor_hidden_cache
  initial_lambda: 1.0
  epsilon_start: 1.0
  epsilon_end: 0.1
```

---

## 💾 数据目录 (data/)

```
data/
├── datasets/                    # 数据集缓存（自动下载）
│   └── lerobot/
│       └── aloha_sim_insertion_human/
│
├── anchor_cache/                # Token-level cache
│   ├── shard_0.pt
│   └── metadata.json
│
├── anchor_hidden_cache/         # Hidden feature cache（推荐）
│   ├── shard_0.pt
│   └── metadata.json
│
└── mcq_test/                    # MCQ 测试数据
    ├── test.jsonl
    └── images/
```

**数据目录说明**:
- `datasets/`: HuggingFace Hub 自动下载的数据集
- `anchor_cache/`: Token-level cache（旧版本）
- `anchor_hidden_cache/`: Hidden feature cache（推荐）
- `mcq_test/`: MCQ 评测数据

---

## 📊 输出目录 (outputs/)

```
outputs/
├── baseline/                    # Baseline 训练输出
│   ├── checkpoint-2000/
│   ├── checkpoint-4000/
│   ├── ...
│   ├── checkpoint-10000/
│   ├── train.log
│   └── config.yaml
│
├── craft_token/                 # Token-level CRaFT 输出
│   └── ...
│
├── craft_hidden/                # Hidden CRaFT 输出（推荐）
│   ├── checkpoint-2000/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── craft_state.pt       # CRaFT 状态（λ, ε）
│   ├── ...
│   ├── final/
│   │   ├── model.safetensors
│   │   ├── craft_state.pt
│   │   └── lambda_history.csv   # λ 历史记录
│   └── train.log
│
└── logs/                        # 训练日志
```

**Checkpoint 内容**:
- `model.safetensors`: 模型权重
- `config.json`: 模型配置
- `craft_state.pt`: CRaFT 状态（λ, ε, step）
- `lambda_history.csv`: λ 完整历史

---

## 📈 结果目录 (results/)

```
results/
├── metrics/                     # 评测指标
│   ├── baseline_mcq.json
│   ├── craft_hidden_mcq.json
│   └── comparison_mcq.json
│
├── visualizations/              # 可视化结果
│   ├── loss_comparison.png
│   ├── lambda_history.png
│   └── accuracy_comparison.png
│
└── analysis/                    # 分析报告
    └── experiment_report.md
```

---

## 🔍 关键文件快速定位

### 我想...

**运行第一个实验**:
1. 阅读 `README_CRAFT.md`
2. 按照 `docs/QUICKSTART.md` 操作
3. 运行 `bash scripts/train_craft_hidden_dryrun.sh`

**理解 CRaFT 原理**:
1. 阅读 `docs/craft/CRAFT_TRAINING_GUIDE.md`
2. 查看 `src/lerobot/craft/README.md`
3. 阅读 `docs/API_REFERENCE.md`

**修改训练参数**:
1. 编辑 `configs/craft_hidden.yaml`
2. 或在命令行传递参数
3. 参考 `docs/EXPERIMENT_GUIDE.md`

**生成 Hidden Cache**:
1. 运行 `python -m lerobot.scripts.build_anchor_hidden_cache ...`
2. 参考 `docs/HIDDEN_FEATURE_CACHE_SUMMARY.md`
3. 查看 `docs/EXPERIMENT_GUIDE.md#实验-2`

**评测模型性能**:
1. 准备 MCQ 数据（JSONL 格式）
2. 运行 `python -m lerobot.scripts.eval_mcq_likelihood ...`
3. 参考 `docs/MCQ_LIKELIHOOD_EVAL.md`

**解决问题**:
1. 查看 `docs/TROUBLESHOOTING.md`
2. 检查日志 `outputs/*/train.log`
3. 在 GitHub 提交 Issue

**贡献代码**:
1. 阅读 `CONTRIBUTING.md`
2. 查看 `docs/API_REFERENCE.md`
3. 运行测试 `pytest tests/ -v`

---

## 📋 文件类型说明

### Python 文件 (.py)

| 类型 | 位置 | 说明 |
|------|------|------|
| 模块 | `src/lerobot/craft/*.py` | CRaFT 核心算法 |
| 脚本 | `src/lerobot/scripts/*.py` | 训练和评测脚本 |
| 测试 | `tests/*.py` | 单元测试和集成测试 |

### 配置文件

| 类型 | 位置 | 说明 |
|------|------|------|
| YAML | `configs/*.yaml` | 训练配置 |
| JSON | `data/*/metadata.json` | Cache 元数据 |
| TOML | `pyproject.toml` | 项目配置 |

### 文档文件 (.md)

| 类型 | 位置 | 说明 |
|------|------|------|
| 用户文档 | `docs/*.md` | 使用指南 |
| 技术文档 | `docs/craft/*.md` | 技术详解 |
| 代码文档 | `src/*/README.md` | 模块说明 |

### 数据文件

| 类型 | 位置 | 说明 |
|------|------|------|
| Cache | `data/*_cache/*.pt` | PyTorch 张量 |
| 数据集 | `data/datasets/` | Parquet + MP4 |
| 测试数据 | `data/mcq_test/*.jsonl` | JSONL 格式 |

### 模型文件

| 类型 | 位置 | 说明 |
|------|------|------|
| 权重 | `outputs/*/checkpoint-*/model.safetensors` | SafeTensors 格式 |
| 配置 | `outputs/*/checkpoint-*/config.json` | JSON 格式 |
| 状态 | `outputs/*/checkpoint-*/craft_state.pt` | PyTorch 格式 |

---

## 🎯 常用路径

```bash
# 训练脚本
src/lerobot/scripts/lerobot_train_craft.py

# CRaFT 配置
src/lerobot/craft/craft_config.py

# 梯度手术
src/lerobot/craft/grad_surgery.py

# 保留损失
src/lerobot/craft/retention_loss.py

# 快速开始
docs/QUICKSTART.md

# 实验指南
docs/EXPERIMENT_GUIDE.md

# API 参考
docs/API_REFERENCE.md

# 故障排查
docs/TROUBLESHOOTING.md

# 训练脚本
scripts/train_craft.sh

# Dry-run 测试
scripts/train_craft_hidden_dryrun.sh

# 配置模板
configs/craft_hidden.yaml

# 测试
tests/test_grad_surgery_math.py
```

---

## 📊 目录大小估算

| 目录 | 预期大小 | 说明 |
|------|----------|------|
| `src/` | ~50 MB | 源代码 |
| `docs/` | ~5 MB | 文档 |
| `tests/` | ~2 MB | 测试 |
| `data/datasets/` | ~5 GB | 数据集（取决于数据集大小） |
| `data/anchor_hidden_cache/` | ~50 MB | Hidden cache (1000 样本) |
| `outputs/baseline/` | ~2 GB | Baseline checkpoint |
| `outputs/craft_hidden/` | ~2 GB | CRaFT checkpoint |

**总计**: ~12 GB（包含数据集和 checkpoint）

---

## 🔄 工作流程

### 典型实验流程

```
1. 安装环境
   └─> 阅读 docs/QUICKSTART.md

2. Baseline 训练
   └─> 运行 python -m lerobot.scripts.lerobot_train ...
   └─> 输出到 outputs/baseline/

3. 生成 Hidden Cache
   └─> 运行 python -m lerobot.scripts.build_anchor_hidden_cache ...
   └─> 输出到 data/anchor_hidden_cache/

4. CRaFT 训练
   └─> 运行 python -m lerobot.scripts.lerobot_train_craft ...
   └─> 输出到 outputs/craft_hidden/

5. MCQ 评测
   └─> 运行 python -m lerobot.scripts.eval_mcq_likelihood ...
   └─> 输出到 results/metrics/

6. 结果分析
   └─> 查看 results/visualizations/
   └─> 生成报告 results/analysis/
```

---

## 📝 维护清单

### 定期清理

```bash
# 清理旧的 checkpoint（保留最新的）
rm -rf outputs/*/checkpoint-{2000,4000,6000,8000}

# 清理缓存
rm -rf data/datasets/*/cache

# 清理日志
rm -rf outputs/*/tensorboard
```

### 备份重要文件

```bash
# 备份配置
cp configs/*.yaml backups/configs/

# 备份最终 checkpoint
cp -r outputs/*/final backups/checkpoints/

# 备份结果
cp -r results/ backups/results/
```

---

## 🔗 相关链接

- **项目主页**: [README_CRAFT.md](../README_CRAFT.md)
- **文档导航**: [docs/README.md](README.md)
- **快速开始**: [docs/QUICKSTART.md](QUICKSTART.md)
- **实验指南**: [docs/EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)

---

**最后更新**: 2026-02-17

**提示**: 使用 `tree` 命令查看完整目录结构：
```bash
tree -L 3 -I '__pycache__|*.pyc|.git'
```

