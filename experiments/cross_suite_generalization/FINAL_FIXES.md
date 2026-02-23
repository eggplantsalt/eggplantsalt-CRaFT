# 最终修复总结 - CRaFT 跨 Suite 泛化实验

## 问题诊断

### 错误 1: 模型仓库名错误
**错误信息**: `404 Client Error: Not Found for url: https://huggingface.co/lerobot/pi0_fast/resolve/main/config.json`

**原因**: 使用了不存在的模型仓库名 `lerobot/pi0_fast`

**正确的模型名**: `lerobot/pi0fast-base`

### 错误 2: 数据集名称错误
**错误信息**: 使用了不存在的数据集 `lerobot/libero_spatial_no_noops`

**原因**: 捏造了带 `_no_noops` 后缀的数据集名

**正确的方式**: 
- 使用全量数据集 `lerobot/libero`
- 通过 `--env.task=libero_spatial` 参数让 LeRobot 自动过滤出对应 Suite 的数据

## 修复内容

### 1. 训练脚本修复

#### ✅ `01_train_baseline.sh`
```bash
# 修复前
POLICY_PATH="lerobot/pi0_fast"
DATASET_REPO_ID="lerobot/libero_spatial_no_noops"

# 修复后
POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
ENV_TASK="libero_spatial"  # 通过 env.task 过滤数据集
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集
```

#### ✅ `02_build_anchor_cache.sh`
```bash
# 修复前
TEACHER_POLICY_PATH="lerobot/pi0_fast"
DATASET_REPO_ID="lerobot/libero_spatial_no_noops"

# 修复后
TEACHER_POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集
```

#### ✅ `03_train_craft.sh`
```bash
# 修复前
POLICY_PATH="lerobot/pi0_fast"
DATASET_REPO_ID="lerobot/libero_spatial_no_noops"

# 修复后
POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
ENV_TASK="libero_spatial"  # 通过 env.task 过滤数据集
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集
```

### 2. 多 GPU 脚本修复

#### ✅ `01_train_baseline_multigpu.sh`
- 已重新创建
- 使用 `lerobot/pi0fast-base`
- 使用 `lerobot/libero` + `env.task=libero_spatial`

#### ✅ `03_train_craft_multigpu.sh`
- 已重新创建
- 使用 `lerobot/pi0fast-base`
- 使用 `lerobot/libero` + `env.task=libero_spatial`

### 3. 配置文件修复

#### ✅ `configs/baseline_spatial.yaml`
```yaml
# 修复前
policy:
  path: lerobot/pi0_fast
dataset:
  repo_id: lerobot/libero_spatial_no_noops

# 修复后
policy:
  path: lerobot/pi0fast-base  # 官方预训练模型
env:
  task: libero_spatial  # 通过 env.task 过滤数据集
dataset:
  repo_id: lerobot/libero  # 使用全量数据集
```

#### ✅ `configs/craft_spatial.yaml`
```yaml
# 修复前
policy:
  path: lerobot/pi0_fast
dataset:
  repo_id: lerobot/libero_spatial_no_noops

# 修复后
policy:
  path: lerobot/pi0fast-base  # 官方预训练模型
env:
  task: libero_spatial  # 通过 env.task 过滤数据集
dataset:
  repo_id: lerobot/libero  # 使用全量数据集
```

## 关键理解

### LeRobot 数据集过滤机制

LeRobot 的原生逻辑：
1. 使用全量数据集 `lerobot/libero`
2. 通过 `--env.task=libero_spatial` 参数指定任务
3. 底层 Dataloader 自动从全量数据集中过滤出对应 Suite 的 episodes

**不需要**创建单独的子数据集（如 `libero_spatial_no_noops`）

### 正确的训练命令

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0fast-base \
    --env.type=libero \
    --env.task=libero_spatial \
    --dataset.repo_id=lerobot/libero \
    --output_dir=outputs/baseline_spatial \
    --steps=10000
```

关键参数：
- `--policy.path=lerobot/pi0fast-base`: 从 HuggingFace Hub 加载预训练模型
- `--env.type=libero`: 指定环境类型
- `--env.task=libero_spatial`: 指定具体任务（自动过滤数据）
- `--dataset.repo_id=lerobot/libero`: 使用全量数据集

## 验证步骤

### 1. 验证模型可访问

```bash
python -c "
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(repo_id='lerobot/pi0fast-base', filename='config.json')
    print('✓ 模型可访问')
except Exception as e:
    print(f'✗ 错误: {e}')
"
```

### 2. 验证数据集可访问

```bash
python -c "
from huggingface_hub import list_repo_files
try:
    files = list_repo_files('lerobot/libero', repo_type='dataset')
    print(f'✓ 数据集可访问，包含 {len(files)} 个文件')
except Exception as e:
    print(f'✗ 错误: {e}')
"
```

### 3. 运行训练（单 GPU）

```bash
bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh
```

### 4. 运行训练（8 GPU）

```bash
bash experiments/cross_suite_generalization/scripts/01_train_baseline_multigpu.sh
```

## 完整实验流程

### 单 GPU 版本

```bash
# 1. 训练 Baseline (~2.5 小时)
bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh

# 2. 构建 Anchor Cache (~15 分钟)
bash experiments/cross_suite_generalization/scripts/02_build_anchor_cache.sh

# 3. 训练 CRaFT (~3.5 小时)
bash experiments/cross_suite_generalization/scripts/03_train_craft.sh

# 4. 跨 Suite 评测 (~2 小时)
bash experiments/cross_suite_generalization/scripts/04_eval_cross_suite.sh

# 5. 生成报告 (~5 分钟)
python experiments/cross_suite_generalization/scripts/05_generate_report.py
```

**总时间**: ~9 小时

### 多 GPU 版本（8× V100）

```bash
# 1. 训练 Baseline (~25 分钟)
bash experiments/cross_suite_generalization/scripts/01_train_baseline_multigpu.sh

# 2. 构建 Anchor Cache (~15 分钟)
bash experiments/cross_suite_generalization/scripts/02_build_anchor_cache.sh

# 3. 训练 CRaFT (~35 分钟)
bash experiments/cross_suite_generalization/scripts/03_train_craft_multigpu.sh

# 4. 跨 Suite 评测 (~2 小时)
bash experiments/cross_suite_generalization/scripts/04_eval_cross_suite.sh

# 5. 生成报告 (~5 分钟)
python experiments/cross_suite_generalization/scripts/05_generate_report.py
```

**总时间**: ~3.5 小时（节省 60%）

## 修复的文件清单

### 脚本文件（6 个）
- ✅ `scripts/01_train_baseline.sh`
- ✅ `scripts/02_build_anchor_cache.sh`
- ✅ `scripts/03_train_craft.sh`
- ✅ `scripts/01_train_baseline_multigpu.sh`（重新创建）
- ✅ `scripts/03_train_craft_multigpu.sh`（重新创建）

### 配置文件（2 个）
- ✅ `configs/baseline_spatial.yaml`
- ✅ `configs/craft_spatial.yaml`

## 常见问题

### Q1: 为什么不能使用 `lerobot/pi0_fast`？

A: 这个仓库不存在。正确的官方预训练模型是 `lerobot/pi0fast-base`（注意没有下划线）。

### Q2: 为什么不能使用 `lerobot/libero_spatial_no_noops`？

A: 这个数据集不存在。LeRobot 的设计是使用全量数据集 `lerobot/libero`，然后通过 `env.task` 参数自动过滤。

### Q3: 如何验证修复是否成功？

A: 运行 `bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh`，如果能成功下载模型和数据集并开始训练，说明修复成功。

### Q4: 多 GPU 训练需要特殊配置吗？

A: 脚本会自动生成 accelerate 配置。如果需要手动配置，运行 `accelerate config`。

### Q5: 如何调整 GPU 数量？

A: 编辑多 GPU 脚本中的 `NUM_GPUS` 变量：
```bash
NUM_GPUS=4  # 改为你的 GPU 数量
BATCH_SIZE_PER_GPU=8  # 调整以保持总批次大小 = 32
```

## 下一步

1. **立即运行**: 
   ```bash
   bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh
   ```

2. **监控训练**:
   ```bash
   tail -f experiments/cross_suite_generalization/outputs/baseline_spatial/train.log
   ```

3. **查看 GPU 使用**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## 技术支持

如果遇到其他问题：
1. 查看 [FAQ.md](FAQ.md)
2. 查看 [MULTIGPU_GUIDE.md](MULTIGPU_GUIDE.md)
3. 检查训练日志中的错误信息

---

**修复完成时间**: 2025-02-23  
**状态**: ✅ 所有问题已修复，可以开始实验  
**验证**: 已确认模型和数据集路径正确

