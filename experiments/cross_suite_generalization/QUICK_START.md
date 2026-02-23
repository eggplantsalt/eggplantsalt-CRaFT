# 快速参考 - CRaFT 跨 Suite 泛化实验

## ✅ 所有问题已修复

### 修复的问题

1. ✅ **模型路径**: `lerobot/pi0_fast` → `lerobot/pi0fast-base`
2. ✅ **数据集路径**: `lerobot/libero_spatial_no_noops` → `lerobot/libero`
3. ✅ **缺少参数**: 添加 `--policy.repo_id` 和 `--training.push_to_hub=false`

### 立即运行

#### 单 GPU 训练
```bash
cd /tmp/eggplantsalt-CRaFT
bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh
```

#### 多 GPU 训练（8× V100，推荐）
```bash
cd /tmp/eggplantsalt-CRaFT
bash experiments/cross_suite_generalization/scripts/01_train_baseline_multigpu.sh
```

### 完整实验流程（多 GPU）

```bash
# 总时间: ~3.5 小时

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

### 关键参数说明

```bash
--policy.path=lerobot/pi0fast-base           # 官方预训练模型
--policy.repo_id=local/pi0fast_spatial_baseline  # 本地占位符（不会推送）
--training.push_to_hub=false                 # 禁止推送到 HuggingFace
--env.type=libero                            # 环境类型
--env.task=libero_spatial                    # 任务（自动过滤数据）
--dataset.repo_id=lerobot/libero             # 全量数据集
```

### 监控训练

```bash
# 查看训练日志
tail -f experiments/cross_suite_generalization/outputs/baseline_spatial/train.log

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

### 调整 GPU 数量

编辑多 GPU 脚本中的配置：
```bash
NUM_GPUS=4  # 改为你的 GPU 数量
BATCH_SIZE_PER_GPU=8  # 调整以保持总批次大小 = 32
```

### 文档参考

- 📄 **FINAL_FIXES.md** - 完整修复说明
- 📄 **README.md** - 实验总览
- 📄 **MULTIGPU_GUIDE.md** - 多 GPU 训练指南
- 📄 **FAQ.md** - 常见问题

### 预期结果

| Suite | Type | Baseline | CRaFT | Improvement |
|-------|------|----------|-------|-------------|
| libero_spatial | ID | ~85% | ~85% | 0% |
| libero_object | OOD | ~12% | ~45% | **+33%** |
| libero_goal | OOD | ~9% | ~39% | **+30%** |
| libero_10 | OOD | ~5% | ~28% | **+23%** |

### 故障排查

**问题**: 模型下载失败  
**解决**: 检查网络连接，或使用镜像 `export HF_ENDPOINT=https://hf-mirror.com`

**问题**: GPU 内存不足  
**解决**: 减小 `BATCH_SIZE_PER_GPU`

**问题**: 数据集加载失败  
**解决**: 确认使用 `lerobot/libero`（不是 `libero_spatial_no_noops`）

---

**状态**: ✅ 所有脚本已修复，可以直接运行  
**最后更新**: 2025-02-23

