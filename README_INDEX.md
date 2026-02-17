# 📋 LeRobot CRaFT 项目文件索引

## 🎯 快速导航

### 🚀 新手必读（按顺序阅读）

1. **快速开始** → [`docs/guides/README_HIDDEN_STATE.md`](docs/guides/README_HIDDEN_STATE.md)
2. **命令速查** → [`docs/guides/COMMANDS_CHEATSHEET.md`](docs/guides/COMMANDS_CHEATSHEET.md)
3. **完整指南** → [`docs/craft/HIDDEN_STATE_ANCHORING_GUIDE.md`](docs/craft/HIDDEN_STATE_ANCHORING_GUIDE.md)
4. **项目报告** → [`docs/reports/FINAL_REPORT.md`](docs/reports/FINAL_REPORT.md)

---

## 📂 文档目录

### ✅ 最新文档（Hidden State Anchoring）

#### 📖 使用指南
- [`docs/guides/README_HIDDEN_STATE.md`](docs/guides/README_HIDDEN_STATE.md) - 快速开始指南
- [`docs/guides/COMMANDS_CHEATSHEET.md`](docs/guides/COMMANDS_CHEATSHEET.md) - 命令速查表

#### 🔬 CRaFT 核心
- [`docs/craft/HIDDEN_STATE_ANCHORING_GUIDE.md`](docs/craft/HIDDEN_STATE_ANCHORING_GUIDE.md) - Hidden State 完整指南

#### 📊 项目报告
- [`docs/reports/FINAL_REPORT.md`](docs/reports/FINAL_REPORT.md) - 最终项目报告
- [`docs/reports/DELIVERY_SUMMARY.md`](docs/reports/DELIVERY_SUMMARY.md) - 交付总结
- [`docs/reports/IMPLEMENTATION_SUMMARY.md`](docs/reports/IMPLEMENTATION_SUMMARY.md) - 实现总结

### ⚠️ 旧版文档（Token-level Distillation，仅供参考）

#### 🔬 CRaFT 核心
- [`docs/craft/CRAFT_TRAINING_GUIDE.md`](docs/craft/CRAFT_TRAINING_GUIDE.md) - 训练指南（旧版）
- [`docs/craft/CRAFT_INTEGRATION_SUMMARY.md`](docs/craft/CRAFT_INTEGRATION_SUMMARY.md) - 集成总结（旧版）
- [`docs/craft/CRAFT_FILES.md`](docs/craft/CRAFT_FILES.md) - 文件说明（旧版）

#### 📖 使用指南
- [`docs/guides/ANCHOR_CACHE_GUIDE.md`](docs/guides/ANCHOR_CACHE_GUIDE.md) - AnchorCache 指南（旧版）
- [`docs/guides/ANCHOR_CACHE_SUMMARY.md`](docs/guides/ANCHOR_CACHE_SUMMARY.md) - AnchorCache 总结（旧版）

---

## 💻 源代码

### ✅ CRaFT 核心算法
- [`src/lerobot/craft/retention_loss.py`](src/lerobot/craft/retention_loss.py) - Hidden State Loss（最新）
- [`src/lerobot/craft/anchor_cache.py`](src/lerobot/craft/anchor_cache.py) - Cache 加载器（最新）
- [`src/lerobot/craft/grad_surgery.py`](src/lerobot/craft/grad_surgery.py) - 梯度手术
- [`src/lerobot/craft/primal_dual.py`](src/lerobot/craft/primal_dual.py) - 原对偶优化
- [`src/lerobot/craft/craft_config.py`](src/lerobot/craft/craft_config.py) - CRaFT 配置

### ✅ 训练脚本
- [`src/lerobot/scripts/build_anchor_cache.py`](src/lerobot/scripts/build_anchor_cache.py) - 生成 Hidden State Cache（最新）
- [`src/lerobot/scripts/lerobot_train_craft.py`](src/lerobot/scripts/lerobot_train_craft.py) - CRaFT 训练脚本（最新）

### ✅ Shell 脚本
- [`scripts/train_craft.sh`](scripts/train_craft.sh) - 完整训练脚本
- [`scripts/train_craft_dryrun.sh`](scripts/train_craft_dryrun.sh) - 快速验证脚本

---

## 🧪 测试文件

- [`tests/test_hidden_state_anchoring.py`](tests/test_hidden_state_anchoring.py) - Hidden State 单元测试（最新）
- [`tests/test_anchor_cache.py`](tests/test_anchor_cache.py) - AnchorCache 测试（旧版）
- [`tests/test_grad_surgery_math.py`](tests/test_grad_surgery_math.py) - 梯度手术数学验证

---

## 📝 项目记录

- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - 完整目录结构说明
- [`progress.txt`](progress.txt) - 项目进度记录
- [`tests.json`](tests.json) - 测试状态

---

## 🔍 版本说明

| 标识 | 说明 |
|------|------|
| ✅ **最新** | Hidden State Anchoring（本次修改后）|
| ⚠️ **旧版** | Token-level Distillation（部分内容已过时）|

---

## 🚀 快速命令

### 生成 Hidden State AnchorCache
```bash
python -m lerobot.scripts.build_anchor_cache \
    --policy.pretrained_path=physical-intelligence/pi0-fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --out_dir=data/anchor_cache_hidden \
    --num_anchors=1000 \
    --layers_to_save=-2,-1
```

### 训练
```bash
python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=outputs/craft_hidden \
    --steps=1000 \
    --batch_size=8
```

---

**更新时间**：2025-02-17  
**Git Commit**：9e78dc83  
**详细说明**：查看 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)

