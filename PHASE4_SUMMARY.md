# CRaFT Hidden Retention 集成完成

## 阶段 4 总结

### ✅ 已完成任务

#### 1. CraftConfig 扩展
- 添加 `retention_mode` 字段（"hidden" 或 "token_ce"）
- 添加配置验证逻辑
- 默认值设为 "hidden"（推荐模式）

#### 2. 训练循环更新
- `update_policy_craft()` 根据 `retention_mode` 选择计算方式
- Hidden mode: 调用 `compute_hidden_retention_loss()`
- Token CE mode: 调用 `policy.forward()`（向后兼容）
- 清晰的错误提示（cache 类型不匹配）

#### 3. 日志增强
- 启动日志显示 "Retention Mode: hidden"
- 训练日志显示 "mode=hidden | L_ret=X.XXX"
- 保留所有原有指标（λ, ε, dot, cos）

#### 4. 脚本更新
- `scripts/train_craft.sh`: 添加 `retention_mode` 参数
- `scripts/train_craft_hidden_dryrun.sh`: 新增 3-step 测试脚本
- `tests.json`: 更新预期输出

#### 5. 文档
- `PHASE4_INTEGRATION_REPORT.md`: 完整实现报告
- 包含流程图、命令示例、预期日志

---

## 🚀 Dry-Run 命令

### 方式 1: 使用脚本
```bash
bash scripts/train_craft_hidden_dryrun.sh
```

### 方式 2: 直接命令
```bash
python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=outputs/craft_hidden_test \
    --steps=3 \
    --batch_size=2 \
    --eval_freq=0 \
    --log_freq=1 \
    --save_checkpoint=false \
    --num_workers=0 \
    --wandb.enable=false \
    craft.enabled=true \
    craft.retention_mode=hidden \
    craft.anchor_cache_dir=data/anchor_hidden_cache \
    craft.anchor_batch_size=2 \
    craft.retention_freq=1
```

---

## 📊 预期日志输出

### 启动阶段
```
================================================================================
CRaFT 训练配置
================================================================================
CRaFT 启用: True
Retention Mode: hidden          ← 新增
初始 λ: 1.0
λ 学习率: 0.01
...
================================================================================
```

### 训练阶段
```
Step 1/3 | loss=2.345 | grdn=1.234 | lr=1.0e-04 | updt_s=0.523 | data_s=0.012 | mode=hidden | L_ret=0.856 | λ=1.012 | ε=1.000 | dot=-0.234 | cos=-0.156
                                                                                  ^^^^^^^^^^^
                                                                                  新增指标
```

### 关键指标
- **mode=hidden**: 使用 hidden retention loss
- **L_ret**: 保留损失值
- **λ**: Lagrangian 乘子（动态调整）
- **ε**: 保留约束阈值（线性退火）
- **dot**: 梯度点积（负值表示冲突）
- **cos**: 梯度余弦相似度

---

## 📁 修改文件列表

```
src/lerobot/craft/craft_config.py              ← 添加 retention_mode 字段
src/lerobot/scripts/lerobot_train_craft.py    ← 更新训练循环逻辑
scripts/train_craft.sh                         ← 添加 retention_mode 参数
scripts/train_craft_hidden_dryrun.sh           ← 新增 dry-run 脚本
tests.json                                     ← 更新测试配置
PHASE4_INTEGRATION_REPORT.md                   ← 新增实现报告
```

---

## 🔄 Git 提交

```bash
Commit: af8a8b48
Message: feat: integrate hidden retention into craft training loop

Files changed: 9
Insertions: 1081
Deletions: 38
```

**不包含 push**（按要求）

---

## ✅ 验证清单

### 代码层面（已完成）
- [x] `retention_mode` 字段添加到 `CraftConfig`
- [x] 训练循环根据 `retention_mode` 分支
- [x] 错误提示清晰（cache 类型不匹配）
- [x] 日志输出包含 `mode` 指标
- [x] 训练脚本传递参数

### 功能层面（需要服务器测试）
- [ ] 生成 hidden feature cache
- [ ] 运行 dry-run 测试（3 steps）
- [ ] 验证日志输出符合预期
- [ ] 验证梯度能正常反向传播
- [ ] 运行完整训练（1000 steps）

---

## 🎯 下一步行动

### 1. 生成 Hidden Feature Cache
```bash
python -m lerobot.scripts.build_anchor_hidden_cache \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=data/anchor_hidden_cache \
    --num_samples=100 \
    --hidden_layer=-2 \
    --pooling=mean_image_tokens
```

### 2. 运行 Dry-Run 测试
```bash
bash scripts/train_craft_hidden_dryrun.sh
```

### 3. 检查日志
确认以下输出：
- ✓ "Retention Mode: hidden"
- ✓ "✓ AnchorCache 加载成功"
- ✓ "mode=hidden | L_ret=X.XXX"
- ✓ "λ=X.XXX | ε=X.XXX"
- ✓ "dot=X.XXX | cos=X.XXX"

### 4. 完整训练
```bash
bash scripts/train_craft.sh
```

---

## 🏆 技术亮点

### 1. 显式配置
- 不依赖自动检测，通过 `retention_mode` 显式指定
- 更可控、更清晰

### 2. 向后兼容
- 支持旧的 `token_ce` 模式
- 现有 token-level cache 仍可使用

### 3. 可扩展
- 未来可添加更多 retention 模式
- 例如: "hidden_multi_layer", "attention_map" 等

### 4. 错误处理
- Cache 类型不匹配时立即报错
- 错误提示清晰，指导用户使用正确的生成脚本

---

## 📚 相关文档

1. **PHASE4_INTEGRATION_REPORT.md**: 完整实现报告
2. **HIDDEN_RETENTION_LOSS_REPORT.md**: Hidden retention loss 数学验证
3. **HIDDEN_FEATURE_CACHE_SUMMARY.md**: Hidden feature cache 实现总结
4. **docs/CONTEXT.md**: 项目上下文（包含所有阶段）

---

## 🎉 阶段 4 完成！

所有代码已实现并提交，等待服务器上的真实数据测试验证。

