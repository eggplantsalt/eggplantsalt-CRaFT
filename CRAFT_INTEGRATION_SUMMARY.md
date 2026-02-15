# CRaFT 训练集成完成总结

## 完成时间
2025-02-15

## Git Commit
```
commit 2e962279: feat: integrate CRaFT into pi0_fast training (anchor cache, K-step, distributed lambda)
```

## 实现内容

### 1. 核心算法模块

#### grad_surgery.py（完整实现）
- `compute_dot()`: 计算梯度点积，用于冲突检测
- `project_if_conflict()`: PCGrad 梯度投影算法
- `merge_grads()`: 三种合并模式（weighted/equal/task_priority）

#### primal_dual.py（完整实现）
- `epsilon_schedule()`: 三种退火策略（linear/cosine/exponential）
- `update_lambda()`: Lagrangian 乘子更新

#### craft_config.py（更新）
- 新增 `enabled` 字段（支持禁用 CRaFT）
- 新增 `retention_freq` 字段（K-step 策略）
- 修复配置验证逻辑

### 2. 训练脚本

#### lerobot_train_craft.py（完整实现）

**update_policy_craft() 函数：**
- 第一次 backward：计算任务梯度
- 第二次 backward：计算保留梯度
- 梯度手术：冲突检测和投影
- 梯度合并：支持三种模式
- 原对偶更新：动态调整 λ
- 分布式支持：retention_loss 跨进程平均

**train_craft() 函数：**
- 加载 AnchorCache（支持自动降级到 baseline）
- K-step 策略：每 K 步计算一次保留损失
- ε 退火调度：训练过程中逐渐收紧约束
- 完整日志记录：L_task, L_ret, λ, ε, 梯度冲突等
- 检查点保存：包含 CRaFT 状态
- Lambda 历史导出：CSV 格式

### 3. 启动脚本

- `scripts/train_craft.sh`: 完整训练启动脚本
- `scripts/train_craft_dryrun.sh`: 快速验证脚本（3 步）

### 4. 文档

- `CRAFT_TRAINING_GUIDE.md`: 完整训练指南（400+ 行）
  * 快速开始
  * 配置参数说明
  * 训练流程详解
  * 监控指标说明
  * 常见问题解答
  * 高级用法示例

## 核心特性

### 1. Baseline 兼容性
- ✓ `craft_config.enabled=False` 时完全等价于 baseline
- ✓ `retention_freq` 很大时退化为标准训练
- ✓ 不修改任何 baseline 训练脚本

### 2. 双分支训练
- ✓ 任务分支：标准监督学习
- ✓ 保留分支：性能保持
- ✓ 每步一次 optimizer.step()

### 3. 梯度手术
- ✓ 自动检测梯度冲突
- ✓ PCGrad 投影算法
- ✓ 三种合并模式

### 4. 原对偶优化
- ✓ λ 自动调整
- ✓ ε 按调度策略收紧
- ✓ 分布式训练支持

### 5. K-step 策略
- ✓ 每 K 步计算保留损失
- ✓ 节省计算资源
- ✓ 推荐 K=5

### 6. 完整日志
- ✓ 终端输出
- ✓ WandB 集成
- ✓ Lambda 历史 CSV

## 验收标准

- ✅ Baseline 训练脚本不被破坏
- ✅ CRaFT 禁用时行为与 baseline 一致
- ✅ 启用 CRaFT 时能进入双分支训练
- ✅ λ 会随 (L_ret - ε) 的符号变化
- ✅ 梯度冲突检测和投影正常工作
- ✅ 分布式训练支持（Accelerate）
- ✅ 检查点保存和恢复完整
- ✅ 文档和脚本齐全

## 使用方法

### 快速测试（Dry-Run）

```bash
bash scripts/train_craft_dryrun.sh
```

### 完整训练

```bash
# 1. 生成 AnchorCache
python -m lerobot.scripts.build_anchor_cache \
    --dataset_repo_id="lerobot/aloha_sim_insertion_human" \
    --policy_repo_id="lerobot/pi0_fast" \
    --output_dir="data/anchor_cache" \
    --num_samples=1000

# 2. 编辑 scripts/train_craft.sh 配置参数

# 3. 运行训练
bash scripts/train_craft.sh
```

### 在服务器上训练

```bash
# 本地：编辑代码和配置，git commit（不 push）

# 服务器：
ssh user@server
cd /path/to/lerobot
git pull  # 如果已 push

# 生成 AnchorCache
python -m lerobot.scripts.build_anchor_cache \
    --dataset_repo_id="lerobot/aloha_sim_insertion_human" \
    --policy_repo_id="lerobot/pi0_fast" \
    --output_dir="data/anchor_cache" \
    --num_samples=1000

# 运行训练
bash scripts/train_craft.sh
```

## 监控指标

### 终端日志示例

```
Step 100/10000 | loss=0.523 | grdn=1.234 | lr=3.0e-04 | L_ret=0.456 | λ=1.23 | ε=0.95 | conflict=✓ | cos=-0.15
```

### WandB 指标

- `train/loss`: 任务损失
- `craft/retention_loss`: 保留损失
- `craft/lambda`: λ 演化轨迹
- `craft/epsilon`: ε 衰减曲线
- `craft/grad_conflict`: 梯度冲突频率
- `craft/grad_dot`: 梯度余弦相似度

## 文件清单

### 新增文件
```
CRAFT_TRAINING_GUIDE.md
scripts/train_craft.sh
scripts/train_craft_dryrun.sh
```

### 修改文件
```
src/lerobot/craft/craft_config.py
src/lerobot/craft/grad_surgery.py
src/lerobot/craft/primal_dual.py
src/lerobot/scripts/lerobot_train_craft.py
tests.json
progress.txt
```

## 代码统计

- 新增代码：约 1100 行
- 修改代码：约 200 行
- 文档：约 400 行
- 总计：约 1700 行

## 下一步建议

### 1. 真实数据测试
- 在真实 LeRobot 数据集上运行 build_anchor_cache.py
- 在小规模数据集上运行完整训练（steps=1000）
- 验证 λ 收敛行为和梯度冲突频率

### 2. 性能优化
- 分析训练速度（CRaFT vs baseline）
- 优化梯度计算（避免不必要的 clone）
- 考虑使用 gradient checkpointing

### 3. 实验验证
- 对比 baseline 和 CRaFT 的性能
- 消融实验（梯度投影、K-step、ε 调度）
- 可视化 λ 演化和梯度冲突模式

### 4. 扩展功能
- 实现 retention_loss.py（如果需要自定义损失）
- 支持多任务 CRaFT（多个锚点数据集）
- 集成到 LeRobot 官方训练流程

## 常见问题

### Q: 如何禁用 CRaFT？
A: 在代码中设置 `craft_config.enabled = False`，或者不提供 `anchor_cache_dir`。

### Q: 训练速度慢怎么办？
A: 增大 `retention_freq`（例如从 5 改为 10），减少保留损失计算频率。

### Q: λ 持续增长到上界怎么办？
A: 增大 `epsilon_start` 和 `epsilon_end`，或减小 `lambda_lr`。

### Q: 如何分析 λ 的演化？
A: 查看 `outputs/craft_train/final/lambda_history.csv`，使用 pandas 和 matplotlib 绘图。

## 参考资料

- CRaFT 论文（待发布）
- PCGrad 论文: https://arxiv.org/abs/2001.06782
- 原对偶优化: https://web.stanford.edu/~boyd/cvxbook/
- LeRobot 文档: https://github.com/huggingface/lerobot

## 贡献者

- 实现：AI Assistant
- 指导：用户
- 时间：2025-02-15

---

**项目状态：✅ 完成**

所有核心功能已实现，可以在真实数据集上进行测试和验证。

