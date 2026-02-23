# CRaFT 跨 Suite 泛化实验 - 常见问题

## 环境配置

### Q1: 如何安装 LIBERO？

```bash
pip install libero-robotics
```

如果遇到依赖问题，可以尝试：

```bash
pip install libero-robotics --no-deps
pip install -r requirements.txt
```

### Q2: CUDA 不可用怎么办？

如果没有 GPU，可以使用 CPU 运行（速度会很慢）：

```bash
# 在所有命令中将 --policy.device=cuda 改为 --policy.device=cpu
```

建议使用 Google Colab 或其他云 GPU 服务。

### Q3: 数据集下载失败

如果 HuggingFace Hub 连接失败，可以：

1. 使用镜像站点：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. 手动下载数据集：
```bash
huggingface-cli download lerobot/libero_spatial_no_noops --repo-type dataset
```

---

## 训练问题

### Q4: GPU 内存不足

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案**:

1. 减小批次大小：
```bash
--batch_size=16  # 从 32 降到 16
craft.anchor_batch_size=8  # 从 16 降到 8
```

2. 使用混合精度训练：
```bash
--policy.use_amp=true
```

3. 减少 worker 数量：
```bash
--num_workers=2  # 从 4 降到 2
```

### Q5: 训练速度太慢

**优化建议**:

1. 增加批次大小（如果 GPU 内存允许）：
```bash
--batch_size=64
```

2. 减少评估频率：
```bash
--eval_freq=5000  # 从 2000 增加到 5000
```

3. 使用更少的 worker：
```bash
--num_workers=0  # 禁用多进程数据加载
```

### Q6: Anchor Cache 生成失败

**错误信息**: `FileNotFoundError: metadata.json not found`

**解决方案**:

1. 检查输出目录权限
2. 重新运行生成脚本：
```bash
bash scripts/02_build_anchor_cache.sh
```

3. 手动验证 cache：
```python
import torch
cache = torch.load("outputs/anchor_cache/shard_0000.pt")
print(cache.keys())
```

---

## 评测问题

### Q7: 评测时环境初始化失败

**错误信息**: `Failed to initialize LIBERO environment`

**解决方案**:

1. 检查 LIBERO 安装：
```python
import libero
print(libero.__version__)
```

2. 设置环境变量：
```bash
export LIBERO_PATH=/path/to/libero
```

3. 使用虚拟显示（无头服务器）：
```bash
xvfb-run -a python -m lerobot.scripts.lerobot_eval ...
```

### Q8: 评测结果不一致

**可能原因**:

1. 随机种子不同
2. 环境初始化状态不同
3. 模型加载问题

**解决方案**:

1. 固定随机种子：
```bash
--seed=42
```

2. 增加评测 episodes：
```bash
--eval.n_episodes=100  # 从 50 增加到 100
```

3. 多次运行取平均值

---

## CRaFT 特定问题

### Q9: CRaFT 训练不收敛

**症状**: `retention_loss` 持续上升，`lambda` 持续增大

**解决方案**:

1. 调整 `lambda_lr`（降低）：
```bash
craft.lambda_lr=0.0005  # 从 0.001 降到 0.0005
```

2. 调整 `epsilon` 范围：
```bash
craft.epsilon_start=2.0  # 增大初始值
craft.epsilon_end=0.1  # 增大最终值
```

3. 增加 `retention_freq`：
```bash
craft.retention_freq=10  # 从 5 增加到 10
```

### Q10: CRaFT 性能不如 Baseline

**可能原因**:

1. 超参数设置不当
2. Anchor Cache 质量差
3. 训练步数不足

**解决方案**:

1. 增加训练步数：
```bash
--steps=20000  # 从 10000 增加到 20000
```

2. 调整 CRaFT 超参数：
```bash
craft.initial_lambda=0.5  # 降低初始 lambda
craft.use_grad_projection=false  # 尝试禁用梯度投影
```

3. 重新生成 Anchor Cache（增加样本数）：
```bash
--num_samples=2000  # 从 1000 增加到 2000
```

### Q11: 梯度冲突检测过于频繁

**症状**: 日志显示大量 `grad_conflict=True`

**解决方案**:

1. 调整冲突阈值：
```bash
craft.conflict_threshold=-0.2  # 从 -0.1 降到 -0.2（更宽松）
```

2. 禁用梯度投影：
```bash
craft.use_grad_projection=false
```

---

## 结果分析问题

### Q12: 报告生成失败

**错误信息**: `ModuleNotFoundError: No module named 'matplotlib'`

**解决方案**:

```bash
pip install matplotlib
```

### Q13: 成功率为 0%

**可能原因**:

1. 模型未正确加载
2. 评测环境配置错误
3. 训练步数太少

**解决方案**:

1. 检查 checkpoint 路径
2. 增加训练步数
3. 查看评测日志中的错误信息

---

## 实验设计问题

### Q14: 为什么选择 libero_spatial 作为 ID 任务？

`libero_spatial` 是一个中等难度的任务集，包含 10 个任务，适合作为微调目标。其他 Suites 作为 OOD 测试集，可以有效评估泛化能力。

### Q15: 可以使用其他 Suite 作为 ID 任务吗？

可以！只需修改脚本中的 `ENV_TASK` 和 `DATASET_REPO_ID`：

```bash
ENV_TASK="libero_object"
DATASET_REPO_ID="lerobot/libero_object_no_noops"
```

### Q16: 10000 步训练够吗？

10000 步是一个合理的起点。根据实际情况：

- 如果损失未收敛，增加到 20000 步
- 如果过拟合，减少到 5000 步
- 观察验证集性能决定

---

## 性能优化

### Q17: 如何加速实验？

1. **并行评测**（如果有多个 GPU）：
```bash
# GPU 0: 评测 Baseline
CUDA_VISIBLE_DEVICES=0 bash scripts/04_eval_cross_suite.sh &

# GPU 1: 评测 CRaFT
CUDA_VISIBLE_DEVICES=1 bash scripts/04_eval_cross_suite.sh &
```

2. **减少评测 episodes**：
```bash
N_EPISODES=25  # 从 50 降到 25
```

3. **使用更小的模型**（如果可用）

### Q18: 如何节省磁盘空间？

1. 禁用视频保存：
```bash
--eval.save_video=false
```

2. 只保存最终 checkpoint：
```bash
--save_freq=10000  # 只在最后保存
```

3. 使用更小的 Anchor Cache：
```bash
--num_samples=500  # 从 1000 降到 500
```

---

## 调试技巧

### Q19: 如何调试训练过程？

1. **启用详细日志**：
```bash
--log_freq=10  # 更频繁的日志
```

2. **使用 TensorBoard**（如果支持）：
```bash
tensorboard --logdir=outputs/
```

3. **检查中间输出**：
```python
# 在训练脚本中添加
print(f"task_loss: {task_loss.item()}")
print(f"retention_loss: {retention_loss.item()}")
print(f"lambda: {current_lambda}")
```

### Q20: 如何验证 CRaFT 是否正常工作？

检查训练日志中的关键指标：

1. `retention_loss` 应该逐渐下降并稳定在 `epsilon` 附近
2. `lambda` 应该在合理范围内（0.5-5.0）
3. `task_loss` 应该持续下降
4. `grad_conflict` 应该偶尔出现（不是每步都有）

---

## 联系支持

如果以上方案都无法解决问题，请：

1. 查看完整错误日志
2. 检查 GitHub Issues
3. 提交新的 Issue（附带完整错误信息和配置）

---

**最后更新**: 2025-02-23

