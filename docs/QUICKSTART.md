# 快速开始指南

> 5 分钟上手 CRaFT 训练

---

## 前置要求

- Python >= 3.10
- CUDA >= 11.8（可选，用于 GPU 加速）
- 16GB+ RAM
- 50GB+ 可用磁盘空间

---

## 步骤 1: 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd lerobot

# 安装依赖
pip install -e .

# 验证安装
lerobot-info
```

**预期输出**:
```
LeRobot version: 0.4.4
Python version: 3.10.x
PyTorch version: 2.2.1
CUDA available: True
```

---

## 步骤 2: Baseline 训练（无 CRaFT）

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/baseline \
    --steps=1000 \
    --batch_size=8 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=500
```

**预期输出**:
```
Step 100/1000 | loss=2.345 | grdn=1.234 | lr=1.0e-04
Step 200/1000 | loss=2.123 | grdn=1.156 | lr=1.0e-04
...
Training completed!
```

---

## 步骤 3: 生成 Hidden Feature Cache

```bash
python -m lerobot.scripts.build_anchor_hidden_cache \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=data/anchor_hidden_cache \
    --num_samples=100 \
    --hidden_layer=-2 \
    --pooling=mean_image_tokens \
    --batch_size=8
```

**预期输出**:
```
================================================================================
Hidden Feature Cache 生成器
================================================================================
数据集: lerobot/aloha_sim_insertion_human
策略: lerobot/pi0_fast
输出目录: data/anchor_hidden_cache
样本数: 100
Hidden Layer: -2
Pooling: mean_image_tokens
================================================================================

Processing: 100%|████████████████████| 100/100 [00:30<00:00, 3.33it/s]

✓ Cache 生成完成！
  - 总样本数: 100
  - 输出目录: data/anchor_hidden_cache
  - 文件大小: 45.2 MB
```

---

## 步骤 4: CRaFT 训练

```bash
python -m lerobot.scripts.lerobot_train_craft \
    --policy.path=lerobot/pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/craft_trained \
    --steps=1000 \
    --batch_size=8 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=500 \
    craft.enabled=true \
    craft.retention_mode=hidden \
    craft.anchor_cache_dir=data/anchor_hidden_cache \
    craft.anchor_batch_size=8 \
    craft.retention_freq=1 \
    craft.initial_lambda=1.0 \
    craft.epsilon_start=1.0 \
    craft.epsilon_end=0.1
```

**预期输出**:
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
================================================================================

✓ AnchorCache 加载成功: 100 样本

Step 1/1000 | loss=2.345 | mode=hidden | L_ret=0.856 | λ=1.012 | ε=1.000 | dot=-0.234 | cos=-0.156
Step 2/1000 | loss=2.123 | mode=hidden | L_ret=0.789 | λ=1.019 | ε=0.999 | conflict=✓ | dot=-0.189 | cos=-0.123
...
```

---

## 步骤 5: MCQ 评测（可选）

```bash
# 准备测试数据（JSONL 格式）
cat > data/mcq_test.jsonl << EOF
{"image_path": "data/test_images/scene1.jpg", "question": "What should the robot do?", "choices": ["pick up", "move left", "stop"], "answer_index": 0}
{"image_path": "data/test_images/scene2.jpg", "question": "What is the robot doing?", "choices": ["grasping", "navigating", "observing"], "answer_index": 2}
EOF

# 对比两个 checkpoint
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline \
    --checkpoint_path_b=outputs/craft_trained \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=100
```

**预期输出**:
```
================================================================================
对比结果
================================================================================
Checkpoint A: outputs/baseline
  Accuracy: 75.00%
  Avg Margin: 1.8234
  Correct: 75/100

Checkpoint B: outputs/craft_trained
  Accuracy: 85.00%
  Avg Margin: 2.3456
  Correct: 85/100

差异:
  Accuracy: +10.00%
  Avg Margin: +0.5222
```

---

## 常见问题

### Q1: CUDA 不可用

```bash
# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，安装对应 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 内存不足

```bash
# 减小 batch_size
--batch_size=4

# 或使用梯度累积
--batch_size=4 --gradient_accumulation_steps=2
```

### Q3: 数据集下载慢

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定本地路径
--dataset.root=/path/to/local/dataset
```

---

## 下一步

- 📖 阅读 [完整实验指南](EXPERIMENT_GUIDE.md) 了解详细步骤
- 📚 查看 [CRaFT 训练指南](craft/CRAFT_TRAINING_GUIDE.md) 深入理解原理
- 🔧 参考 [API 文档](API_REFERENCE.md) 进行自定义开发

---

**提示**: 如果遇到问题，请查看 [故障排查指南](TROUBLESHOOTING.md) 或在 GitHub Issues 提问。

