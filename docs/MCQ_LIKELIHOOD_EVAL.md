# MCQ Likelihood Evaluation

## 概述

`eval_mcq_likelihood.py` 是一个用于评测 Pi0Fast 模型在多选题（Multiple Choice Question）任务上的脚本。

**核心特点**：
- 不使用 `generate()`，仅用 `forward()` 计算 logits
- 使用 teacher forcing 计算每个选项的 log-likelihood
- 支持单 checkpoint 评测和双 checkpoint 对比

---

## 使用方法

### 1. 基础评测

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model_checkpoint \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=100 \
    --batch_size=4
```

### 2. 对比两个 checkpoint

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/checkpoint_before \
    --checkpoint_path_b=outputs/checkpoint_after \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=100
```

### 3. 保存详细结果

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model_checkpoint \
    --data_jsonl=data/mcq_test.jsonl \
    --output_json=results/mcq_results.json
```

---

## 数据格式

### JSONL 格式

每行一个 JSON 对象：

```json
{
  "image_path": "path/to/image.jpg",
  "question": "What action should the robot take?",
  "choices": [
    "pick up the cup",
    "move left",
    "stop"
  ],
  "answer_index": 0
}
```

**字段说明**：
- `image_path`: 图像文件路径（相对或绝对路径）
- `question`: 问题文本
- `choices`: 选项列表（字符串数组）
- `answer_index`: 正确答案的索引（0-based）

### 示例数据

创建测试数据：

```python
import json

samples = [
    {
        "image_path": "data/images/scene1.jpg",
        "question": "What is the robot doing?",
        "choices": [
            "picking up an object",
            "moving to the left",
            "stopping and waiting"
        ],
        "answer_index": 0
    },
    {
        "image_path": "data/images/scene2.jpg",
        "question": "What should the robot do next?",
        "choices": [
            "continue forward",
            "turn around",
            "grasp the handle"
        ],
        "answer_index": 2
    }
]

with open("data/mcq_test.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
```

---

## 评测原理

### Log-Likelihood 计算

对于每个选项，脚本计算：

```
log P(choice_tokens | image + question)
```

**步骤**：
1. 构造完整 prompt：`{question}\nAnswer: {choice}`
2. Tokenize prompt
3. 使用 teacher forcing：将 choice tokens 作为目标
4. 计算每个 choice token 的 log-probability
5. 累加得到总 log-likelihood

**选择预测**：
- 选择 log-likelihood 最大的选项作为预测

### Margin 计算

```
margin = log_lik(top1) - log_lik(top2)
```

- Margin 越大，模型对 top1 选项越有信心
- 平均 margin 反映模型的整体置信度

---

## 输出指标

### 单 Checkpoint 评测

```
================================================================================
评测结果
================================================================================
Checkpoint: outputs/model_checkpoint
Accuracy: 85.00%
Average Margin (top1 - top2): 2.3456
Correct: 85/100
```

**指标说明**：
- **Accuracy**: 正确率（预测正确的样本数 / 总样本数）
- **Average Margin**: top1 和 top2 选项的 log-likelihood 差值平均值
- **Correct**: 正确样本数 / 总样本数

### 双 Checkpoint 对比

```
================================================================================
对比结果
================================================================================
Checkpoint A: outputs/checkpoint_before
  Accuracy: 75.00%
  Avg Margin: 1.8234
  Correct: 75/100

Checkpoint B: outputs/checkpoint_after
  Accuracy: 85.00%
  Avg Margin: 2.3456
  Correct: 85/100

差异:
  Accuracy: +10.00%
  Avg Margin: +0.5222
```

---

## 详细结果 JSON

使用 `--output_json` 保存详细结果：

```json
{
  "accuracy": 0.85,
  "avg_margin": 2.3456,
  "correct_count": 85,
  "total_count": 100,
  "results": [
    {
      "predicted_index": 0,
      "log_likelihoods": [-2.34, -5.67, -8.90],
      "correct": true,
      "margin": 3.33,
      "answer_index": 0
    },
    ...
  ]
}
```

---

## Smoke Test

运行最小测试（不需要真实 checkpoint）：

```bash
python tests/test_mcq_likelihood_smoke.py
```

**测试内容**：
1. 数据加载和格式验证
2. 图像预处理
3. Mock evaluation 结构验证

---

## 使用场景

### 1. 评测 CRaFT 训练效果

对比训练前后的模型：

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline \
    --checkpoint_path_b=outputs/craft_trained \
    --data_jsonl=data/mcq_test.jsonl
```

### 2. 评测持续学习能力

评测模型在旧任务上的保留能力：

```bash
# 评测在旧任务数据上的表现
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/after_new_task \
    --data_jsonl=data/old_task_mcq.jsonl
```

### 3. 快速验证

使用少量样本快速验证：

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=10
```

---

## 注意事项

### 1. 图像格式

- 支持常见图像格式（JPG, PNG 等）
- 自动 resize 到 224x224
- 归一化到 [0, 1]

### 2. Token 对齐

- 脚本假设 choice tokens 在序列末尾
- 如果 tokenization 方式不同，可能需要调整代码

### 3. 内存使用

- 当前实现为逐样本评测（未使用 batch）
- 未来可优化为 batch 评测以提高速度

### 4. 设备选择

默认使用 CUDA（如果可用），可通过 `--device` 指定：

```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model \
    --data_jsonl=data/mcq_test.jsonl \
    --device=cpu
```

---

## 扩展

### 自定义 Prompt 格式

修改 `compute_choice_loglikelihood()` 中的 prompt 构造：

```python
# 当前格式
prompt = f"{question_text}\nAnswer: {choice_text}"

# 自定义格式
prompt = f"Question: {question_text}\nThe robot should: {choice_text}"
```

### 添加更多指标

在 `evaluate_sample()` 中添加自定义指标：

```python
# 计算 entropy
probs = np.exp(log_likelihoods) / np.sum(np.exp(log_likelihoods))
entropy = -np.sum(probs * np.log(probs + 1e-10))
```

---

## 故障排除

### 问题 1: Checkpoint 加载失败

```
FileNotFoundError: Checkpoint not found
```

**解决方案**：
- 检查 checkpoint 路径是否正确
- 确保 checkpoint 目录包含必要文件（config.json, model.safetensors 等）

### 问题 2: 图像加载失败

```
FileNotFoundError: Image not found
```

**解决方案**：
- 检查 JSONL 中的 image_path 是否正确
- 使用绝对路径或相对于脚本运行目录的路径

### 问题 3: OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 使用 `--device=cpu`
- 减少 `--max_samples`
- 未来版本将支持更小的 batch size

---

## 参考

- Pi0Fast 论文: [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
- Teacher Forcing: 使用真实目标序列作为输入，计算 log-likelihood
- Log-Likelihood: 衡量模型对特定序列的置信度

---

## 贡献

欢迎提交 PR 改进此脚本：
- 添加 batch 评测支持
- 支持更多 prompt 格式
- 添加更多评测指标
- 优化内存使用

