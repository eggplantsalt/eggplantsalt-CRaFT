# MCQ Likelihood Evaluation - 实现总结

## ✅ 阶段 5 完成

### 任务目标
创建一个评测脚本，使用 forward logits（而非 generate）计算多选题答案概率。

---

## 📁 新增文件

### 1. 核心脚本
**文件**: `src/lerobot/scripts/eval_mcq_likelihood.py` (约 400 行)

**功能**:
- 加载 JSONL 格式的多选题数据
- 对每个选项计算 log P(choice_tokens | image+question)
- 使用 teacher forcing 累加 token log-probability
- 选择 log-likelihood 最大的选项作为预测
- 输出 accuracy 和 average margin (top1-top2)
- 支持对比两个 checkpoint

**核心函数**:
```python
def compute_choice_loglikelihood(policy, image_tensor, question_text, choice_text, device):
    """计算单个 choice 的 log-likelihood"""
    # 1. 构造 prompt: "{question}\nAnswer: {choice}"
    # 2. Tokenize
    # 3. Embed prefix (image + tokens)
    # 4. Forward pass 获取 logits
    # 5. 计算 choice tokens 的 log-probability
    # 6. 累加得到总 log-likelihood
    return log_likelihood

def evaluate_sample(policy, sample, device):
    """评测单个样本"""
    # 1. 加载图像
    # 2. 对每个 choice 计算 log-likelihood
    # 3. 选择最大的作为预测
    # 4. 计算 margin (top1 - top2)
    return predicted_index, log_likelihoods, correct, margin

def evaluate_checkpoint(checkpoint_path, data_jsonl, max_samples, batch_size, device):
    """评测单个 checkpoint"""
    # 1. 加载 policy
    # 2. 加载数据
    # 3. 逐样本评测
    # 4. 计算 accuracy 和 avg_margin
    return results

def compare_checkpoints(checkpoint_a, checkpoint_b, ...):
    """对比两个 checkpoint"""
    # 1. 分别评测两个 checkpoint
    # 2. 输出对比结果
    return results_a, results_b
```

### 2. Smoke Test
**文件**: `tests/test_mcq_likelihood_smoke.py` (约 200 行)

**功能**:
- 创建测试图像和 JSONL 数据
- 验证数据加载和格式
- 验证图像预处理
- Mock evaluation 结构验证

**测试用例**:
```python
def test_mcq_likelihood_smoke():
    """数据加载和格式验证"""
    # 1. 创建临时测试数据（2 条样本）
    # 2. 验证 JSONL 格式
    # 3. 验证图像加载
    # 4. 验证 tensor shape 和 dtype

def test_mcq_likelihood_mock_evaluation():
    """Mock evaluation 结构验证"""
    # 1. 创建 mock policy
    # 2. 验证基本结构
```

### 3. 文档
**文件**: `docs/MCQ_LIKELIHOOD_EVAL.md` (约 400 行)

**内容**:
- 使用方法（基础评测、对比评测、保存结果）
- 数据格式（JSONL 格式说明和示例）
- 评测原理（log-likelihood 计算、margin 计算）
- 输出指标（accuracy, avg_margin）
- 使用场景（评测 CRaFT、持续学习、快速验证）
- 注意事项和故障排除

### 4. 示例数据
**文件**: `data/mcq_test_sample.jsonl` (2 条样本)

```jsonl
{"image_path": "...", "question": "...", "choices": [...], "answer_index": 0}
{"image_path": "...", "question": "...", "choices": [...], "answer_index": 2}
```

---

## 🎯 核心特性

### 1. Teacher Forcing 计算
不使用 `generate()`，而是：
```python
# 1. 构造完整序列：image + question + "Answer: " + choice
# 2. Forward pass 获取 logits
# 3. 提取 choice tokens 对应位置的 logits
# 4. 计算 log P(choice_tokens | prefix)
# 5. 累加得到总 log-likelihood
```

### 2. 选项选择
```python
# 对每个 choice 计算 log-likelihood
log_likelihoods = [compute_choice_loglikelihood(...) for choice in choices]

# 选择最大的
predicted_index = argmax(log_likelihoods)
```

### 3. Margin 计算
```python
# 衡量模型置信度
sorted_logliks = sorted(log_likelihoods, reverse=True)
margin = sorted_logliks[0] - sorted_logliks[1]
```

### 4. 双 Checkpoint 对比
```python
# 评测两个 checkpoint
results_a = evaluate_checkpoint(checkpoint_a, ...)
results_b = evaluate_checkpoint(checkpoint_b, ...)

# 输出差异
print(f"Accuracy: {results_b['accuracy'] - results_a['accuracy']:.2%}")
print(f"Avg Margin: {results_b['avg_margin'] - results_a['avg_margin']:.4f}")
```

---

## 📊 使用示例

### 基础评测
```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model_checkpoint \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=100
```

**输出**:
```
================================================================================
评测结果
================================================================================
Checkpoint: outputs/model_checkpoint
Accuracy: 85.00%
Average Margin (top1 - top2): 2.3456
Correct: 85/100
```

### 对比评测
```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline \
    --checkpoint_path_b=outputs/craft_trained \
    --data_jsonl=data/mcq_test.jsonl
```

**输出**:
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

### Smoke Test
```bash
python tests/test_mcq_likelihood_smoke.py
```

**输出**:
```
================================================================================
MCQ Likelihood Evaluation - Smoke Test
================================================================================

[Test 1] Data loading and format validation
✓ Smoke test passed!
  - Loaded 2 samples
  - Image shape: torch.Size([3, 224, 224])
  - Sample format validated

[Test 2] Mock evaluation structure
✓ Mock evaluation structure validated!

================================================================================
All smoke tests passed! ✓
================================================================================
```

---

## 🔧 技术实现

### 1. 图像预处理
```python
def load_image(image_path, image_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor  # [C, H, W], [0, 1]
```

### 2. Prompt 构造
```python
prompt = f"{question_text}\nAnswer: {choice_text}"
tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
```

### 3. Forward Pass
```python
# Embed prefix (image + tokens + BOS)
prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = \
    policy.model.embed_prefix_fast(images, img_masks, tokens_with_bos, masks_with_bos)

# Forward pass
position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
att_4d = policy.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

(prefix_out, _), _ = policy.model.paligemma_with_expert.forward(
    attention_mask=att_4d,
    position_ids=position_ids,
    inputs_embeds=[prefix_embs, None],
    use_cache=False,
)

# Get logits
logits = lm_head(prefix_out)  # [1, seq_len, vocab_size]
```

### 4. Log-Likelihood 计算
```python
# 提取 choice tokens 对应的 logits
choice_logits = logits[:, -(num_choice_tokens+1):-1, :]

# 计算 log probabilities
log_probs = F.log_softmax(choice_logits, dim=-1)

# 提取目标 token 的 log prob
target_log_probs = log_probs.gather(dim=-1, index=choice_targets.unsqueeze(-1)).squeeze(-1)

# 累加
log_likelihood = target_log_probs.sum().item()
```

---

## 📈 应用场景

### 1. 评测 CRaFT 训练效果
对比训练前后模型在多选题上的表现：
```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/baseline \
    --checkpoint_path_b=outputs/craft_trained \
    --data_jsonl=data/mcq_test.jsonl
```

### 2. 评测持续学习能力
评测模型在旧任务上的保留能力：
```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/after_new_task \
    --data_jsonl=data/old_task_mcq.jsonl
```

### 3. 快速验证
使用少量样本快速验证模型：
```bash
python -m lerobot.scripts.eval_mcq_likelihood \
    --checkpoint_path=outputs/model \
    --data_jsonl=data/mcq_test.jsonl \
    --max_samples=10
```

---

## ⚠️ 注意事项

### 1. 当前限制
- **逐样本评测**: 当前实现为逐样本评测（batch_size 参数保留用于未来优化）
- **Token 对齐**: 假设 choice tokens 在序列末尾，如果 tokenization 方式不同可能需要调整
- **内存使用**: 每次 forward pass 都重新计算完整序列

### 2. 未来优化
- 支持 batch 评测以提高速度
- 支持 KV cache 以减少重复计算
- 支持更灵活的 prompt 格式
- 添加更多评测指标（entropy, confidence 等）

### 3. 依赖
- Pi0Fast policy 必须已加载
- 图像文件必须存在且可读
- JSONL 格式必须正确

---

## 📦 Git 提交

```bash
Commit: ad3f4dce
Message: feat: add MCQ likelihood eval script for pi0_fast

Files changed: 3
Insertions: 990
- src/lerobot/scripts/eval_mcq_likelihood.py (新增)
- tests/test_mcq_likelihood_smoke.py (新增)
- docs/MCQ_LIKELIHOOD_EVAL.md (新增)
- data/mcq_test_sample.jsonl (新增)
```

**未执行 push**（按要求）

---

## ✅ 完成清单

- [x] 创建 `eval_mcq_likelihood.py` 脚本
- [x] 实现 log-likelihood 计算（teacher forcing）
- [x] 实现单 checkpoint 评测
- [x] 实现双 checkpoint 对比
- [x] 输出 accuracy 和 avg_margin
- [x] 创建 smoke test（2 条样例）
- [x] 创建完整文档
- [x] 创建示例 JSONL 数据
- [x] Git commit（未 push）

---

## 🎉 阶段 5 完成！

MCQ likelihood 评测脚本已完成，可用于评测 Pi0Fast 模型在多选题任务上的表现，特别适合评测 CRaFT 训练效果和持续学习能力。

