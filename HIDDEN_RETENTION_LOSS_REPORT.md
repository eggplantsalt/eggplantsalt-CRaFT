# Hidden Retention Loss 实现总结

## 任务完成情况

✅ **所有任务已完成并通过测试**

---

## 📁 修改文件列表

### 修改文件（1个）

1. **src/lerobot/craft/retention_loss.py**
   - 新增 `compute_hidden_retention_loss(policy, anchor_batch, craft_config)` 主入口函数
   - 新增 `extract_student_hidden_features()` 提取 student hidden states
   - 新增 `pool_hidden_states()` 支持 4 种 pooling 策略
   - 新增 `identify_image_tokens()` 识别图像 tokens（3 层 fallback）
   - 更新 `compute_retention_loss_hidden()` 在 float32 中计算（稳定性）

### 新增文件（1个）

2. **tests/test_hidden_retention_loss_math.py** (340+ 行)
   - 7 个 CPU 单元测试
   - 使用 tiny mock Transformer 验证数学正确性
   - **测试结果：全部通过 ✓**

---

## 🎯 核心功能实现

### 1. compute_hidden_retention_loss()

主入口函数，负责完整的 retention loss 计算流程。

```python
def compute_hidden_retention_loss(
    policy,
    anchor_batch: dict,
    craft_config,
) -> tuple[Tensor, dict]:
    """
    计算 hidden state 保留损失
    
    【返回值】
    - loss: 保留损失标量张量（float32）
    - metrics: 指标字典
        - retention_loss: loss 值
        - student_hidden_norm: student hidden states 的范数
        - target_features_norm: target features 的范数
    """
```

**实现要点：**
- ✅ 使用 policy 原生的 `output_hidden_states=True`（最小侵入）
- ✅ 从 `craft_config.hidden_layer` 提取指定层
- ✅ 使用与 cache 一致的 pooling 策略
- ✅ 在 float32 中计算 loss（数值稳定）
- ✅ 支持反向传播

### 2. extract_student_hidden_features()

从 student 模型提取 hidden features。

```python
def extract_student_hidden_features(
    policy,
    anchor_batch: dict,
    craft_config,
) -> Tensor:
    """
    提取 student hidden features
    
    【返回值】
    Tensor: [B, hidden_dim] 的 pooled features
    """
```

**实现策略：**
1. **优先使用原生 API**：
   ```python
   if hasattr(policy, '_paligemma_model'):
       outputs = policy._paligemma_model(
           pixel_values=pixel_values,
           input_ids=input_ids,
           attention_mask=attention_mask,
           output_hidden_states=True,
       )
   ```

2. **Fallback 手动构造**：
   ```python
   else:
       # 手动构造 forward pass
       prefix_embs, ... = policy.model.embed_prefix_fast(...)
       outputs = language_model.forward(
           inputs_embeds=prefix_embs,
           output_hidden_states=True,
       )
   ```

3. **提取指定层**：
   ```python
   hidden_layer = meta.get("hidden_layer", -2)
   if hidden_layer < 0:
       actual_idx = total_layers + hidden_layer + 1
   hidden_state = all_hidden_states[actual_idx]
   ```

### 3. pool_hidden_states()

支持 4 种 pooling 策略。

```python
def pool_hidden_states(
    hidden_states: Tensor,  # [B, seq_len, hidden_dim]
    attention_mask: Tensor,
    pooling: str,
    policy,
    input_ids: Tensor,
) -> Tensor:  # [B, hidden_dim]
```

**Pooling 策略：**

#### mean_image_tokens（推荐）
```python
# 识别图像 tokens 范围
num_image_tokens = identify_image_tokens(policy)  # 196 for PaliGemma

# 提取图像 tokens
image_hidden = hidden_states[:, :num_image_tokens, :]  # [B, 196, hidden_dim]

# 平均池化
pooled = image_hidden.mean(dim=1)  # [B, hidden_dim]
```

#### mean_masked
```python
# 对所有非 padding tokens 取平均
mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
masked_hidden = hidden_states * mask
pooled = masked_hidden.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
```

#### last_token
```python
# 取最后一个非 padding token
lengths = attention_mask.sum(dim=1).long() - 1  # [B]
pooled = hidden_states[torch.arange(B), lengths]
```

#### cls_token
```python
# 取第一个 token
pooled = hidden_states[:, 0, :]
```

### 4. identify_image_tokens()

识别图像 tokens 数量（3 层 fallback）。

```python
def identify_image_tokens(policy) -> int:
    # 方法 1: 从 policy config 获取（优先）
    if hasattr(policy, 'config') and hasattr(policy.config, 'image_seq_length'):
        return policy.config.image_seq_length
    
    # 方法 2: 计算
    if hasattr(policy, 'config') and hasattr(policy.config, 'image_resolution'):
        h, w = policy.config.image_resolution
        patch_size = policy.config.patch_size
        return (h // patch_size) * (w // patch_size)
    
    # 方法 3: 默认值（PaliGemma 224x224, patch_size=16）
    return 196
```

---

## ✅ 测试结果

### 7 个单元测试全部通过

```
============================================================
[SUCCESS] All tests passed!
============================================================

Test 1: MSE Loss Correctness
[OK] MSE Loss: 1.979173
[OK] Expected: 1.979173
[OK] Difference: 0.00e+00

Test 2: Cosine Loss Correctness
[OK] Cosine Loss: 1.012244
[OK] Expected: 1.012244
[OK] Cosine Similarity: -0.012243

Test 3: Loss Range
[OK] Identical hidden states:
     MSE Loss: 0.00e+00 (should be ~0)
     Cosine Loss: -4.47e-08 (should be ~0)
[OK] Opposite hidden states:
     Cosine Loss: 2.000000 (should be ~2)

Test 4: Gradient Flow
[OK] Gradient exists: True
[OK] Gradient norm: 0.086421
[OK] Gradient shape: torch.Size([4, 2, 2, 64])

Test 5: Pooling Strategies
[OK] mean_image_tokens   : shape=torch.Size([4, 64]), norm=1.8710
[OK] mean_masked         : shape=torch.Size([4, 64]), norm=1.1177
[OK] last_token          : shape=torch.Size([4, 64]), norm=8.1868
[OK] cls_token           : shape=torch.Size([4, 64]), norm=8.2046

Test 6: Float32 Stability
[OK] Input dtype: torch.float16
[OK] Loss dtype: torch.float32
[OK] Loss value: 1.916182
[OK] Loss is finite: True

Test 7: End-to-End with Tiny Transformer
[OK] Student hidden shape: torch.Size([4, 20, 64])
[OK] Teacher hidden shape: torch.Size([4, 20, 64])
[OK] Pooled shape: torch.Size([4, 64])
[OK] Loss: 0.268769
[OK] Gradients exist: True
```

---

## 🔧 使用示例

### 在训练循环中使用

```python
from lerobot.craft.retention_loss import compute_hidden_retention_loss

# 在训练循环中
for step in range(total_steps):
    # 1. 计算任务损失
    task_batch = next(task_dataloader)
    task_loss, _ = policy.forward(task_batch)
    
    # 2. 计算保留损失（每 K 步）
    if step % craft_config.retention_freq == 0:
        anchor_batch = next(anchor_dl_iter)
        
        # 计算 hidden retention loss
        retention_loss, metrics = compute_hidden_retention_loss(
            policy,
            anchor_batch,
            craft_config
        )
        
        # 记录 metrics
        print(f"Retention Loss: {metrics['retention_loss']:.4f}")
        print(f"Student Norm: {metrics['student_hidden_norm']:.4f}")
        print(f"Target Norm: {metrics['target_features_norm']:.4f}")
        
        # 反向传播
        retention_loss.backward()
```

### 配置示例

```python
from lerobot.craft import CraftConfig

craft_config = CraftConfig(
    enabled=True,
    anchor_cache_dir="data/anchor_hidden_cache",
    hidden_layer=-2,  # 倒数第二层
    pooling="mean_image_tokens",  # 推荐
    loss_type="mse",  # 或 "cosine"
    retention_freq=1,  # 每步计算
)
```

---

## 📊 关键设计决策

### 1. 为什么使用原生 output_hidden_states？

**优点：**
- ✅ 最小侵入：不修改模型结构
- ✅ 兼容性好：大多数 Transformer 模型都支持
- ✅ 性能优化：模型内部已优化
- ✅ 易于维护：不依赖 forward hook

**实现：**
```python
# 优先使用原生 API
outputs = policy._paligemma_model(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,  # 关键参数
)
all_hidden_states = outputs.hidden_states
```

### 2. 为什么在 float32 中计算 loss？

**原因：**
- ✅ 数值稳定：避免 float16 的精度问题
- ✅ 梯度稳定：float32 梯度更准确
- ✅ 兼容性：PyTorch 优化器通常使用 float32

**实现：**
```python
# 转换到 float32
student_features_f32 = student_features.float()
target_features_f32 = target_features.float()

# 计算 loss（float32）
loss = F.mse_loss(student_features_f32, target_features_f32)
```

### 3. 为什么推荐 mean_image_tokens pooling？

**优点：**
- ✅ 语义丰富：图像 tokens 包含视觉信息
- ✅ 稳定性好：平均池化比单个 token 更鲁棒
- ✅ 任务相关：机器人任务主要依赖视觉输入
- ✅ 实验验证：在 vision-language 模型中表现良好

**对比：**
| Pooling | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| mean_image_tokens | 语义丰富、稳定 | 需要识别图像 tokens | 视觉任务（推荐） |
| mean_masked | 简单、通用 | 包含文本信息 | 通用场景 |
| last_token | 捕获序列信息 | 单点不稳定 | 序列任务 |
| cls_token | 全局表征 | 依赖模型设计 | 分类任务 |

### 4. 为什么支持 MSE 和 Cosine loss？

**MSE Loss（推荐）：**
```python
loss = F.mse_loss(student_features, target_features)
```
- ✅ 直接优化：最小化特征差异
- ✅ 稳定性好：梯度平滑
- ✅ 易于调试：loss 值直观

**Cosine Loss：**
```python
cosine_sim = F.cosine_similarity(student_features, target_features, dim=1)
loss = (1.0 - cosine_sim).mean()
```
- ✅ 方向对齐：关注特征方向而非幅度
- ✅ 归一化：对特征尺度不敏感
- ✅ 适用场景：当特征幅度变化大时

---

## 📝 Git Commit

```
commit: 4684ec00
message: feat: hidden-state retention loss implementation + tests

Files changed:
- src/lerobot/craft/retention_loss.py (修改)
- tests/test_hidden_retention_loss_math.py (新增)

Key features:
- Uses model native output_hidden_states=True (minimal intrusion)
- Extracts hidden states from configurable layer (default -2)
- Supports 4 pooling strategies: mean_image_tokens, mean_masked, last_token, cls_token
- Computes loss in float32 for numerical stability
- Supports MSE and cosine loss
- Full gradient flow verified

All tests passed (7/7)

Status: ✓ Committed, not pushed
```

---

## 🚀 下一步

1. **集成到训练循环**
   - 在 `lerobot_train_craft.py` 中调用 `compute_hidden_retention_loss()`
   - 替换现有的 `compute_retention_loss_hidden()`

2. **端到端测试**
   - 在真实数据集上测试
   - 验证梯度流和 loss 收敛

3. **性能对比**
   - 对比 MSE vs Cosine loss
   - 对比不同 pooling 策略
   - 对比不同 hidden_layer 选择

4. **文档更新**
   - 更新 CRaFT 训练指南
   - 添加 retention_mode=hidden 使用教程

