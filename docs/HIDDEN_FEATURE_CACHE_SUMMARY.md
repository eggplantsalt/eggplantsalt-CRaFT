# Hidden Feature Cache 实现总结

## 概述

实现了离线 Hidden Feature Cache 系统，用于 CRaFT 训练的 `retention_mode=hidden`。与现有的 `build_anchor_cache.py` 不同，新系统保存单个 pooled feature vector，而非多层多向量，大幅减少存储空间。

---

## 修改文件列表

### 1. 新增文件

#### `src/lerobot/scripts/build_anchor_hidden_cache.py`
- **功能**: 离线生成 Hidden Feature Cache
- **输入参数**:
  - `--teacher_policy_path`: Teacher policy 路径（例如: physical-intelligence/pi0-fast）
  - `--dataset_repo_id`: Dataset repo ID（例如: lerobot/aloha_sim_insertion_human）
  - `--output_dir`: 输出目录
  - `--num_samples`: 采样数量
  - `--prompts_file`: Prompts 文件路径（可选，默认使用内置 prompts）
  - `--hidden_layer`: 要提取的层索引（默认 -2，倒数第二层）
  - `--pooling`: Pooling 策略（默认 mean_image_tokens）
  - `--dtype`: 数据类型（默认 float16）
  - `--shard_size`: 每个 shard 的大小（默认 100）

- **Pooling 策略**:
  1. `mean_image_tokens`: 对图像 tokens 取平均（推荐）
  2. `mean_masked`: 对所有非 padding tokens 取平均
  3. `last_token`: 取最后一个 token
  4. `cls_token`: 取第一个 token（假设是 CLS）

- **输出格式**:
```python
{
    "pixel_values": Tensor[B, C, H, W],      # 图像，float32，[-1, 1]
    "input_ids": Tensor[B, seq_len],         # 完整输入序列
    "attention_mask": Tensor[B, seq_len],    # 注意力掩码
    "target_features": Tensor[B, hidden_dim], # Pooled features (float16)
    "meta": {
        "hidden_layer": int,                  # 使用的层索引
        "pooling": str,                       # Pooling 策略
        "dtype": str,                         # 数据类型
    }
}
```

#### `tests/test_hidden_cache_format.py`
- **功能**: CPU 单元测试
- **测试用例**:
  1. `test_hidden_cache_dataset_loading`: 验证 dataset 加载和字段格式
  2. `test_hidden_cache_cross_shard_access`: 验证跨 shard 访问
  3. `test_hidden_cache_dataloader_collation`: 验证 batch collation
  4. `test_hidden_cache_backward_compatibility`: 验证三种 cache 类型兼容性
  5. `test_target_features_dtype_conversion`: 验证 dtype 转换

#### `tests/verify_hidden_cache.py`
- **功能**: 独立验证脚本（不依赖完整环境）
- **测试结果**: 全部通过 ✓

### 2. 修改文件

#### `src/lerobot/craft/anchor_cache.py`
- **更新**: 支持三种 cache 类型
  1. **Token-level cache**（旧版本）: 包含 `labels`
  2. **Hidden state cache**（多层多向量）: 包含 `teacher_hidden`
  3. **Hidden feature cache**（单个 pooled vector）: 包含 `target_features`

- **自动检测逻辑**:
```python
if "target_features" in shard_data:
    # Hidden feature cache（pooled vector）
    sample["target_features"] = shard_data["target_features"][local_idx]
    sample["meta"] = shard_data["meta"]
elif "teacher_hidden" in shard_data:
    # Hidden state cache（多层多向量）
    sample["teacher_hidden"] = shard_data["teacher_hidden"][local_idx]
    sample["meta"] = shard_data["meta"]
elif "labels" in shard_data:
    # Token-level cache（旧版本）
    sample["labels"] = shard_data["labels"][local_idx]
```

---

## Hidden Cache 字段说明

### 基础字段（所有类型都有）
- `pixel_values`: Tensor[C, H, W], 图像，float32，归一化到 [-1, 1]
- `input_ids`: Tensor[seq_len], 完整输入序列（prompt + BOS）
- `attention_mask`: Tensor[seq_len], 注意力掩码

### 类型特定字段

#### Token-level Cache
- `labels`: Tensor[seq_len], Teacher 生成的 tokens

#### Hidden State Cache
- `teacher_hidden`: Tensor[n_layers, n_vecs, hidden_dim], 多层多向量
- `meta`: dict, 包含 `layers_to_save`, `pooling` 等

#### Hidden Feature Cache（新增）
- `target_features`: Tensor[hidden_dim], 单个 pooled vector
- `meta`: dict, 包含 `hidden_layer`, `pooling`, `dtype`

---

## 如何识别图像 Tokens

### 方法 1: 从 Policy Config 获取
```python
if hasattr(policy.config, 'image_seq_length'):
    num_image_tokens = policy.config.image_seq_length
```

### 方法 2: 计算
```python
if hasattr(policy.config, 'image_resolution') and hasattr(policy.config, 'patch_size'):
    h, w = policy.config.image_resolution
    patch_size = policy.config.patch_size
    num_image_tokens = (h // patch_size) * (w // patch_size)
```

### 方法 3: 默认值
```python
# PaliGemma 224x224, patch_size=16
num_image_tokens = 196  # (224 // 16) ** 2
```

### 图像 Tokens 范围
对于 PaliGemma/Pi0 架构：
- 图像 tokens 在序列开头
- 范围: `[0, num_image_tokens)`
- 文本 tokens 在图像 tokens 之后

---

## 使用示例

### 生成 Hidden Feature Cache

```bash
python src/lerobot/scripts/build_anchor_hidden_cache.py \
    --teacher_policy_path=physical-intelligence/pi0-fast \
    --dataset_repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=data/anchor_hidden_cache \
    --num_samples=1000 \
    --hidden_layer=-2 \
    --pooling=mean_image_tokens \
    --dtype=float16
```

### 加载和使用

```python
from lerobot.craft.anchor_cache import build_anchor_dataloader
from lerobot.datasets.utils import cycle

# 创建 DataLoader
anchor_dataloader = build_anchor_dataloader(
    cache_dir="data/anchor_hidden_cache",
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# 创建无限迭代器
anchor_dl_iter = cycle(anchor_dataloader)

# 在训练循环中使用
for step in range(10000):
    anchor_batch = next(anchor_dl_iter)
    
    # anchor_batch 包含:
    # - pixel_values: [B, 3, 224, 224]
    # - input_ids: [B, seq_len]
    # - attention_mask: [B, seq_len]
    # - target_features: [B, hidden_dim]  # float16
    # - meta: dict
    
    # 计算 retention loss
    student_features = extract_student_features(model, anchor_batch)
    retention_loss = F.mse_loss(student_features, anchor_batch["target_features"])
```

---

## 与现有系统的对比

### build_anchor_cache.py（现有）
- **输出**: `teacher_hidden` [B, n_layers, n_vecs, hidden_dim]
- **存储**: 多层多向量，较大
- **用途**: 需要多层信息的场景

### build_anchor_hidden_cache.py（新增）
- **输出**: `target_features` [B, hidden_dim]
- **存储**: 单个 pooled vector，极小
- **用途**: 只需要单层表征的场景（推荐）

### 存储空间对比
假设：
- hidden_dim = 2048
- n_layers = 2
- n_vecs = 10
- dtype = float16 (2 bytes)

**现有系统**: 2 × 10 × 2048 × 2 = 81,920 bytes/sample
**新系统**: 2048 × 2 = 4,096 bytes/sample
**减少**: 95% ↓

---

## 验证结果

### 测试通过
```
============================================================
[SUCCESS] All tests passed!
============================================================

Test 1: Basic functionality
[OK] Dataset size: 5
[OK] target_features shape: torch.Size([128])
[OK] target_features dtype: torch.float16
[OK] meta: {'hidden_layer': -2, 'pooling': 'mean_image_tokens', 'dtype': 'float16'}

Test 2: Three cache types compatibility
[OK] token_level: labels shape = torch.Size([50])
[OK] hidden_state: teacher_hidden shape = torch.Size([2, 10, 128])
[OK] hidden_feature: target_features shape = torch.Size([128])

Test 3: DataLoader Batch Collation
[OK] Batch pixel_values shape: torch.Size([4, 3, 224, 224])
[OK] Batch input_ids shape: torch.Size([4, 50])
[OK] Batch target_features shape: torch.Size([4, 256])
[OK] Batch target_features dtype: torch.float16
```

---

## Git Commit

```
commit: 9782dbef
message: feat: add offline hidden anchor cache builder + loader

- Add: build_anchor_hidden_cache.py for pooled feature extraction
- Update: anchor_cache.py to support 3 cache types (token/hidden_state/hidden_feature)
- Add: test_hidden_cache_format.py with 5 unit tests
- Add: verify_hidden_cache.py for standalone validation

Key features:
- Pooled features: single vector [hidden_dim] instead of [n_layers, n_vecs, hidden_dim]
- 4 pooling strategies: mean_image_tokens, mean_masked, last_token, cls_token
- Backward compatible with existing token-level and hidden-state caches
- Configurable dtype (float16/float32/bfloat16) for memory efficiency
```

---

## 下一步

1. 在真实数据集上生成 hidden feature cache
2. 在 CRaFT 训练中集成 `retention_mode=hidden`
3. 对比三种 cache 类型的性能：
   - Token-level（旧版）
   - Hidden state（多层多向量）
   - Hidden feature（单个 pooled vector，新版）
4. 根据实验结果调优 pooling 策略和 hidden_layer 选择

