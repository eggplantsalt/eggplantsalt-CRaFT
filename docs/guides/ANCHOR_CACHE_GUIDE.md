# AnchorCache 离线生成和读取使用指南

## 📋 概述

AnchorCache 是 CRaFT 训练框架的核心组件，用于存储预生成的 teacher outputs，避免在线调用 teacher 模型的开销。

### 核心特性
- ✅ **离线生成**: Teacher 生成在训练前完成
- ✅ **确定性**: temperature=0 保证可复现
- ✅ **Token-level CE**: 正确的 labels mask 规则
- ✅ **内存高效**: 分 shard 存储，按需加载
- ✅ **自动探测**: 支持多种 dataset 图像 key 命名

---

## 🚀 快速开始

### 1. 生成 AnchorCache

```bash
# 基础用法
python src/lerobot/scripts/build_anchor_cache.py \
    --policy.pretrained_path=physical-intelligence/pi0-fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --out_dir=data/anchor_cache \
    --num_anchors=1000

# 自定义配置
python src/lerobot/scripts/build_anchor_cache.py \
    --policy.pretrained_path=physical-intelligence/pi0-fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --out_dir=data/anchor_cache \
    --num_anchors=1000 \
    --prompts_file=prompts.json \
    --max_new_tokens=256 \
    --shard_size=100 \
    --seed=42 \
    --device=cuda
```

### 2. 在训练中使用

```python
from lerobot.craft.anchor_cache import build_anchor_dataloader
from lerobot.datasets.utils import cycle

# 创建 DataLoader
anchor_dataloader = build_anchor_dataloader(
    cache_dir="data/anchor_cache",
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# 创建无限迭代器
anchor_dl_iter = cycle(anchor_dataloader)

# 在训练循环中使用
for step in range(total_steps):
    anchor_batch = next(anchor_dl_iter)
    # anchor_batch 包含: pixel_values, input_ids, attention_mask, labels
    retention_loss = compute_retention_loss(policy, anchor_batch)
```

---

## 📁 输出格式

### 目录结构
```
data/anchor_cache/
├── metadata.json          # 元数据
├── shard_0000.pt         # Shard 0
├── shard_0001.pt         # Shard 1
└── ...
```

### Shard 文件格式
每个 `.pt` 文件包含：
```python
{
    "pixel_values": Tensor[B, C, H, W],  # 图像，float32，[-1, 1]
    "input_ids": Tensor[B, seq_len],     # 完整输入序列
    "attention_mask": Tensor[B, seq_len], # 注意力掩码
    "labels": Tensor[B, seq_len],        # 标签（正确的 mask）
    "prompts": List[str],                # Prompt 字符串（调试用）
}
```

### Labels Mask 规则
```
序列结构: [Prompt tokens] [Teacher suffix tokens] [Padding]
Labels:   [-100 ...    ] [token_ids ...       ] [-100 ...]
          ↑              ↑                      ↑
          不计算损失      计算损失                不计算损失（EOS 后）
```

**规则详解**:
1. **Prompt tokens**: -100（不计算损失）
2. **Teacher suffix tokens**: 实际 token ids（计算损失）
3. **EOS 之后**: -100（不计算损失）
4. **Padding**: -100（不计算损失）

---

## 🎯 Prompts 配置

### 默认 Prompts
如果不提供 `--prompts_file`，使用内置默认：
```python
[
    "Pick up the object",
    "Place the object in the container",
    "Move to the target position",
    "Grasp the item",
    "Release the object",
]
```

### 自定义 Prompts
创建 `prompts.json`:
```json
{
    "prompts": [
        "Pick up the red block",
        "Place the block in the blue box",
        "Move to the target position",
        "Grasp the cup",
        "Release the object gently"
    ]
}
```

使用：
```bash
python src/lerobot/scripts/build_anchor_cache.py \
    --prompts_file=prompts.json \
    ...
```

---

## 🔍 图像 Key 自动探测

脚本会自动探测 dataset 中的图像 keys，支持多种命名方式：

### 支持的命名格式
1. `observation.images.{camera_name}` (LeRobot 标准)
2. `observation.image`
3. `pixels.{camera_name}` (LIBERO 格式)

### 探测优先级
1. **dataset.meta.camera_keys** (最可靠)
2. **features 中的 image/video 类型**
3. **从第一个样本推断** (fallback)

### 多相机支持
- 当前版本使用第一个探测到的相机
- 未来版本将支持多相机融合

---

## ⚙️ 参数说明

### 必需参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `--policy.pretrained_path` | Teacher 模型路径 | `physical-intelligence/pi0-fast` |
| `--dataset.repo_id` | LeRobot dataset | `lerobot/aloha_sim_insertion_human` |
| `--out_dir` | 输出目录 | `data/anchor_cache` |
| `--num_anchors` | Anchor 数量 | `1000` |

### 可选参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompts_file` | None | Prompts JSON 文件路径 |
| `--max_new_tokens` | 256 | Teacher 生成的最大 token 数 |
| `--shard_size` | 100 | 每个 shard 的样本数 |
| `--seed` | 42 | 随机种子 |
| `--device` | cuda | 设备 (cuda/cpu) |

---

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest tests/test_anchor_cache.py -v

# 运行特定测试
pytest tests/test_anchor_cache.py::test_labels_mask_rules -v

# 直接运行（不需要 pytest）
python tests/test_anchor_cache.py
```

### 测试覆盖
- ✅ Labels mask 规则验证
- ✅ 数据格式验证
- ✅ DataLoader 功能验证
- ✅ 跨 shard 访问验证
- ✅ Padding 不计算损失验证

---

## 📊 性能建议

### Shard 大小选择
- **小 shard (50-100)**: 更灵活，内存占用低
- **大 shard (200-500)**: 减少文件数，I/O 更高效
- **推荐**: 100-200 样本/shard

### DataLoader 配置
```python
anchor_dataloader = build_anchor_dataloader(
    cache_dir="data/anchor_cache",
    batch_size=16,        # 任务批次的 50%-100%
    num_workers=4,        # 4-8 个工作进程
    shuffle=True,         # 训练时打乱
    pin_memory=True,      # GPU 训练时启用
)
```

### 内存优化
- Shard 缓存机制：只加载当前需要的 shard
- 按需加载：不会一次性加载所有数据
- 多进程加载：利用 DataLoader 的 num_workers

---

## 🔧 故障排除

### 问题 1: 找不到图像 keys
**错误**: `ValueError: 无法探测到任何图像 keys`

**解决**:
1. 检查 dataset 是否包含图像数据
2. 手动指定图像 key（需修改脚本）
3. 验证 dataset 格式是否正确

### 问题 2: CUDA 内存不足
**错误**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减小 `--shard_size`
2. 使用 CPU: `--device=cpu`
3. 分批生成（多次运行，修改 `--num_anchors`）

### 问题 3: Teacher 生成速度慢
**解决**:
1. 使用 KV cache（pi0_fast 默认启用）
2. 减小 `--max_new_tokens`
3. 使用更快的 GPU

### 问题 4: Labels mask 不正确
**验证**:
```python
# 运行测试验证
pytest tests/test_anchor_cache.py::test_labels_mask_rules -v

# 手动检查
import torch
shard = torch.load("data/anchor_cache/shard_0000.pt")
labels = shard["labels"][0]
print(f"Prompt 部分 (-100): {labels[:10]}")
print(f"Suffix 部分 (token ids): {labels[10:20]}")
```

---

## 📝 完整示例

### 端到端工作流

```bash
# 1. 生成 AnchorCache
python src/lerobot/scripts/build_anchor_cache.py \
    --policy.pretrained_path=physical-intelligence/pi0-fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --out_dir=data/anchor_cache_aloha \
    --num_anchors=1000 \
    --shard_size=100 \
    --max_new_tokens=256 \
    --seed=42

# 2. 验证生成结果
python -c "
from lerobot.craft.anchor_cache import AnchorCacheDataset
dataset = AnchorCacheDataset('data/anchor_cache_aloha')
print(f'总样本数: {len(dataset)}')
sample = dataset[0]
print(f'样本 keys: {sample.keys()}')
print(f'图像形状: {sample[\"pixel_values\"].shape}')
print(f'序列长度: {sample[\"input_ids\"].shape}')
"

# 3. 运行测试
pytest tests/test_anchor_cache.py -v

# 4. 在训练中使用
python src/lerobot/scripts/lerobot_train_craft.py \
    --policy.type=pi0_fast \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --craft.anchor_cache_dir=data/anchor_cache_aloha \
    --batch_size=32 \
    --steps=10000 \
    --output_dir=outputs/craft_training
```

---

## 🔗 相关文档

- **build_anchor_cache.py**: 离线生成脚本
- **anchor_cache.py**: 数据加载模块
- **test_anchor_cache.py**: 单元测试
- **CRAFT_FILES.md**: CRaFT 文件组织指南

---

## 📞 技术支持

如有问题，请检查：
1. 日志输出（脚本会打印详细信息）
2. 测试结果（`pytest tests/test_anchor_cache.py -v`）
3. 元数据文件（`data/anchor_cache/metadata.json`）

**最后更新**: 2026-02-15

