# 故障排查指南

> 常见问题和解决方案

---

## 📋 目录

1. [安装问题](#安装问题)
2. [CUDA 和 GPU 问题](#cuda-和-gpu-问题)
3. [内存问题](#内存问题)
4. [训练问题](#训练问题)
5. [数据问题](#数据问题)
6. [CRaFT 特定问题](#craft-特定问题)
7. [性能问题](#性能问题)

---

## 安装问题

### 问题 1: pip install 失败

**症状**:
```
ERROR: Could not find a version that satisfies the requirement lerobot
```

**原因**: PyPI 上可能没有最新版本

**解决方案**:
```bash
# 从源码安装
git clone <your-repo-url>
cd lerobot
pip install -e .
```

### 问题 2: 依赖冲突

**症状**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**解决方案**:
```bash
# 创建新的虚拟环境
conda create -n lerobot_clean python=3.10
conda activate lerobot_clean

# 重新安装
pip install -e .
```

### 问题 3: 缺少系统依赖

**症状**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install -y mesa-libGL glib2
```

---

## CUDA 和 GPU 问题

### 问题 1: CUDA 不可用

**症状**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**诊断**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

**解决方案**:
```bash
# 安装对应 CUDA 版本的 PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 问题 2: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案 1: 减小 batch_size**
```bash
# 从 8 减到 4
--batch_size=4
```

**解决方案 2: 使用梯度累积**
```bash
# 等效 batch_size=8，但显存占用减半
--batch_size=4 --gradient_accumulation_steps=2
```

**解决方案 3: 使用混合精度**
```bash
--use_amp=true
```

**解决方案 4: 清理 GPU 缓存**
```python
import torch
torch.cuda.empty_cache()
```

### 问题 3: GPU 利用率低

**症状**: GPU 利用率 < 50%

**原因**: 数据加载瓶颈

**解决方案**:
```bash
# 增加数据加载线程
--num_workers=4

# 启用 pin_memory
--pin_memory=true

# 使用更快的数据格式（MP4 而非图像序列）
```

---

## 内存问题

### 问题 1: CPU 内存不足

**症状**:
```
MemoryError: Unable to allocate array
```

**解决方案 1: 减少数据集缓存**
```bash
# 不缓存整个数据集
--dataset.cache=false
```

**解决方案 2: 使用流式加载**
```bash
--dataset.streaming=true
```

**解决方案 3: 减少 num_workers**
```bash
--num_workers=2  # 从 4 减到 2
```

### 问题 2: 数据集下载占用大量空间

**症状**: 磁盘空间不足

**解决方案**:
```bash
# 设置缓存目录到大容量磁盘
export HF_HOME=/path/to/large/disk

# 或在代码中指定
--dataset.root=/path/to/large/disk/datasets
```

---

## 训练问题

### 问题 1: 训练不收敛

**症状**: 损失不下降或震荡

**诊断**:
```python
# 检查学习率
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# 检查梯度范数
print(f"Gradient norm: {grad_norm}")

# 检查数据统计
from lerobot.datasets import LeRobotDataset
dataset = LeRobotDataset("...")
print(dataset.stats)
```

**解决方案 1: 调整学习率**
```bash
# 学习率过大
--training.lr=5e-5  # 从 1e-4 降到 5e-5

# 学习率过小
--training.lr=3e-4  # 从 1e-4 升到 3e-4
```

**解决方案 2: 使用学习率调度器**
```bash
--training.lr_scheduler=cosine
--training.warmup_steps=1000
```

**解决方案 3: 检查数据归一化**
```python
# 确保数据已正确归一化
print(f"Mean: {dataset.stats['observation.state']['mean']}")
print(f"Std: {dataset.stats['observation.state']['std']}")
```

### 问题 2: 梯度爆炸

**症状**:
```
Step 100/10000 | loss=nan | grdn=inf
```

**解决方案 1: 启用梯度裁剪**
```bash
--training.grad_clip_norm=10
```

**解决方案 2: 降低学习率**
```bash
--training.lr=1e-5
```

**解决方案 3: 检查数据质量**
```python
# 检查是否有异常值
import torch
batch = next(iter(dataloader))
print(f"Max value: {batch['observation'].max()}")
print(f"Min value: {batch['observation'].min()}")
print(f"Has NaN: {torch.isnan(batch['observation']).any()}")
```

### 问题 3: 训练速度慢

**症状**: 每步耗时 > 1 秒

**诊断**:
```python
import time

# 测量数据加载时间
start = time.time()
batch = next(iter(dataloader))
print(f"Data loading: {time.time() - start:.3f}s")

# 测量前向传播时间
start = time.time()
output = model(batch)
print(f"Forward pass: {time.time() - start:.3f}s")

# 测量反向传播时间
start = time.time()
loss.backward()
print(f"Backward pass: {time.time() - start:.3f}s")
```

**解决方案**: 见 [性能问题](#性能问题)

---

## 数据问题

### 问题 1: 数据集下载失败

**症状**:
```
ConnectionError: Failed to download dataset from HuggingFace Hub
```

**解决方案 1: 使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**解决方案 2: 手动下载**
```bash
# 从 HuggingFace Hub 手动下载
# 然后指定本地路径
--dataset.root=/path/to/local/dataset
```

**解决方案 3: 使用代理**
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### 问题 2: 数据集格式错误

**症状**:
```
KeyError: 'observation.images.top'
```

**诊断**:
```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset("...")
print(f"Available keys: {dataset[0].keys()}")
print(f"Features: {dataset.features}")
```

**解决方案**: 检查数据集是否为 LeRobotDataset v3 格式

### 问题 3: 视频解码失败

**症状**:
```
RuntimeError: Failed to decode video frame
```

**解决方案 1: 重新编码视频**
```bash
# 使用 ffmpeg 重新编码
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 18 output.mp4
```

**解决方案 2: 使用图像格式**
```bash
# 转换为图像序列
--dataset.image_format=png
```

---

## CRaFT 特定问题

### 问题 1: AnchorCache 加载失败

**症状**:
```
FileNotFoundError: AnchorCache directory not found: data/anchor_hidden_cache
```

**诊断**:
```bash
# 检查目录是否存在
ls -lh data/anchor_hidden_cache/

# 检查文件
ls -lh data/anchor_hidden_cache/*.pt
ls -lh data/anchor_hidden_cache/metadata.json
```

**解决方案**: 重新生成 cache
```bash
python -m lerobot.scripts.build_anchor_hidden_cache \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=data/anchor_hidden_cache \
    --num_samples=1000
```

### 问题 2: Retention mode 不匹配

**症状**:
```
ValueError: retention_mode=hidden 需要 hidden feature cache，但 anchor_batch 不包含 'target_features'
```

**原因**: Cache 类型与 retention_mode 不匹配

**解决方案**:
```bash
# 如果使用 hidden mode，需要 hidden feature cache
python -m lerobot.scripts.build_anchor_hidden_cache ...

# 如果使用 token_ce mode，需要 token-level cache
python -m lerobot.scripts.build_anchor_cache ...
```

### 问题 3: 梯度冲突过多

**症状**: 日志中 `conflict=✓` 出现频率 > 50%

**诊断**:
```bash
# 检查梯度点积和余弦相似度
# 如果 dot 和 cos 经常为负，说明冲突严重
```

**解决方案 1: 调整冲突阈值**
```bash
# 放宽阈值
craft.conflict_threshold=-0.2  # 从 -0.1 改为 -0.2
```

**解决方案 2: 调整 λ**
```bash
# 降低保留损失权重
craft.initial_lambda=0.5  # 从 1.0 降到 0.5
```

**解决方案 3: 调整 ε**
```bash
# 放宽保留约束
craft.epsilon_start=1.5  # 从 1.0 升到 1.5
```

### 问题 4: λ 增长过快

**症状**: λ 快速达到 λ_max

**原因**: 保留损失持续违反约束

**解决方案 1: 降低 λ 学习率**
```bash
craft.lambda_lr=0.005  # 从 0.01 降到 0.005
```

**解决方案 2: 增大 λ_max**
```bash
craft.lambda_max=20.0  # 从 10.0 升到 20.0
```

**解决方案 3: 调整 ε 调度**
```bash
# 更慢的退火
craft.epsilon_decay_steps=20000  # 从 10000 升到 20000
```

---

## 性能问题

### 问题 1: 数据加载慢

**症状**: `data_s` > 0.5 秒

**解决方案**:
```bash
# 增加 num_workers
--num_workers=4

# 启用 prefetch
--prefetch_factor=2

# 使用 SSD 存储数据集
```

### 问题 2: 前向传播慢

**症状**: `updt_s` > 1.0 秒

**解决方案 1: 使用混合精度**
```bash
--use_amp=true
```

**解决方案 2: 使用 TorchScript**
```python
policy_scripted = torch.jit.script(policy)
```

**解决方案 3: 使用更小的模型**
```bash
# 使用更小的 hidden_dim
--policy.dim_model=256  # 从 512 降到 256
```

### 问题 3: 保存 checkpoint 慢

**症状**: 保存 checkpoint 耗时 > 30 秒

**解决方案 1: 减少保存频率**
```bash
--training.save_freq=5000  # 从 1000 升到 5000
```

**解决方案 2: 使用更快的存储**
```bash
# 保存到 SSD
--output_dir=/path/to/ssd/outputs
```

**解决方案 3: 异步保存**
```python
# 在后台线程保存
import threading

def save_checkpoint_async(checkpoint, path):
    thread = threading.Thread(target=torch.save, args=(checkpoint, path))
    thread.start()
```

---

## 调试技巧

### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 使用 pdb 调试

```python
import pdb

# 在代码中插入断点
pdb.set_trace()
```

### 3. 使用 Rerun 可视化

```python
import rerun as rr

rr.init("debug_session", spawn=True)
rr.log("observation/image", rr.Image(image))
rr.log("action", rr.Scalar(action_value))
```

### 4. 检查张量统计

```python
def check_tensor(tensor, name="tensor"):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")
```

### 5. 性能分析

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # 运行代码
    output = model(batch)
    loss = output['loss']
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 获取帮助

如果以上方法都无法解决问题：

1. **查看日志**: 完整的错误堆栈信息
2. **最小复现**: 创建最小可复现示例
3. **提交 Issue**: 在 GitHub 提交详细的 bug 报告
4. **社区讨论**: 在 Discord 或论坛寻求帮助

### Issue 模板

```markdown
**问题描述**
简要描述问题

**复现步骤**
1. 运行命令 `...`
2. 观察到错误 `...`

**预期行为**
应该发生什么

**实际行为**
实际发生了什么

**环境信息**
- OS: Ubuntu 20.04
- Python: 3.10.12
- PyTorch: 2.2.1
- CUDA: 11.8
- GPU: RTX 3090

**错误日志**
```
完整的错误堆栈
```

**已尝试的解决方案**
- 尝试了 X，结果 Y
- 尝试了 Z，结果 W
```

---

## 常见错误代码

| 错误代码 | 说明 | 解决方案 |
|----------|------|----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | GPU 内存不足 | 减小 batch_size |
| `RuntimeError: CUDA error: device-side assert triggered` | CUDA 断言失败 | 检查索引越界 |
| `KeyError: 'observation.images.top'` | 数据集键不存在 | 检查数据集格式 |
| `FileNotFoundError` | 文件不存在 | 检查路径 |
| `ConnectionError` | 网络连接失败 | 使用镜像或代理 |
| `ValueError: retention_mode` | 配置错误 | 检查 retention_mode |

---

**最后更新**: 2026-02-17

**提示**: 如果遇到新问题，欢迎提交 PR 更新本文档！

