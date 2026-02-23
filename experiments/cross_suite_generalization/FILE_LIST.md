# CRaFT 跨 Suite 泛化能力验证实验 - 文件清单

本实验包包含以下文件：

## 📄 文档文件

### 主文档
- **README.md** - 实验总览和快速开始指南
- **TECHNICAL_DETAILS.md** - 技术实现细节和算法原理
- **FAQ.md** - 常见问题解答
- **FILE_LIST.md** - 本文件，文件清单

## 🔧 脚本文件

### 主流程脚本
- **run_full_experiment.sh** - 一键运行完整实验流程

### 分步脚本（scripts/ 目录）
1. **00_verify_environment.py** - 环境验证脚本
2. **01_train_baseline.sh** - 训练 Baseline (Naive SFT)
3. **02_build_anchor_cache.sh** - 构建 Anchor Cache
4. **03_train_craft.sh** - 训练 CRaFT
5. **04_eval_cross_suite.sh** - 跨 Suite 评测
6. **05_generate_report.py** - 生成对比报告

### 辅助脚本
- **quick_test.py** - 快速测试脚本（10-15 分钟）
- **dataset_info.py** - 数据集信息查看

## ⚙️ 配置文件（configs/ 目录）

- **baseline_spatial.yaml** - Baseline 训练配置
- **craft_spatial.yaml** - CRaFT 训练配置

## 📂 输出目录（自动生成）

### outputs/
- **baseline_spatial/** - Baseline 训练输出
  - checkpoints/010000/ - 最终 checkpoint
  - logs/ - 训练日志
- **craft_spatial/** - CRaFT 训练输出
  - checkpoints/010000/ - 最终 checkpoint
  - logs/ - 训练日志
- **anchor_cache/** - Anchor Cache 数据
  - shard_*.pt - 数据分片
  - metadata.json - 元数据

### results/
- **baseline_spatial_on_spatial/** - Baseline 在 libero_spatial 上的评测结果
- **baseline_spatial_on_object/** - Baseline 在 libero_object 上的评测结果
- **baseline_spatial_on_goal/** - Baseline 在 libero_goal 上的评测结果
- **baseline_spatial_on_10/** - Baseline 在 libero_10 上的评测结果
- **craft_spatial_on_spatial/** - CRaFT 在 libero_spatial 上的评测结果
- **craft_spatial_on_object/** - CRaFT 在 libero_object 上的评测结果
- **craft_spatial_on_goal/** - CRaFT 在 libero_goal 上的评测结果
- **craft_spatial_on_10/** - CRaFT 在 libero_10 上的评测结果
- **comparison_report.md** - 对比报告
- **comparison_table.csv** - 对比表格
- **success_rate_comparison.png** - 可视化图表

## 📊 目录结构

```
cross_suite_generalization/
├── README.md                          # 主文档
├── TECHNICAL_DETAILS.md               # 技术细节
├── FAQ.md                             # 常见问题
├── FILE_LIST.md                       # 本文件
├── run_full_experiment.sh             # 一键运行脚本
│
├── scripts/                           # 脚本目录
│   ├── 00_verify_environment.py       # 环境验证
│   ├── 01_train_baseline.sh           # 训练 Baseline
│   ├── 02_build_anchor_cache.sh       # 构建 Cache
│   ├── 03_train_craft.sh              # 训练 CRaFT
│   ├── 04_eval_cross_suite.sh         # 跨 Suite 评测
│   ├── 05_generate_report.py          # 生成报告
│   ├── quick_test.py                  # 快速测试
│   └── dataset_info.py                # 数据集信息
│
├── configs/                           # 配置目录
│   ├── baseline_spatial.yaml          # Baseline 配置
│   └── craft_spatial.yaml             # CRaFT 配置
│
├── outputs/                           # 输出目录（自动生成）
│   ├── baseline_spatial/
│   ├── craft_spatial/
│   └── anchor_cache/
│
└── results/                           # 结果目录（自动生成）
    ├── baseline_spatial_on_*/
    ├── craft_spatial_on_*/
    ├── comparison_report.md
    ├── comparison_table.csv
    └── success_rate_comparison.png
```

## 🚀 使用流程

### 1. 环境验证
```bash
python experiments/cross_suite_generalization/scripts/00_verify_environment.py
```

### 2. 查看数据集信息
```bash
python experiments/cross_suite_generalization/scripts/dataset_info.py
```

### 3. 快速测试（可选）
```bash
python experiments/cross_suite_generalization/scripts/quick_test.py
```

### 4. 运行完整实验
```bash
bash experiments/cross_suite_generalization/run_full_experiment.sh
```

### 5. 查看结果
```bash
cat experiments/cross_suite_generalization/results/comparison_report.md
```

## 📝 文件说明

### 核心脚本说明

#### run_full_experiment.sh
- 自动执行完整实验流程
- 包含错误处理和进度显示
- 记录每个步骤的耗时
- 预计运行时间：6-9 小时

#### 01_train_baseline.sh
- 训练标准的 Naive SFT 模型
- 使用 libero_spatial 数据集
- 训练 10,000 步
- 输出 checkpoint 到 outputs/baseline_spatial/

#### 02_build_anchor_cache.sh
- 为 CRaFT 训练生成 Anchor Cache
- 从 pi0_fast 提取 hidden states
- 生成 1,000 个样本
- 输出到 outputs/anchor_cache/

#### 03_train_craft.sh
- 训练 CRaFT 模型
- 使用双目标优化（Task Loss + Retention Loss）
- 加载 Anchor Cache
- 输出 checkpoint 到 outputs/craft_spatial/

#### 04_eval_cross_suite.sh
- 在 4 个 Suites 上评测两个模型
- 每个 Suite 运行 50 episodes
- 生成 8 个评测结果文件
- 输出到 results/

#### 05_generate_report.py
- 汇总所有评测结果
- 生成 Markdown 报告
- 生成 CSV 表格
- 生成可视化图表

### 辅助脚本说明

#### 00_verify_environment.py
- 检查 Python 版本
- 检查必要的库（PyTorch, LIBERO, etc.）
- 检查 CUDA 可用性
- 检查目录结构和文件完整性

#### quick_test.py
- 使用少量数据快速测试流程
- 训练 100 步（而非 10,000 步）
- 使用 50 个 Anchor Cache 样本
- 仅评测 libero_spatial（5 episodes）
- 预计运行时间：10-15 分钟

#### dataset_info.py
- 显示 LIBERO 各个 Suite 的信息
- 检查数据集可用性
- 说明实验设计

### 配置文件说明

#### baseline_spatial.yaml
- 标准的 Pi0-fast 训练配置
- 无 CRaFT 约束
- 用于 Baseline 训练

#### craft_spatial.yaml
- CRaFT 训练配置
- 包含所有 CRaFT 超参数
- 指定 Anchor Cache 路径

## 🔍 关键文件内容

### Anchor Cache 格式

每个 shard_*.pt 文件包含：
```python
{
    "pixel_values": Tensor[B, C, H, W],      # 图像
    "input_ids": Tensor[B, seq_len],         # 输入序列
    "attention_mask": Tensor[B, seq_len],    # 注意力掩码
    "target_features": Tensor[B, hidden_dim], # Teacher hidden states
    "meta": {
        "hidden_layer": int,
        "pooling": str,
        "dtype": str,
    }
}
```

### 评测结果格式

每个 eval_info.json 文件包含：
```json
{
    "success_rate": 0.85,
    "avg_reward": 1.0,
    "avg_length": 150,
    "per_episode": [
        {"success": true, "reward": 1.0, "length": 145},
        {"success": false, "reward": 0.0, "length": 280},
        ...
    ]
}
```

## 📦 依赖关系

```
run_full_experiment.sh
├── 01_train_baseline.sh
│   └── lerobot_train.py
├── 02_build_anchor_cache.sh
│   └── build_anchor_hidden_cache.py
├── 03_train_craft.sh
│   ├── lerobot_train_craft.py
│   └── outputs/anchor_cache/ (依赖步骤 2)
├── 04_eval_cross_suite.sh
│   ├── lerobot_eval.py
│   ├── outputs/baseline_spatial/ (依赖步骤 1)
│   └── outputs/craft_spatial/ (依赖步骤 3)
└── 05_generate_report.py
    └── results/**/eval_info.json (依赖步骤 4)
```

## 💾 磁盘空间需求

| 项目 | 大小 | 说明 |
|------|------|------|
| Baseline Checkpoint | ~2 GB | 包含模型权重和优化器状态 |
| CRaFT Checkpoint | ~2 GB | 包含模型权重和优化器状态 |
| Anchor Cache | ~1 GB | 1,000 个样本的 hidden states |
| 评测视频（可选） | ~5 GB | 8 个评测任务 × 50 episodes |
| 训练日志 | ~100 MB | TensorBoard 日志和文本日志 |
| **总计** | **~10 GB** | 不包含数据集本身 |

## 🔄 更新日志

### v1.0 (2025-02-23)
- 初始版本
- 完整的实验流程
- 详细的文档和脚本

## 📞 支持

如有问题，请参考：
1. **FAQ.md** - 常见问题解答
2. **TECHNICAL_DETAILS.md** - 技术细节
3. **README.md** - 快速开始指南

---

**最后更新**: 2025-02-23

