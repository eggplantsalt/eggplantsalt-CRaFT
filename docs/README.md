# 文档导航

> LeRobot + CRaFT 项目完整文档索引

---

## 🚀 快速开始

**新手必读**，5 分钟上手：

- [README](../README_CRAFT.md) - 项目概述和核心特性
- [快速开始指南](QUICKSTART.md) - 安装和基础使用
- [完整实验指南](EXPERIMENT_GUIDE.md) - 详细的实验操作步骤

---

## 📚 核心文档

### 用户文档

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [快速开始](QUICKSTART.md) | 5 分钟上手教程 | 所有用户 |
| [实验指南](EXPERIMENT_GUIDE.md) | 完整实验操作步骤 | 研究人员 |
| [CRaFT 训练指南](craft/CRAFT_TRAINING_GUIDE.md) | CRaFT 训练详解 | 高级用户 |
| [故障排查](TROUBLESHOOTING.md) | 常见问题解决 | 所有用户 |

### 技术文档

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [API 参考](API_REFERENCE.md) | 完整 API 文档 | 开发者 |
| [项目架构](OVERVIEW.md) | 技术架构详解 | 开发者 |
| [Hidden Feature Cache](HIDDEN_FEATURE_CACHE_SUMMARY.md) | Cache 生成和使用 | 研究人员 |
| [MCQ 评测工具](MCQ_LIKELIHOOD_EVAL.md) | 评测脚本使用 | 研究人员 |

### 开发文档

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [CRaFT 模块说明](../src/lerobot/craft/README.md) | 代码结构 | 开发者 |
| [贡献指南](../CONTRIBUTING.md) | 如何贡献代码 | 贡献者 |
| [测试指南](TESTING.md) | 测试编写和运行 | 开发者 |

---

## 📖 按主题浏览

### 1. 安装和配置

- [环境安装](QUICKSTART.md#步骤-1-安装)
- [依赖项说明](OVERVIEW.md#依赖项列表)
- [GPU 配置](TROUBLESHOOTING.md#cuda-配置)
- [数据集准备](EXPERIMENT_GUIDE.md#数据集准备)

### 2. 训练

- [Baseline 训练](EXPERIMENT_GUIDE.md#实验-1-baseline-训练)
- [CRaFT 训练](EXPERIMENT_GUIDE.md#实验-4-craft-训练hidden)
- [训练参数配置](OVERVIEW.md#训练参数配置)
- [训练监控](EXPERIMENT_GUIDE.md#监控训练)

### 3. CRaFT 核心

- [CRaFT 原理](craft/CRAFT_TRAINING_GUIDE.md#craft-原理)
- [梯度手术](API_REFERENCE.md#梯度手术模块)
- [原对偶优化](API_REFERENCE.md#原对偶优化)
- [Hidden State Anchoring](HIDDEN_FEATURE_CACHE_SUMMARY.md)

### 4. 数据处理

- [生成 Hidden Cache](EXPERIMENT_GUIDE.md#实验-2-生成-hidden-feature-cache)
- [数据集格式](OVERVIEW.md#数据格式)
- [数据增强](OVERVIEW.md#数据增强)

### 5. 评测

- [MCQ 评测](EXPERIMENT_GUIDE.md#实验-5-mcq-评测)
- [性能对比](EXPERIMENT_GUIDE.md#实验-6-对比分析)
- [指标说明](EXPERIMENT_GUIDE.md#关键指标说明)

### 6. 故障排查

- [CUDA 问题](TROUBLESHOOTING.md#cuda-问题)
- [内存不足](TROUBLESHOOTING.md#内存问题)
- [训练不收敛](TROUBLESHOOTING.md#训练问题)
- [数据加载错误](TROUBLESHOOTING.md#数据问题)

---

## 🎯 按角色浏览

### 研究人员

**目标**: 运行实验，评估 CRaFT 效果

1. [快速开始](QUICKSTART.md) - 了解基础
2. [实验指南](EXPERIMENT_GUIDE.md) - 完成所有实验
3. [CRaFT 训练指南](craft/CRAFT_TRAINING_GUIDE.md) - 深入理解
4. [结果分析](EXPERIMENT_GUIDE.md#实验-6-对比分析) - 解读结果

**推荐阅读顺序**: 1 → 2 → 3 → 4

### 开发者

**目标**: 理解代码，进行二次开发

1. [项目架构](OVERVIEW.md) - 了解整体结构
2. [API 参考](API_REFERENCE.md) - 掌握接口
3. [CRaFT 模块](../src/lerobot/craft/README.md) - 代码细节
4. [贡献指南](../CONTRIBUTING.md) - 开发规范

**推荐阅读顺序**: 1 → 2 → 3 → 4

### 新手用户

**目标**: 快速上手，运行第一个实验

1. [README](../README_CRAFT.md) - 项目概述
2. [快速开始](QUICKSTART.md) - 安装和运行
3. [故障排查](TROUBLESHOOTING.md) - 解决问题

**推荐阅读顺序**: 1 → 2 → 3

---

## 📂 文档结构

```
docs/
├── README.md                           # 本文件：文档导航
├── QUICKSTART.md                       # 快速开始指南
├── EXPERIMENT_GUIDE.md                 # 完整实验指南
├── API_REFERENCE.md                    # API 参考文档
├── OVERVIEW.md                         # 项目架构文档
├── TROUBLESHOOTING.md                  # 故障排查指南
├── TESTING.md                          # 测试指南
├── HIDDEN_FEATURE_CACHE_SUMMARY.md     # Hidden Cache 文档
├── MCQ_LIKELIHOOD_EVAL.md              # MCQ 评测文档
├── craft/                              # CRaFT 专题文档
│   └── CRAFT_TRAINING_GUIDE.md         # CRaFT 训练指南
└── CONTEXT.md                          # 项目上下文（用户自定义）
```

---

## 🔍 搜索文档

### 按关键词

- **安装**: [快速开始](QUICKSTART.md#步骤-1-安装)
- **训练**: [实验指南](EXPERIMENT_GUIDE.md)
- **CRaFT**: [CRaFT 训练指南](craft/CRAFT_TRAINING_GUIDE.md)
- **梯度手术**: [API 参考](API_REFERENCE.md#梯度手术模块)
- **Hidden Cache**: [Hidden Feature Cache](HIDDEN_FEATURE_CACHE_SUMMARY.md)
- **MCQ 评测**: [MCQ 评测工具](MCQ_LIKELIHOOD_EVAL.md)
- **故障排查**: [故障排查指南](TROUBLESHOOTING.md)
- **API**: [API 参考](API_REFERENCE.md)

### 按文件类型

- **配置文件**: [实验指南 - 配置](EXPERIMENT_GUIDE.md#准备配置文件)
- **脚本**: [API 参考 - 训练脚本](API_REFERENCE.md#训练脚本)
- **测试**: [测试指南](TESTING.md)
- **示例**: [实验指南](EXPERIMENT_GUIDE.md)

---

## 📝 文档更新日志

### 2026-02-17
- ✅ 重构所有文档
- ✅ 创建文档导航
- ✅ 添加完整实验指南
- ✅ 添加 API 参考文档
- ✅ 删除过时文档

### 2026-02-16
- ✅ 添加 MCQ 评测文档
- ✅ 添加 Hidden Cache 文档

### 2026-02-15
- ✅ 添加 CRaFT 训练指南
- ✅ 初始文档创建

---

## 🤝 贡献文档

发现文档问题或有改进建议？

1. 在 GitHub 提交 Issue
2. 提交 Pull Request
3. 联系维护者

详见 [贡献指南](../CONTRIBUTING.md)。

---

## 📧 获取帮助

- **GitHub Issues**: 报告 bug 或提问
- **Discord**: 加入社区讨论
- **Email**: 联系维护者

---

## 🔗 外部资源

- [LeRobot 官方文档](https://huggingface.co/docs/lerobot)
- [HuggingFace Hub](https://huggingface.co/lerobot)
- [PyTorch 文档](https://pytorch.org/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers)

---

**提示**: 使用 Ctrl+F 在页面内搜索关键词，快速定位所需文档。

**最后更新**: 2026-02-17

