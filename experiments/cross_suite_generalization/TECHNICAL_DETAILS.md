# CRaFT 跨 Suite 泛化能力验证实验 - 技术实现细节

> 本文档详细说明实验的技术实现细节，包括数据流、算法原理、代码结构等。

---

## 目录

1. [实验架构](#实验架构)
2. [数据流设计](#数据流设计)
3. [CRaFT 算法实现](#craft-算法实现)
4. [评测指标计算](#评测指标计算)
5. [关键代码解析](#关键代码解析)
6. [超参数调优指南](#超参数调优指南)

---

## 实验架构

### 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    实验流程总览                              │
└─────────────────────────────────────────────────────────────┘

Step 1: Baseline 训练
  ├─ 输入: libero_spatial 数据集
  ├─ 模型: pi0_fast (预训练)
  ├─ 方法: 标准行为克隆 (Naive SFT)
  └─ 输出: baseline_spatial/checkpoints/010000/

Step 2: Anchor Cache 构建
  ├─ 输入: libero_spatial 数据集 + pi0_fast (teacher)
  ├─ 方法: 提取 hidden states 并 pooling
  └─ 输出: anchor_cache/shard_*.pt + metadata.json

Step 3: CRaFT 训练
  ├─ 输入: libero_spatial 数据集 + anchor_cache
  ├─ 模型: pi0_fast (预训练)
  ├─ 方法: 双目标优化 (Task Loss + Retention Loss)
  └─ 输出: craft_spatial/checkpoints/010000/

Step 4: 跨 Suite 评测
  ├─ 模型: baseline_spatial + craft_spatial
  ├─ 环境: libero_spatial, libero_object, libero_goal, libero_10
  └─ 输出: eval_info.json (每个组合)

Step 5: 结果分析
  ├─ 输入: 所有 eval_info.json
  └─ 输出: comparison_report.md + 可视化图表
```

---

## 数据流设计

### Baseline 训练数据流

```python
# 伪代码
dataset = load_dataset("lerobot/libero_spatial_no_noops")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # 标准监督学习
    images = batch["observation.images.image"]  # [B, C, H, W]
    actions = batch["action"]                    # [B, action_dim]
    
    # 前向传播
    predicted_actions = policy(images)
    
    # 计算损失
    loss = F.mse_loss(predicted_actions, actions)
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### CRaFT 训练数据流

```python
# 伪代码
task_dataset = load_dataset("lerobot/libero_spatial_no_noops")
anchor_cache = load_anchor_cache("anchor_cache/")

task_dataloader = DataLoader(task_dataset, batch_size=32)
anchor_dataloader = DataLoader(anchor_cache, batch_size=16)

for task_batch, anchor_batch in zip(task_dataloader, anchor_dataloader):
    # ============================================================
    # Phase 1: 任务损失（标准训练）
    # ============================================================
    task_images = task_batch["observation.images.image"]
    task_actions = task_batch["action"]
    
    task_predicted = policy(task_images)
    task_loss = F.mse_loss(task_predicted, task_actions)
    
    # 反向传播并保存梯度
    task_loss.backward()
    task_grads = [p.grad.clone() for p in policy.parameters()]
    optimizer.zero_grad()
    
    # ============================================================
    # Phase 2: 保留损失（Hidden State Anchoring）
    # ============================================================
    anchor_images = anchor_batch["pixel_values"]
    anchor_target_features = anchor_batch["target_features"]  # Teacher hidden states
    
    # 提取 student hidden states
    student_hidden = policy.extract_hidden_states(anchor_images, layer=-2)
    student_pooled = pool_hidden_states(student_hidden, mode="mean_image_tokens")
    
    # 计算保留损失
    retention_loss = F.mse_loss(student_pooled, anchor_target_features)
    
    # 反向传播并保存梯度
    retention_loss.backward()
    retention_grads = [p.grad.clone() for p in policy.parameters()]
    optimizer.zero_grad()
    
    # ============================================================
    # Phase 3: 梯度手术（Gradient Surgery）
    # ============================================================
    if use_grad_projection:
        # 检测梯度冲突
        dot_product = compute_dot(task_grads, retention_grads)
        
        if dot_product < conflict_threshold:  # 冲突
            # 投影 retention_grads 到 task_grads 的正交空间
            retention_grads = project_if_conflict(task_grads, retention_grads)
    
    # ============================================================
    # Phase 4: 合并梯度并更新
    # ============================================================
    final_grads = merge_grads(task_grads, retention_grads, lambda_weight)
    
    # 应用最终梯度
    for p, g in zip(policy.parameters(), final_grads):
        p.grad = g
    
    optimizer.step()
    
    # ============================================================
    # Phase 5: 更新 Lambda（原对偶优化）
    # ============================================================
    lambda_weight = update_lambda(
        lambda_weight, 
        retention_loss.item(), 
        epsilon_threshold, 
        lambda_lr
    )
```

---

## CRaFT 算法实现

### 核心组件

#### 1. Hidden State Extraction

```python
def extract_student_hidden_with_pooling(
    policy: PreTrainedPolicy,
    batch: dict,
    hidden_layer: int = -2,
    pooling: str = "mean_image_tokens"
) -> torch.Tensor:
    """
    提取 student 模型的 hidden states 并进行 pooling
    
    Args:
        policy: 策略模型
        batch: 输入批次
        hidden_layer: 提取的层索引（-2 表示倒数第二层）
        pooling: Pooling 策略
    
    Returns:
        pooled_features: [B, hidden_dim]
    """
    # 前向传播并提取 hidden states
    with torch.no_grad():
        outputs = policy.forward_with_hidden_states(batch)
    
    hidden_states = outputs.hidden_states[hidden_layer]  # [B, seq_len, hidden_dim]
    
    # Pooling
    if pooling == "mean_image_tokens":
        # 只对图像 tokens 取平均（排除文本 tokens）
        image_mask = get_image_token_mask(batch)
        pooled = (hidden_states * image_mask.unsqueeze(-1)).sum(dim=1) / image_mask.sum(dim=1, keepdim=True)
    elif pooling == "mean_masked":
        # 对所有非 padding tokens 取平均
        attention_mask = batch["attention_mask"]
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    elif pooling == "last_token":
        # 取最后一个 token
        pooled = hidden_states[:, -1, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    
    return pooled  # [B, hidden_dim]
```

#### 2. Retention Loss Computation

```python
def compute_retention_loss_hidden(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    计算 hidden state retention loss
    
    Args:
        student_features: [B, hidden_dim]
        teacher_features: [B, hidden_dim]
        loss_type: "mse" 或 "cosine"
    
    Returns:
        loss: scalar
    """
    if loss_type == "mse":
        # L2 距离
        loss = F.mse_loss(student_features, teacher_features)
    elif loss_type == "cosine":
        # Cosine 距离
        cos_sim = F.cosine_similarity(student_features, teacher_features, dim=-1)
        loss = 1.0 - cos_sim.mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss
```

#### 3. Gradient Surgery

```python
def compute_dot(grads1: List[torch.Tensor], grads2: List[torch.Tensor]) -> float:
    """
    计算两组梯度的点积（归一化）
    
    Returns:
        dot_product: 范围 [-1, 1]，类似 cosine similarity
    """
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            dot += (g1 * g2).sum().item()
            norm1 += (g1 * g1).sum().item()
            norm2 += (g2 * g2).sum().item()
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def project_if_conflict(
    grads1: List[torch.Tensor],
    grads2: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    如果梯度冲突，将 grads2 投影到 grads1 的正交空间
    
    投影公式:
        grads2_proj = grads2 - (grads2 · grads1 / ||grads1||^2) * grads1
    
    Returns:
        projected_grads2: 投影后的梯度
    """
    dot = 0.0
    norm1_sq = 0.0
    
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            dot += (g1 * g2).sum().item()
            norm1_sq += (g1 * g1).sum().item()
    
    if norm1_sq == 0:
        return grads2
    
    # 投影系数
    proj_coef = dot / norm1_sq
    
    # 投影
    projected = []
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            g2_proj = g2 - proj_coef * g1
            projected.append(g2_proj)
        else:
            projected.append(g2)
    
    return projected
```

#### 4. Primal-Dual Optimization

```python
def update_lambda(
    current_lambda: float,
    retention_loss: float,
    epsilon: float,
    lambda_lr: float,
    lambda_max: float = 10.0
) -> float:
    """
    更新 Lagrangian 乘子 λ
    
    更新规则:
        λ ← λ + λ_lr * (L_retain - ε)
        λ ← clip(λ, 0, λ_max)
    
    直觉:
        - 如果 L_retain > ε（违反约束），增大 λ，加大保留损失权重
        - 如果 L_retain < ε（满足约束），减小 λ，减小保留损失权重
    """
    violation = retention_loss - epsilon
    new_lambda = current_lambda + lambda_lr * violation
    new_lambda = max(0.0, min(new_lambda, lambda_max))
    
    return new_lambda
```

---

## 评测指标计算

### 成功率计算

```python
def compute_success_rate(eval_results: List[Dict]) -> float:
    """
    从评测结果中计算成功率
    
    Args:
        eval_results: 每个 episode 的结果列表
            [
                {"success": True, "reward": 1.0, ...},
                {"success": False, "reward": 0.0, ...},
                ...
            ]
    
    Returns:
        success_rate: 范围 [0, 1]
    """
    if not eval_results:
        return 0.0
    
    successes = sum(1 for ep in eval_results if ep.get("success", False))
    return successes / len(eval_results)
```

### 跨 Suite 泛化指标

```python
def compute_ood_generalization_score(
    id_baseline: float,
    id_craft: float,
    ood_baseline: List[float],
    ood_craft: List[float]
) -> Dict[str, float]:
    """
    计算 OOD 泛化指标
    
    Returns:
        {
            "id_preservation": float,  # ID 性能保持度
            "ood_improvement": float,  # OOD 平均改进
            "generalization_score": float,  # 综合泛化分数
        }
    """
    # ID 性能保持度（越接近 1 越好）
    id_preservation = id_craft / max(id_baseline, 1e-6)
    
    # OOD 平均改进（绝对值）
    ood_improvement = np.mean([c - b for c, b in zip(ood_craft, ood_baseline)])
    
    # 综合泛化分数（加权）
    generalization_score = 0.3 * id_preservation + 0.7 * (ood_improvement / 100)
    
    return {
        "id_preservation": id_preservation,
        "ood_improvement": ood_improvement,
        "generalization_score": generalization_score,
    }
```

---

## 关键代码解析

### 训练循环核心逻辑

```python
# src/lerobot/scripts/lerobot_train_craft.py

def train_craft(cfg: TrainPipelineConfig):
    # 初始化
    policy = make_policy(cfg.policy)
    task_dataset = make_dataset(cfg.dataset)
    anchor_cache = load_anchor_cache(cfg.craft.anchor_cache_dir)
    
    optimizer = make_optimizer(policy.parameters(), cfg.optimizer)
    
    # CRaFT 状态
    current_lambda = cfg.craft.initial_lambda
    current_epsilon = cfg.craft.epsilon_start
    
    # 训练循环
    for step in range(cfg.steps):
        # 退火 epsilon
        current_epsilon = anneal_epsilon(
            step, 
            cfg.craft.epsilon_start, 
            cfg.craft.epsilon_end, 
            cfg.craft.epsilon_anneal_steps
        )
        
        # 获取批次
        task_batch = next(task_dataloader)
        
        # 每 K 步计算一次保留损失
        if step % cfg.craft.retention_freq == 0:
            anchor_batch = next(anchor_dataloader)
        else:
            anchor_batch = None
        
        # 执行 CRaFT 更新
        metrics, output_dict, current_lambda = update_policy_craft(
            train_metrics=metrics,
            policy=policy,
            task_batch=task_batch,
            anchor_batch=anchor_batch,
            optimizer=optimizer,
            grad_clip_norm=cfg.grad_clip_norm,
            accelerator=accelerator,
            craft_config=cfg.craft,
            current_lambda=current_lambda,
            current_epsilon=current_epsilon,
        )
        
        # 日志记录
        if step % cfg.log_freq == 0:
            log_metrics(metrics, step)
        
        # 评估
        if step % cfg.eval_freq == 0:
            eval_metrics = eval_policy(policy, eval_env)
            log_metrics(eval_metrics, step)
        
        # 保存 checkpoint
        if step % cfg.save_freq == 0:
            save_checkpoint(policy, optimizer, step, cfg.output_dir)
```

### 评测循环核心逻辑

```python
# src/lerobot/scripts/lerobot_eval.py

def eval_policy_all(
    policy: PreTrainedPolicy,
    env: gym.Env,
    n_episodes: int,
    batch_size: int
) -> Dict[str, Any]:
    """
    评测策略在环境中的性能
    """
    policy.eval()
    
    results = []
    
    for episode_idx in range(n_episodes):
        # 重置环境
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # 策略推理
            with torch.no_grad():
                action = policy.select_action(obs)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        # 记录结果
        results.append({
            "success": info.get("success", False),
            "reward": episode_reward,
            "length": episode_length,
        })
    
    # 汇总统计
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_length = np.mean([r["length"] for r in results])
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "per_episode": results,
    }
```

---

## 超参数调优指南

### 关键超参数

| 参数 | 默认值 | 调优范围 | 影响 |
|------|--------|---------|------|
| `initial_lambda` | 1.0 | [0.1, 5.0] | 初始保留损失权重 |
| `lambda_lr` | 0.001 | [0.0001, 0.01] | λ 更新速度 |
| `epsilon_start` | 1.0 | [0.5, 2.0] | 初始约束宽松度 |
| `epsilon_end` | 0.05 | [0.01, 0.2] | 最终约束严格度 |
| `retention_freq` | 5 | [1, 10] | 保留损失计算频率 |
| `conflict_threshold` | -0.1 | [-0.3, 0.0] | 梯度冲突检测阈值 |

### 调优策略

#### 场景 1: Retention Loss 过高

**症状**: `retention_loss` 持续 >> `epsilon`，`lambda` 持续增大

**调优方案**:
1. 增大 `epsilon_start` 和 `epsilon_end`（放宽约束）
2. 降低 `lambda_lr`（减缓 λ 增长）
3. 增加 `retention_freq`（减少保留损失计算频率）

#### 场景 2: ID 性能下降

**症状**: CRaFT 在 `libero_spatial` 上的成功率 < Baseline

**调优方案**:
1. 降低 `initial_lambda`（减小保留损失权重）
2. 增大 `epsilon_end`（放宽最终约束）
3. 禁用梯度投影（`use_grad_projection=false`）

#### 场景 3: OOD 泛化不足

**症状**: CRaFT 在 OOD Suites 上的改进 < 5%

**调优方案**:
1. 增大 `initial_lambda`（增大保留损失权重）
2. 减小 `epsilon_end`（收紧最终约束）
3. 增加 Anchor Cache 样本数（`num_samples=2000`）
4. 增加训练步数（`steps=20000`）

---

## 实验复现清单

### 环境配置

- [ ] Python 3.8+
- [ ] PyTorch 1.13+
- [ ] CUDA 11.7+ (推荐)
- [ ] lerobot (本仓库)
- [ ] libero-robotics
- [ ] matplotlib (用于可视化)

### 数据准备

- [ ] libero_spatial_no_noops 数据集
- [ ] libero_object_no_noops 数据集
- [ ] libero_goal_no_noops 数据集
- [ ] libero_10_no_noops 数据集

### 训练配置

- [ ] Baseline 配置文件
- [ ] CRaFT 配置文件
- [ ] Anchor Cache 生成脚本

### 评测配置

- [ ] 评测脚本（4 个 Suites）
- [ ] 结果分析脚本

---

## 参考文献

1. **CRaFT 原理**: Constrained Retention Fine-Tuning for Continual Learning
2. **Gradient Surgery**: Gradient Surgery for Multi-Task Learning (Yu et al., NeurIPS 2020)
3. **Primal-Dual Optimization**: Constrained Policy Optimization (Achiam et al., ICML 2017)
4. **LIBERO Benchmark**: LIBERO: Benchmarking Knowledge Transfer in Lifelong Robot Learning

---

**最后更新**: 2025-02-23

