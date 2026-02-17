# API 参考文档

> CRaFT 模块完整 API 文档

---

## 📋 目录

1. [CraftConfig](#craftconfig)
2. [梯度手术模块](#梯度手术模块)
3. [原对偶优化](#原对偶优化)
4. [保留损失](#保留损失)
5. [锚点数据加载](#锚点数据加载)
6. [训练脚本](#训练脚本)

---

## CraftConfig

### 类定义

```python
from lerobot.craft import CraftConfig

@dataclass
class CraftConfig:
    """CRaFT 训练配置类"""
    
    # 启用/禁用
    enabled: bool = False
    
    # 锚点数据集配置
    anchor_cache_dir: str = ""
    anchor_batch_size: int = 16
    retention_freq: int = 5
    retention_mode: str = "hidden"  # "token_ce" 或 "hidden"
    
    # 损失权重（原对偶优化）
    initial_lambda: float = 1.0
    lambda_lr: float = 0.01
    lambda_max: float = 10.0
    
    # 保留约束（ε 调度）
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 0  # 0 表示使用 training steps
    
    # 梯度手术
    use_grad_projection: bool = True
    conflict_threshold: float = -0.1
    projection_mode: str = "weighted"  # "weighted", "equal", "task_priority"
    
    # 日志记录
    log_craft_metrics_freq: int = 100
    save_lambda_history: bool = True
```

### 参数说明

#### enabled
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 是否启用 CRaFT 训练
- **示例**:
  ```python
  config = CraftConfig(enabled=True)
  ```

#### anchor_cache_dir
- **类型**: `str`
- **默认值**: `""`
- **说明**: AnchorCache 目录路径
- **要求**: 如果 `enabled=True`，必须提供
- **示例**:
  ```python
  config = CraftConfig(
      enabled=True,
      anchor_cache_dir="data/anchor_hidden_cache"
  )
  ```

#### retention_mode
- **类型**: `str`
- **默认值**: `"hidden"`
- **可选值**: `"token_ce"`, `"hidden"`
- **说明**: 保留损失计算模式
  - `"token_ce"`: Token-level cross-entropy loss
  - `"hidden"`: Hidden state retention loss（推荐）
- **示例**:
  ```python
  config = CraftConfig(retention_mode="hidden")
  ```

#### initial_lambda
- **类型**: `float`
- **默认值**: `1.0`
- **范围**: `[0.0, inf)`
- **说明**: Lagrangian 乘子 λ 的初始值
- **建议**: 从 1.0 开始，根据实验调整
- **示例**:
  ```python
  config = CraftConfig(initial_lambda=2.0)
  ```

#### epsilon_start / epsilon_end
- **类型**: `float`
- **默认值**: `1.0` / `0.1`
- **说明**: 保留损失阈值的起始/结束值
- **退火**: 从 `epsilon_start` 线性退火到 `epsilon_end`
- **示例**:
  ```python
  config = CraftConfig(
      epsilon_start=1.5,
      epsilon_end=0.05
  )
  ```

#### use_grad_projection
- **类型**: `bool`
- **默认值**: `True`
- **说明**: 是否启用梯度投影（解决梯度冲突）
- **算法**: 基于 PCGrad
- **示例**:
  ```python
  config = CraftConfig(use_grad_projection=True)
  ```

---

## 梯度手术模块

### compute_dot

计算两个梯度向量的点积。

```python
from lerobot.craft.grad_surgery import compute_dot

def compute_dot(
    grads_a: list[torch.Tensor | None],
    grads_b: list[torch.Tensor | None]
) -> torch.Tensor:
    """
    计算两个梯度向量的点积
    
    参数:
        grads_a: 第一个梯度列表
        grads_b: 第二个梯度列表
    
    返回:
        dot_product: 点积标量张量
    
    示例:
        >>> task_grads = [p.grad for p in model.parameters()]
        >>> ret_grads = [p.grad for p in model.parameters()]
        >>> dot = compute_dot(task_grads, ret_grads)
        >>> print(f"Dot product: {dot.item()}")
    """
```

**数学定义**:
```
dot = Σ (g_a[i] · g_b[i])
```

### project_if_conflict

检测梯度冲突并进行投影。

```python
from lerobot.craft.grad_surgery import project_if_conflict

def project_if_conflict(
    grads_task: list[torch.Tensor | None],
    grads_retain: list[torch.Tensor | None],
    conflict_threshold: float = -0.1
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], bool]:
    """
    检测梯度冲突并进行投影
    
    参数:
        grads_task: 任务梯度
        grads_retain: 保留梯度
        conflict_threshold: 冲突阈值（余弦相似度）
    
    返回:
        grads_task_proj: 投影后的任务梯度
        grads_retain_proj: 投影后的保留梯度
        conflict_detected: 是否检测到冲突
    
    示例:
        >>> task_proj, ret_proj, conflict = project_if_conflict(
        ...     task_grads, ret_grads, conflict_threshold=-0.1
        ... )
        >>> if conflict:
        ...     print("Gradient conflict detected and resolved!")
    """
```

**算法**:
```
如果 cos(g_task, g_retain) < threshold:
    g_task_proj = g_task - (g_task · g_retain / ||g_retain||²) * g_retain
    g_retain_proj = g_retain - (g_retain · g_task / ||g_task||²) * g_task
否则:
    g_task_proj = g_task
    g_retain_proj = g_retain
```

### merge_grads

合并任务梯度和保留梯度。

```python
from lerobot.craft.grad_surgery import merge_grads

def merge_grads(
    grads_task: list[torch.Tensor | None],
    grads_retain: list[torch.Tensor | None],
    lambda_weight: float,
    mode: str = "weighted"
) -> list[torch.Tensor | None]:
    """
    合并任务梯度和保留梯度
    
    参数:
        grads_task: 任务梯度
        grads_retain: 保留梯度
        lambda_weight: λ 权重
        mode: 合并模式
    
    返回:
        merged_grads: 合并后的梯度
    
    示例:
        >>> final_grads = merge_grads(
        ...     task_grads, ret_grads, lambda_weight=1.5, mode="weighted"
        ... )
    """
```

**合并模式**:
- `"weighted"`: `g_final = g_task + λ * g_retain`
- `"equal"`: `g_final = 0.5 * (g_task + g_retain)`
- `"task_priority"`: `g_final = g_task + min(λ, 1.0) * g_retain`

---

## 原对偶优化

### epsilon_schedule

计算当前步的 ε 值。

```python
from lerobot.craft.primal_dual import epsilon_schedule

def epsilon_schedule(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    total_steps: int,
    schedule_type: str = "linear"
) -> float:
    """
    计算当前步的 epsilon 值
    
    参数:
        step: 当前训练步数
        epsilon_start: 起始值
        epsilon_end: 结束值
        total_steps: 总步数
        schedule_type: 调度类型
    
    返回:
        epsilon: 当前 ε 值
    
    示例:
        >>> eps = epsilon_schedule(
        ...     step=5000, epsilon_start=1.0, epsilon_end=0.1, total_steps=10000
        ... )
        >>> print(f"Current epsilon: {eps:.4f}")  # 0.5500
    """
```

**调度类型**:
- `"linear"`: 线性退火
- `"cosine"`: 余弦退火
- `"exponential"`: 指数退火

### update_lambda

更新 Lagrangian 乘子 λ。

```python
from lerobot.craft.primal_dual import update_lambda

def update_lambda(
    current_lambda: float,
    retention_loss: float,
    epsilon: float,
    lambda_lr: float,
    lambda_max: float
) -> float:
    """
    更新 Lagrangian 乘子 λ
    
    参数:
        current_lambda: 当前 λ 值
        retention_loss: 保留损失值
        epsilon: 当前 ε 阈值
        lambda_lr: λ 学习率
        lambda_max: λ 最大值
    
    返回:
        new_lambda: 更新后的 λ 值
    
    示例:
        >>> new_lambda = update_lambda(
        ...     current_lambda=1.0,
        ...     retention_loss=0.8,
        ...     epsilon=1.0,
        ...     lambda_lr=0.01,
        ...     lambda_max=10.0
        ... )
        >>> print(f"New lambda: {new_lambda:.4f}")  # 0.998
    """
```

**更新规则**:
```
λ_new = clip(λ + λ_lr * (L_retain - ε), 0, λ_max)
```

---

## 保留损失

### compute_hidden_retention_loss

计算 hidden state 保留损失（主入口）。

```python
from lerobot.craft.retention_loss import compute_hidden_retention_loss

def compute_hidden_retention_loss(
    policy: PreTrainedPolicy,
    anchor_batch: dict,
    craft_config: CraftConfig
) -> tuple[torch.Tensor, dict]:
    """
    计算 hidden state 保留损失
    
    参数:
        policy: 策略模型
        anchor_batch: 锚点数据批次
        craft_config: CRaFT 配置
    
    返回:
        loss: 保留损失张量
        metrics: 指标字典
    
    示例:
        >>> anchor_batch = next(anchor_dl_iter)
        >>> loss, metrics = compute_hidden_retention_loss(
        ...     policy, anchor_batch, craft_config
        ... )
        >>> print(f"Retention loss: {metrics['retention_loss']:.4f}")
    """
```

**anchor_batch 格式**:
```python
{
    "pixel_values": torch.Tensor,  # [B, C, H, W]
    "input_ids": torch.Tensor,     # [B, seq_len]
    "attention_mask": torch.Tensor,  # [B, seq_len]
    "target_features": torch.Tensor,  # [B, hidden_dim]
    "meta": {
        "hidden_layer": int,
        "pooling": str,
        "dtype": str
    }
}
```

### extract_student_hidden_features

提取 student 模型的 hidden features。

```python
from lerobot.craft.retention_loss import extract_student_hidden_features

def extract_student_hidden_features(
    policy: PreTrainedPolicy,
    anchor_batch: dict,
    craft_config: CraftConfig
) -> torch.Tensor:
    """
    提取 student hidden features
    
    参数:
        policy: 策略模型
        anchor_batch: 锚点数据
        craft_config: CRaFT 配置
    
    返回:
        features: Hidden features [B, hidden_dim]
    
    示例:
        >>> features = extract_student_hidden_features(
        ...     policy, anchor_batch, craft_config
        ... )
        >>> print(f"Feature shape: {features.shape}")  # [8, 2048]
    """
```

---

## 锚点数据加载

### AnchorCacheDataset

锚点数据集类。

```python
from lerobot.craft.anchor_cache import AnchorCacheDataset

class AnchorCacheDataset(torch.utils.data.Dataset):
    """
    锚点数据集加载器
    
    参数:
        cache_dir: Cache 目录路径
        transform: 可选的数据转换
    
    示例:
        >>> dataset = AnchorCacheDataset(
        ...     cache_dir="data/anchor_hidden_cache"
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Sample keys: {sample.keys()}")
    """
    
    def __init__(self, cache_dir: str, transform=None):
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本"""
        pass
```

**返回格式**:
```python
{
    "pixel_values": torch.Tensor,
    "input_ids": torch.Tensor,
    "attention_mask": torch.Tensor,
    "target_features": torch.Tensor,  # 仅 hidden mode
    "labels": torch.Tensor,  # 仅 token_ce mode
    "meta": dict
}
```

---

## 训练脚本

### lerobot_train_craft

CRaFT 训练主函数。

```python
from lerobot.scripts.lerobot_train_craft import train_craft

@parser.wrap()
def train_craft(
    cfg: TrainPipelineConfig,
    craft_config: CraftConfig | None = None,
    accelerator: Accelerator | None = None
):
    """
    CRaFT 训练主函数
    
    参数:
        cfg: 训练配置
        craft_config: CRaFT 配置
        accelerator: 分布式训练加速器
    
    示例:
        >>> from lerobot.configs.train import TrainPipelineConfig
        >>> from lerobot.craft import CraftConfig
        >>> 
        >>> cfg = TrainPipelineConfig(...)
        >>> craft_cfg = CraftConfig(enabled=True, ...)
        >>> 
        >>> train_craft(cfg, craft_config=craft_cfg)
    """
```

### update_policy_craft

单步 CRaFT 训练更新。

```python
from lerobot.scripts.lerobot_train_craft import update_policy_craft

def update_policy_craft(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    task_batch: dict,
    anchor_batch: dict | None,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    craft_config: CraftConfig,
    current_lambda: float,
    current_epsilon: float,
    lr_scheduler=None,
    lock=None
) -> tuple[MetricsTracker, dict, float]:
    """
    执行单步 CRaFT 训练
    
    参数:
        train_metrics: 训练指标跟踪器
        policy: 策略模型
        task_batch: 任务数据批次
        anchor_batch: 锚点数据批次
        optimizer: 优化器
        grad_clip_norm: 梯度裁剪范数
        accelerator: 加速器
        craft_config: CRaFT 配置
        current_lambda: 当前 λ 值
        current_epsilon: 当前 ε 值
        lr_scheduler: 学习率调度器
        lock: 线程锁
    
    返回:
        train_metrics: 更新后的指标
        output_dict: 输出字典
        new_lambda: 更新后的 λ 值
    
    训练流程:
        1. 前向传播（任务数据）→ L_task
        2. 反向传播 → ∇L_task
        3. 前向传播（锚点数据）→ L_retain
        4. 反向传播 → ∇L_retain
        5. 梯度手术（投影）
        6. 合并梯度
        7. 优化器更新
        8. 更新 λ
    """
```

---

## 使用示例

### 完整训练流程

```python
from lerobot.configs.train import TrainPipelineConfig
from lerobot.craft import CraftConfig
from lerobot.scripts.lerobot_train_craft import train_craft

# 1. 创建配置
train_cfg = TrainPipelineConfig(
    policy=PolicyConfig(path="lerobot/pi0_fast"),
    dataset=DatasetConfig(repo_id="lerobot/aloha_sim_insertion_human"),
    training=TrainingConfig(
        offline_steps=10000,
        batch_size=8,
        lr=1e-4
    ),
    output_dir="outputs/craft_training"
)

craft_cfg = CraftConfig(
    enabled=True,
    retention_mode="hidden",
    anchor_cache_dir="data/anchor_hidden_cache",
    anchor_batch_size=8,
    retention_freq=1,
    initial_lambda=1.0,
    lambda_lr=0.01,
    epsilon_start=1.0,
    epsilon_end=0.1,
    use_grad_projection=True
)

# 2. 运行训练
train_craft(train_cfg, craft_config=craft_cfg)
```

### 自定义梯度手术

```python
from lerobot.craft.grad_surgery import compute_dot, project_if_conflict, merge_grads

# 1. 计算任务梯度
task_loss.backward()
task_grads = [p.grad.clone() for p in model.parameters()]

# 2. 计算保留梯度
optimizer.zero_grad()
retention_loss.backward()
retention_grads = [p.grad.clone() for p in model.parameters()]

# 3. 检测冲突
dot = compute_dot(task_grads, retention_grads)
print(f"Gradient dot product: {dot.item()}")

# 4. 投影（如果冲突）
task_proj, ret_proj, conflict = project_if_conflict(
    task_grads, retention_grads, conflict_threshold=-0.1
)
if conflict:
    print("Conflict detected and resolved!")

# 5. 合并梯度
final_grads = merge_grads(task_proj, ret_proj, lambda_weight=1.5)

# 6. 设置梯度
optimizer.zero_grad()
for param, grad in zip(model.parameters(), final_grads):
    if grad is not None:
        param.grad = grad

# 7. 优化器更新
optimizer.step()
```

### 自定义 ε 调度

```python
from lerobot.craft.primal_dual import epsilon_schedule

# 线性退火
for step in range(10000):
    eps = epsilon_schedule(
        step, epsilon_start=1.0, epsilon_end=0.1,
        total_steps=10000, schedule_type="linear"
    )
    print(f"Step {step}: epsilon = {eps:.4f}")
```

---

## 类型定义

```python
from typing import TypedDict

class AnchorBatch(TypedDict):
    """锚点数据批次类型"""
    pixel_values: torch.Tensor  # [B, C, H, W]
    input_ids: torch.Tensor     # [B, seq_len]
    attention_mask: torch.Tensor  # [B, seq_len]
    target_features: torch.Tensor  # [B, hidden_dim]
    meta: dict

class CraftMetrics(TypedDict):
    """CRaFT 训练指标类型"""
    retention_loss: float
    lambda_value: float
    epsilon_value: float
    grad_dot: float
    grad_cos: float
    grad_conflict: bool
```

---

## 常量

```python
# 默认配置
DEFAULT_CRAFT_CONFIG = CraftConfig(
    enabled=False,
    retention_mode="hidden",
    initial_lambda=1.0,
    lambda_lr=0.01,
    epsilon_start=1.0,
    epsilon_end=0.1,
    use_grad_projection=True,
    conflict_threshold=-0.1
)

# 支持的 retention 模式
RETENTION_MODES = ["token_ce", "hidden"]

# 支持的 pooling 策略
POOLING_STRATEGIES = [
    "mean_image_tokens",
    "mean_masked",
    "last_token",
    "cls_token"
]

# 支持的梯度合并模式
MERGE_MODES = ["weighted", "equal", "task_priority"]
```

---

## 异常

```python
class CraftConfigError(Exception):
    """CRaFT 配置错误"""
    pass

class AnchorCacheError(Exception):
    """锚点数据加载错误"""
    pass

class GradientSurgeryError(Exception):
    """梯度手术错误"""
    pass
```

---

## 参考

- [CRaFT 训练指南](craft/CRAFT_TRAINING_GUIDE.md)
- [Hidden Feature Cache](HIDDEN_FEATURE_CACHE_SUMMARY.md)
- [实验操作指南](EXPERIMENT_GUIDE.md)

---

**最后更新**: 2026-02-17

