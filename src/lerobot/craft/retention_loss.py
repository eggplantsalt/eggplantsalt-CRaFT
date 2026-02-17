#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
保留损失计算模块 (Retention Loss - Hidden State Anchoring)
==========================================================

【模块功能】
在锚点数据上计算保留损失，使用 hidden state 表征蒸馏而非 token-level distillation。

【核心思想】
保留损失通过比较 student 和 teacher 的 hidden states 来衡量模型对已学习知识的记忆程度。
相比 token-level distillation，hidden state anchoring 更稳定，不受输出 token 不稳定性的影响。

【数学定义】
L_retain = MSE(student_hidden, teacher_hidden)
或
L_retain = 1 - CosineSimilarity(student_hidden, teacher_hidden)

【优势】
1. 稳定性: hidden states 比 tokens 更平滑，不受生成随机性影响
2. 避免 token 问题: π0_fast 不稳定输出自然语言时仍可用
3. 高效性: 只需比较少量 pooled vectors

【使用示例】
```python
from lerobot.craft.retention_loss import compute_retention_loss_hidden

# 在训练循环中
for step in range(total_steps):
    # 1. 计算任务损失（新数据）
    task_batch = next(task_dataloader)
    task_loss, _ = policy.forward(task_batch)
    
    # 2. 计算保留损失（锚点数据）
    anchor_batch = next(anchor_dataloader)
    
    # 提取 student hidden states
    student_hidden = extract_student_hidden_states(policy, anchor_batch)
    
    # 计算 hidden state retention loss
    retention_loss = compute_retention_loss_hidden(
        student_hidden,
        anchor_batch["teacher_hidden"],
        loss_type="mse"
    )
```
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_retention_loss_hidden(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    loss_type: str = "mse",
    reduction: str = "mean",
) -> Tensor:
    """
    计算 hidden state 保留损失
    
    【功能说明】
    比较 student 和 teacher 的 hidden states，计算表征蒸馏损失。
    
    【参数】
    student_hidden: Tensor
        Student 模型的 hidden states
        Shape: [B, n_layers, n_vecs, hidden_dim]
        - B: batch size
        - n_layers: 层数（通常为 2，对应 [-2, -1] 层）
        - n_vecs: 向量数（通常为 2，vision_pooled + text_pooled）
        - hidden_dim: hidden dimension（例如 2048）
        
    teacher_hidden: Tensor
        Teacher 模型的 hidden states（从 AnchorCache 加载）
        Shape: [B, n_layers, n_vecs, hidden_dim]
        
    loss_type: str = "mse"
        损失类型
        - "mse": Mean Squared Error（推荐，稳定）
        - "cosine": Cosine Similarity Loss（方向对齐）
        - "l1": L1 Loss（稀疏性）
        
    reduction: str = "mean"
        损失归约模式
        - "mean": 返回批次平均损失（推荐）
        - "sum": 返回批次总损失
        - "none": 返回每个样本的损失
    
    【返回值】
    Tensor: 保留损失标量张量
    
    【实现细节】
    1. MSE Loss:
       L = mean((student - teacher)^2)
       
    2. Cosine Loss:
       L = 1 - mean(cosine_similarity(student, teacher))
       
    3. L1 Loss:
       L = mean(|student - teacher|)
    
    【示例】
    >>> import torch
    >>> 
    >>> # 模拟 hidden states
    >>> student_hidden = torch.randn(8, 2, 2, 2048)  # [B, n_layers, n_vecs, hidden_dim]
    >>> teacher_hidden = torch.randn(8, 2, 2, 2048)
    >>> 
    >>> # 计算 MSE loss
    >>> loss_mse = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="mse")
    >>> print(f"MSE Loss: {loss_mse.item():.4f}")
    >>> 
    >>> # 计算 Cosine loss
    >>> loss_cosine = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="cosine")
    >>> print(f"Cosine Loss: {loss_cosine.item():.4f}")
    """
    # 检查形状
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"Student 和 Teacher hidden states 形状不匹配: "
            f"student={student_hidden.shape}, teacher={teacher_hidden.shape}"
        )
    
    # 确保 teacher_hidden 在正确的设备上
    if teacher_hidden.device != student_hidden.device:
        teacher_hidden = teacher_hidden.to(student_hidden.device)
    
    # 转换到 float32 计算（更稳定）
    student_hidden = student_hidden.float()
    teacher_hidden = teacher_hidden.float()
    
    if loss_type == "mse":
        # MSE Loss: mean((student - teacher)^2)
        loss = F.mse_loss(student_hidden, teacher_hidden, reduction="none")
        # loss shape: [B, n_layers, n_vecs, hidden_dim]
        
        # 对所有维度求平均（除了 batch 维度，如果 reduction="none"）
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            # 对每个样本求平均（保留 batch 维度）
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    elif loss_type == "cosine":
        # Cosine Similarity Loss: 1 - cosine_similarity
        # Flatten to [B, n_layers * n_vecs * hidden_dim]
        student_flat = student_hidden.view(student_hidden.shape[0], -1)
        teacher_flat = teacher_hidden.view(teacher_hidden.shape[0], -1)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)  # [B]
        
        # Loss = 1 - cosine_similarity (range: [0, 2])
        loss = 1.0 - cosine_sim
        
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass  # Already per-sample
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    elif loss_type == "l1":
        # L1 Loss: mean(|student - teacher|)
        loss = F.l1_loss(student_hidden, teacher_hidden, reduction="none")
        
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of ['mse', 'cosine', 'l1']")
    
    return loss


def extract_student_hidden_with_pooling(
    policy,
    anchor_batch: dict,
    layers_to_extract: list[int],
    meta: dict,
) -> Tensor:
    """
    从 student 模型提取 hidden states 并进行 pooling（与 teacher 相同的策略）
    
    【功能说明】
    运行 student forward pass，提取指定层的 hidden states，并按照 meta 中的
    pooling 策略进行 pooling，使其与 teacher hidden states 的形状一致。
    
    【参数】
    policy: PreTrainedPolicy
        Student 策略模型（例如 PI0FastPolicy）
        
    anchor_batch: dict
        锚点数据批次，包含：
        - pixel_values: [B, C, H, W]
        - input_ids: [B, seq_len]
        - attention_mask: [B, seq_len]
        
    layers_to_extract: list[int]
        要提取的层索引（与 teacher 相同）
        例如: [-2, -1] 表示最后两层
        
    meta: dict
        元数据，包含：
        - num_vision_tokens: 视觉 token 数量
        - num_text_tokens: 文本 token 数量
        - pooling_strategy: {"vision": "mean", "text": "last"}
    
    【返回值】
    Tensor: Student hidden states [B, n_layers, n_vecs, hidden_dim]
    
    【实现步骤】
    1. 准备输入（与 teacher 相同的格式）
    2. 运行 forward pass，设置 output_hidden_states=True
    3. 提取指定层的 hidden states
    4. 按照 meta 中的 pooling 策略进行 pooling
    5. 返回 pooled hidden states
    
    【示例】
    >>> # 在训练循环中
    >>> anchor_batch = next(anchor_dl_iter)
    >>> meta = anchor_batch["meta"]
    >>> layers_to_extract = meta["layers_to_save"]
    >>> 
    >>> # 提取 student hidden states
    >>> student_hidden = extract_student_hidden_with_pooling(
    ...     policy,
    ...     anchor_batch,
    ...     layers_to_extract,
    ...     meta
    ... )
    >>> 
    >>> # 计算 retention loss
    >>> retention_loss = compute_retention_loss_hidden(
    ...     student_hidden,
    ...     anchor_batch["teacher_hidden"]
    ... )
    """
    policy.eval()  # 确保在评估模式（虽然需要梯度，但不需要 dropout）
    
    device = next(policy.parameters()).device
    
    # 准备输入
    pixel_values = anchor_batch["pixel_values"].to(device)
    input_ids = anchor_batch["input_ids"].to(device)
    attention_mask = anchor_batch["attention_mask"].to(device)
    
    bsize = pixel_values.shape[0]
    
    # 准备图像输入（与 teacher 相同）
    images = [pixel_values]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device)]
    
    # 添加 BOS token（与 teacher 相同）
    bos_token = torch.full(
        (bsize, 1), policy._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
    )
    tokens = torch.cat([input_ids, bos_token], dim=1)
    masks = torch.cat([attention_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
    
    # Embed prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = (
        policy.model.embed_prefix_fast(
            images, img_masks, tokens, masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
    )
    
    if policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
    
    # Forward pass 并提取 hidden states
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = policy.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
    
    language_model = policy.model.paligemma_with_expert.paligemma.language_model
    outputs = language_model.forward(
        inputs_embeds=prefix_embs,
        attention_mask=att_4d,
        position_ids=position_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # 提取指定层的 hidden states
    all_hidden_states = outputs.hidden_states
    total_layers = len(all_hidden_states) - 1
    
    selected_hidden_states = []
    for layer_idx in layers_to_extract:
        if layer_idx < 0:
            actual_idx = total_layers + layer_idx + 1
        else:
            actual_idx = layer_idx + 1
        
        if 0 <= actual_idx < len(all_hidden_states):
            selected_hidden_states.append(all_hidden_states[actual_idx])
    
    # Pooling（与 teacher 相同的策略）
    num_vision_tokens = meta["num_vision_tokens"]
    num_text_tokens = meta["num_text_tokens"]
    
    pooled_hidden_states = []
    
    for hidden_state in selected_hidden_states:
        # Vision tokens: mean pooling
        vision_hidden = hidden_state[:, :num_vision_tokens, :]
        vision_pooled = vision_hidden.mean(dim=1)  # [B, hidden_dim]
        
        # Text tokens: last token
        text_hidden = hidden_state[:, num_vision_tokens:, :]
        text_masks = masks  # [B, num_text_tokens]
        last_text_indices = text_masks.sum(dim=1) - 1  # [B]
        text_pooled = text_hidden[torch.arange(bsize), last_text_indices]  # [B, hidden_dim]
        
        # 拼接: [B, 2, hidden_dim]
        layer_pooled = torch.stack([vision_pooled, text_pooled], dim=1)
        pooled_hidden_states.append(layer_pooled)
    
    # 拼接所有层: [B, n_layers, 2, hidden_dim]
    student_hidden = torch.stack(pooled_hidden_states, dim=1)
    
    return student_hidden


def compute_hidden_retention_loss(
    policy,
    anchor_batch: dict,
    craft_config,
) -> tuple[Tensor, dict]:
    """
    计算 hidden state 保留损失（主入口函数）
    
    【功能说明】
    这是 retention_mode=hidden 的主入口函数，负责：
    1. 从 student 模型提取 hidden states
    2. 与 anchor_batch 中的 target_features 比较
    3. 计算 MSE 或 cosine loss
    4. 返回 loss 和 metrics
    
    【参数】
    policy: PreTrainedPolicy
        Student 策略模型（例如 PI0FastPolicy）
        
    anchor_batch: dict
        锚点数据批次，必须包含：
        - pixel_values: [B, C, H, W]
        - input_ids: [B, seq_len]
        - attention_mask: [B, seq_len]
        - target_features: [B, hidden_dim]  # Teacher pooled features
        - meta: dict  # 包含 hidden_layer, pooling, dtype
        
    craft_config: CraftConfig
        CRaFT 配置，包含：
        - hidden_layer: 要提取的层索引（默认 -2）
        - pooling: Pooling 策略（默认 mean_image_tokens）
        - loss_type: 损失类型（mse 或 cosine）
    
    【返回值】
    tuple[Tensor, dict]:
        - loss: 保留损失标量张量（float32，稳定）
        - metrics: 指标字典，包含：
            - retention_loss: loss 值
            - student_hidden_norm: student hidden states 的范数
            - target_features_norm: target features 的范数
    
    【实现细节】
    1. 使用 policy 原生的 output_hidden_states=True（优先）
    2. 提取 craft_config.hidden_layer 指定的层
    3. 使用与 cache 一致的 pooling 策略
    4. 在 float32 中计算 loss（更稳定）
    5. 支持反向传播
    
    【示例】
    >>> from lerobot.craft.retention_loss import compute_hidden_retention_loss
    >>> 
    >>> # 在训练循环中
    >>> anchor_batch = next(anchor_dl_iter)
    >>> 
    >>> # 计算 hidden retention loss
    >>> retention_loss, metrics = compute_hidden_retention_loss(
    ...     policy,
    ...     anchor_batch,
    ...     craft_config
    ... )
    >>> 
    >>> # 反向传播
    >>> retention_loss.backward()
    >>> 
    >>> # 记录 metrics
    >>> print(f"Retention Loss: {metrics['retention_loss']:.4f}")
    """
    # 检查 anchor_batch 格式
    if "target_features" not in anchor_batch:
        raise ValueError(
            "anchor_batch 必须包含 'target_features' 字段。"
            "请使用 build_anchor_hidden_cache.py 生成 hidden feature cache。"
        )
    
    # 提取 student hidden features
    student_features = extract_student_hidden_features(
        policy,
        anchor_batch,
        craft_config,
    )
    
    # 获取 target features
    target_features = anchor_batch["target_features"]
    
    # 确保在相同设备和 dtype
    if target_features.device != student_features.device:
        target_features = target_features.to(student_features.device)
    
    # 转换到 float32 计算（更稳定）
    student_features_f32 = student_features.float()
    target_features_f32 = target_features.float()
    
    # 计算 loss
    loss_type = getattr(craft_config, "loss_type", "mse")
    
    if loss_type == "mse":
        loss = F.mse_loss(student_features_f32, target_features_f32, reduction="mean")
    elif loss_type == "cosine":
        # Cosine similarity loss: 1 - cosine_similarity
        cosine_sim = F.cosine_similarity(student_features_f32, target_features_f32, dim=1)
        loss = (1.0 - cosine_sim).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'mse' or 'cosine'")
    
    # 计算 metrics
    metrics = {
        "retention_loss": loss.item(),
        "student_hidden_norm": student_features_f32.norm(dim=1).mean().item(),
        "target_features_norm": target_features_f32.norm(dim=1).mean().item(),
    }
    
    return loss, metrics


def extract_student_hidden_features(
    policy,
    anchor_batch: dict,
    craft_config,
) -> Tensor:
    """
    从 student 模型提取 hidden features（与 teacher 相同的 pooling 策略）
    
    【功能说明】
    运行 student forward pass，提取指定层的 hidden states，并按照
    anchor_batch["meta"] 中的 pooling 策略进行 pooling。
    
    【参数】
    policy: PreTrainedPolicy
        Student 策略模型
        
    anchor_batch: dict
        锚点数据批次，包含：
        - pixel_values: [B, C, H, W]
        - input_ids: [B, seq_len]
        - attention_mask: [B, seq_len]
        - meta: dict  # 包含 hidden_layer, pooling
        
    craft_config: CraftConfig
        CRaFT 配置
    
    【返回值】
    Tensor: Student hidden features [B, hidden_dim]
    
    【实现策略】
    1. 优先使用模型原生的 output_hidden_states=True
    2. 提取 meta["hidden_layer"] 指定的层
    3. 使用 meta["pooling"] 指定的 pooling 策略
    4. 返回 pooled features
    """
    device = next(policy.parameters()).device
    
    # 准备输入
    pixel_values = anchor_batch["pixel_values"].to(device)
    input_ids = anchor_batch["input_ids"].to(device)
    attention_mask = anchor_batch["attention_mask"].to(device)
    
    # 获取 meta 信息
    meta = anchor_batch["meta"]
    hidden_layer = meta.get("hidden_layer", -2)
    pooling = meta.get("pooling", "mean_image_tokens")
    
    # 使用模型原生的 output_hidden_states
    # 对于 PI0FastPolicy，需要访问底层的 language_model
    try:
        # 尝试使用 _paligemma_model（如果存在）
        if hasattr(policy, '_paligemma_model'):
            outputs = policy._paligemma_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            all_hidden_states = outputs.hidden_states
        else:
            # Fallback: 手动构造 forward pass
            bsize = pixel_values.shape[0]
            
            # 准备图像输入
            images = [pixel_values]
            img_masks = [torch.ones(bsize, dtype=torch.bool, device=device)]
            
            # 添加 BOS token
            bos_token = torch.full(
                (bsize, 1), policy._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
            )
            tokens = torch.cat([input_ids, bos_token], dim=1)
            masks = torch.cat([attention_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
            
            # Embed prefix
            prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = (
                policy.model.embed_prefix_fast(
                    images, img_masks, tokens, masks,
                    fast_action_tokens=None,
                    fast_action_masks=None,
                )
            )
            
            if policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            
            # Forward pass
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = policy.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
            
            language_model = policy.model.paligemma_with_expert.paligemma.language_model
            outputs = language_model.forward(
                inputs_embeds=prefix_embs,
                attention_mask=att_4d,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden_states = outputs.hidden_states
    except Exception as e:
        raise RuntimeError(f"无法提取 hidden states: {e}")
    
    # 提取指定层的 hidden states
    total_layers = len(all_hidden_states) - 1
    if hidden_layer < 0:
        actual_idx = total_layers + hidden_layer + 1
    else:
        actual_idx = hidden_layer + 1
    
    if not (0 <= actual_idx < len(all_hidden_states)):
        raise ValueError(f"Invalid hidden_layer: {hidden_layer} (total layers: {total_layers})")
    
    hidden_state = all_hidden_states[actual_idx]  # [B, seq_len, hidden_dim]
    
    # Pooling
    pooled_features = pool_hidden_states(
        hidden_state,
        attention_mask,
        pooling,
        policy,
        input_ids,
    )
    
    return pooled_features


def pool_hidden_states(
    hidden_states: Tensor,
    attention_mask: Tensor,
    pooling: str,
    policy,
    input_ids: Tensor,
) -> Tensor:
    """
    对 hidden states 进行 pooling
    
    【参数】
    hidden_states: [B, seq_len, hidden_dim]
    attention_mask: [B, seq_len]
    pooling: Pooling 策略
    policy: Policy 实例（用于识别图像 tokens）
    input_ids: [B, seq_len]
    
    【返回值】
    pooled_features: [B, hidden_dim]
    """
    B, seq_len, hidden_dim = hidden_states.shape
    
    if pooling == "mean_image_tokens":
        # 识别图像 tokens 范围
        num_image_tokens = identify_image_tokens(policy)
        
        # 提取图像 tokens 的 hidden states
        image_hidden = hidden_states[:, :num_image_tokens, :]  # [B, num_image_tokens, hidden_dim]
        
        # 平均池化
        pooled = image_hidden.mean(dim=1)  # [B, hidden_dim]
        
    elif pooling == "mean_masked":
        # 对所有非 padding tokens 取平均
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_hidden = hidden_states * mask
        pooled = masked_hidden.sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [B, hidden_dim]
        
    elif pooling == "last_token":
        # 取最后一个非 padding token
        lengths = attention_mask.sum(dim=1).long() - 1  # [B]
        pooled = hidden_states[torch.arange(B, device=hidden_states.device), lengths]  # [B, hidden_dim]
        
    elif pooling == "cls_token":
        # 取第一个 token（假设是 CLS）
        pooled = hidden_states[:, 0, :]  # [B, hidden_dim]
        
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    return pooled


def identify_image_tokens(policy) -> int:
    """
    识别图像 tokens 的数量
    
    【返回值】
    int: 图像 tokens 数量
    """
    # 方法 1: 从 policy config 获取
    if hasattr(policy, 'config') and hasattr(policy.config, 'image_seq_length'):
        return policy.config.image_seq_length
    
    # 方法 2: 计算
    if hasattr(policy, 'config') and hasattr(policy.config, 'image_resolution') and hasattr(policy.config, 'patch_size'):
        h, w = policy.config.image_resolution
        patch_size = policy.config.patch_size
        return (h // patch_size) * (w // patch_size)
    
    # 方法 3: 默认值（PaliGemma 224x224, patch_size=16）
    return 196


# 向后兼容：保留旧的 token-level 函数签名（但标记为 deprecated）
def compute_retention_loss(
    policy,
    anchor_batch: dict,
    reduction: str = "mean",
) -> Tensor:
    """
    在锚点数据批次上计算保留损失（Token-level，已弃用）
    
    【警告】
    此函数用于 token-level distillation，已被 compute_hidden_retention_loss 替代。
    如果 anchor_batch 包含 target_features，请使用 compute_hidden_retention_loss。
    
    【功能说明】
    直接调用 policy.forward() 方法计算 token-level CE loss。
    
    【参数】
    policy: PreTrainedPolicy
        策略模型
    anchor_batch: dict
        锚点数据批次（必须包含 labels）
    reduction: str = "mean"
        损失归约模式
    
    【返回值】
    Tensor: 保留损失标量张量
    """
    # 检查是否为 hidden feature anchoring
    if "target_features" in anchor_batch:
        raise ValueError(
            "检测到 hidden feature anchoring cache，请使用 compute_hidden_retention_loss() "
            "而非 compute_retention_loss()"
        )
    
    # Token-level distillation（向后兼容）
    loss, _ = policy.forward(anchor_batch)
    return loss
