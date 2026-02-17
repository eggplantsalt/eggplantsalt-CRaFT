#!/usr/bin/env python

"""
CRaFT Hidden State Anchoring 单元测试
====================================

测试 hidden state retention loss 的数学正确性和 pooling 逻辑。
"""

import torch
import pytest


def test_compute_retention_loss_hidden_mse():
    """测试 MSE loss 计算"""
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    # 创建模拟 hidden states
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 128
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    teacher_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # 计算 MSE loss
    loss = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="mse")
    
    # 验证
    assert loss.ndim == 0, "Loss 应该是标量"
    assert loss.item() >= 0, "MSE loss 应该非负"
    
    # 验证数值正确性
    expected_loss = ((student_hidden - teacher_hidden) ** 2).mean()
    assert torch.allclose(loss, expected_loss, atol=1e-6), "MSE loss 计算错误"
    
    print(f"✓ MSE Loss 测试通过: {loss.item():.6f}")


def test_compute_retention_loss_hidden_cosine():
    """测试 Cosine loss 计算"""
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 128
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    teacher_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # 计算 Cosine loss
    loss = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="cosine")
    
    # 验证
    assert loss.ndim == 0, "Loss 应该是标量"
    assert 0 <= loss.item() <= 2, "Cosine loss 应该在 [0, 2] 范围内"
    
    print(f"✓ Cosine Loss 测试通过: {loss.item():.6f}")


def test_compute_retention_loss_hidden_identical():
    """测试相同 hidden states 的 loss 应该为 0"""
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 128
    hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # MSE loss 应该为 0
    loss_mse = compute_retention_loss_hidden(hidden, hidden.clone(), loss_type="mse")
    assert loss_mse.item() < 1e-6, "相同 hidden states 的 MSE loss 应该接近 0"
    
    # Cosine loss 应该为 0
    loss_cosine = compute_retention_loss_hidden(hidden, hidden.clone(), loss_type="cosine")
    assert loss_cosine.item() < 1e-6, "相同 hidden states 的 Cosine loss 应该接近 0"
    
    print("✓ 相同 hidden states 测试通过")


def test_pooling_shape():
    """测试 pooling 后的 shape 正确性"""
    # 模拟 hidden states
    B, seq_len, hidden_dim = 4, 100, 2048
    num_vision_tokens = 64
    num_text_tokens = 36
    
    hidden_state = torch.randn(B, seq_len, hidden_dim)
    
    # Vision pooling: mean
    vision_hidden = hidden_state[:, :num_vision_tokens, :]
    vision_pooled = vision_hidden.mean(dim=1)
    assert vision_pooled.shape == (B, hidden_dim), f"Vision pooled shape 错误: {vision_pooled.shape}"
    
    # Text pooling: last token
    text_hidden = hidden_state[:, num_vision_tokens:num_vision_tokens+num_text_tokens, :]
    text_masks = torch.ones(B, num_text_tokens, dtype=torch.bool)
    last_text_indices = text_masks.sum(dim=1) - 1
    text_pooled = text_hidden[torch.arange(B), last_text_indices]
    assert text_pooled.shape == (B, hidden_dim), f"Text pooled shape 错误: {text_pooled.shape}"
    
    # 拼接
    layer_pooled = torch.stack([vision_pooled, text_pooled], dim=1)
    assert layer_pooled.shape == (B, 2, hidden_dim), f"Layer pooled shape 错误: {layer_pooled.shape}"
    
    print("✓ Pooling shape 测试通过")


def test_device_dtype_compatibility():
    """测试设备和 dtype 兼容性"""
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 2, 2, 2, 64
    
    # 测试不同 dtype
    student_fp32 = torch.randn(B, n_layers, n_vecs, hidden_dim, dtype=torch.float32)
    teacher_fp16 = torch.randn(B, n_layers, n_vecs, hidden_dim, dtype=torch.float16)
    
    loss = compute_retention_loss_hidden(student_fp32, teacher_fp16, loss_type="mse")
    assert loss.dtype == torch.float32, "Loss dtype 应该与 student 一致"
    
    print("✓ Device/dtype 兼容性测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("CRaFT Hidden State Anchoring 单元测试")
    print("=" * 60)
    
    test_compute_retention_loss_hidden_mse()
    test_compute_retention_loss_hidden_cosine()
    test_compute_retention_loss_hidden_identical()
    test_pooling_shape()
    test_device_dtype_compatibility()
    
    print("=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

