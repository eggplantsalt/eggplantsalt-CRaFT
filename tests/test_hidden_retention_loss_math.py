#!/usr/bin/env python

"""
Hidden Retention Loss 数学验证测试
==================================

使用 tiny mock Transformer 验证 hidden retention loss 的数学正确性。
"""

import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    """
    Tiny mock Transformer 用于测试
    
    【结构】
    - Embedding layer
    - 2 个 Transformer layers
    - 输出 hidden states
    """
    
    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        """
        Forward pass
        
        【返回值】
        如果 output_hidden_states=True:
            返回 dict，包含 hidden_states: tuple of [B, seq_len, hidden_dim]
        否则:
            返回最后一层的 hidden states
        """
        # Embedding
        hidden_states = self.embedding(input_ids)  # [B, seq_len, hidden_dim]
        
        # 保存所有层的 hidden states
        all_hidden_states = [hidden_states] if output_hidden_states else None
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        if output_hidden_states:
            return {"hidden_states": tuple(all_hidden_states)}
        else:
            return hidden_states


def test_mse_loss_correctness():
    """测试 MSE loss 的数学正确性"""
    print("=" * 60)
    print("Test 1: MSE Loss Correctness")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    # 创建模拟数据
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 64
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    teacher_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # 计算 MSE loss
    loss = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="mse")
    
    # 手动计算 expected loss
    expected_loss = ((student_hidden - teacher_hidden) ** 2).mean()
    
    # 验证
    assert torch.allclose(loss, expected_loss, atol=1e-6), \
        f"MSE loss 不正确: {loss.item()} != {expected_loss.item()}"
    
    print(f"[OK] MSE Loss: {loss.item():.6f}")
    print(f"[OK] Expected: {expected_loss.item():.6f}")
    print(f"[OK] Difference: {abs(loss.item() - expected_loss.item()):.2e}")
    print()


def test_cosine_loss_correctness():
    """测试 Cosine loss 的数学正确性"""
    print("=" * 60)
    print("Test 2: Cosine Loss Correctness")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    # 创建模拟数据
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 64
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    teacher_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # 计算 Cosine loss
    loss = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="cosine")
    
    # 手动计算 expected loss
    student_flat = student_hidden.view(B, -1)
    teacher_flat = teacher_hidden.view(B, -1)
    cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)
    expected_loss = (1.0 - cosine_sim).mean()
    
    # 验证
    assert torch.allclose(loss, expected_loss, atol=1e-6), \
        f"Cosine loss 不正确: {loss.item()} != {expected_loss.item()}"
    
    print(f"[OK] Cosine Loss: {loss.item():.6f}")
    print(f"[OK] Expected: {expected_loss.item():.6f}")
    print(f"[OK] Cosine Similarity: {cosine_sim.mean().item():.6f}")
    print()


def test_loss_range():
    """测试 loss 的取值范围"""
    print("=" * 60)
    print("Test 3: Loss Range")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 64
    
    # 测试 1: 相同的 hidden states（loss 应该接近 0）
    hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    loss_mse = compute_retention_loss_hidden(hidden, hidden, loss_type="mse")
    loss_cosine = compute_retention_loss_hidden(hidden, hidden, loss_type="cosine")
    
    assert loss_mse.item() < 1e-6, f"相同 hidden states 的 MSE loss 应该接近 0: {loss_mse.item()}"
    assert loss_cosine.item() < 1e-6, f"相同 hidden states 的 Cosine loss 应该接近 0: {loss_cosine.item()}"
    
    print(f"[OK] Identical hidden states:")
    print(f"     MSE Loss: {loss_mse.item():.2e} (should be ~0)")
    print(f"     Cosine Loss: {loss_cosine.item():.2e} (should be ~0)")
    
    # 测试 2: 完全相反的 hidden states（cosine loss 应该接近 2）
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    teacher_hidden = -student_hidden
    
    loss_cosine = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="cosine")
    
    assert 1.9 < loss_cosine.item() < 2.1, \
        f"相反 hidden states 的 Cosine loss 应该接近 2: {loss_cosine.item()}"
    
    print(f"[OK] Opposite hidden states:")
    print(f"     Cosine Loss: {loss_cosine.item():.6f} (should be ~2)")
    print()


def test_gradient_flow():
    """测试梯度是否能正确反向传播"""
    print("=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 64
    
    # 创建需要梯度的 student hidden states
    student_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim, requires_grad=True)
    teacher_hidden = torch.randn(B, n_layers, n_vecs, hidden_dim)
    
    # 计算 loss
    loss = compute_retention_loss_hidden(student_hidden, teacher_hidden, loss_type="mse")
    
    # 反向传播
    loss.backward()
    
    # 验证梯度存在
    assert student_hidden.grad is not None, "Student hidden states 应该有梯度"
    assert not torch.all(student_hidden.grad == 0), "梯度不应该全为 0"
    
    grad_norm = student_hidden.grad.norm().item()
    
    print(f"[OK] Gradient exists: {student_hidden.grad is not None}")
    print(f"[OK] Gradient norm: {grad_norm:.6f}")
    print(f"[OK] Gradient shape: {student_hidden.grad.shape}")
    print()


def test_pooling_strategies():
    """测试不同的 pooling 策略"""
    print("=" * 60)
    print("Test 5: Pooling Strategies")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import pool_hidden_states
    
    B, seq_len, hidden_dim = 4, 50, 64
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    attention_mask = torch.ones(B, seq_len)
    input_ids = torch.randint(0, 100, (B, seq_len))
    
    # Mock policy
    class MockPolicy:
        class Config:
            image_seq_length = 20
        config = Config()
    
    policy = MockPolicy()
    
    # 测试所有 pooling 策略
    pooling_strategies = ["mean_image_tokens", "mean_masked", "last_token", "cls_token"]
    
    for pooling in pooling_strategies:
        pooled = pool_hidden_states(hidden_states, attention_mask, pooling, policy, input_ids)
        
        # 验证 shape
        assert pooled.shape == (B, hidden_dim), \
            f"{pooling}: shape 错误 {pooled.shape} != {(B, hidden_dim)}"
        
        # 验证不是全零
        assert not torch.all(pooled == 0), f"{pooling}: pooled features 不应该全为 0"
        
        print(f"[OK] {pooling:20s}: shape={pooled.shape}, norm={pooled.norm(dim=1).mean().item():.4f}")
    
    print()


def test_float32_stability():
    """测试 float32 计算的稳定性"""
    print("=" * 60)
    print("Test 6: Float32 Stability")
    print("=" * 60)
    
    from lerobot.craft.retention_loss import compute_retention_loss_hidden
    
    B, n_layers, n_vecs, hidden_dim = 4, 2, 2, 64
    
    # 创建 float16 数据
    student_hidden_f16 = torch.randn(B, n_layers, n_vecs, hidden_dim).half()
    teacher_hidden_f16 = torch.randn(B, n_layers, n_vecs, hidden_dim).half()
    
    # 计算 loss（内部会转换到 float32）
    loss = compute_retention_loss_hidden(student_hidden_f16, teacher_hidden_f16, loss_type="mse")
    
    # 验证 loss 是 float32
    assert loss.dtype == torch.float32, f"Loss 应该是 float32: {loss.dtype}"
    
    # 验证 loss 是有限的
    assert torch.isfinite(loss), "Loss 应该是有限的"
    
    print(f"[OK] Input dtype: {student_hidden_f16.dtype}")
    print(f"[OK] Loss dtype: {loss.dtype}")
    print(f"[OK] Loss value: {loss.item():.6f}")
    print(f"[OK] Loss is finite: {torch.isfinite(loss).item()}")
    print()


def test_with_tiny_transformer():
    """使用 tiny transformer 进行端到端测试"""
    print("=" * 60)
    print("Test 7: End-to-End with Tiny Transformer")
    print("=" * 60)
    
    # 创建 student 和 teacher 模型
    student = TinyTransformer(vocab_size=100, hidden_dim=64, num_layers=2)
    teacher = TinyTransformer(vocab_size=100, hidden_dim=64, num_layers=2)
    
    # 创建输入
    B, seq_len = 4, 20
    input_ids = torch.randint(0, 100, (B, seq_len))
    
    # Forward pass
    student_outputs = student(input_ids, output_hidden_states=True)
    teacher_outputs = teacher(input_ids, output_hidden_states=True)
    
    # 提取最后一层的 hidden states
    student_hidden = student_outputs["hidden_states"][-1]  # [B, seq_len, hidden_dim]
    teacher_hidden = teacher_outputs["hidden_states"][-1]
    
    # Pooling: mean over sequence
    student_pooled = student_hidden.mean(dim=1)  # [B, hidden_dim]
    teacher_pooled = teacher_hidden.mean(dim=1)
    
    # 计算 MSE loss
    loss = F.mse_loss(student_pooled, teacher_pooled)
    
    # 反向传播
    loss.backward()
    
    # 验证梯度
    has_grad = any(p.grad is not None for p in student.parameters())
    assert has_grad, "Student 模型应该有梯度"
    
    print(f"[OK] Student hidden shape: {student_hidden.shape}")
    print(f"[OK] Teacher hidden shape: {teacher_hidden.shape}")
    print(f"[OK] Pooled shape: {student_pooled.shape}")
    print(f"[OK] Loss: {loss.item():.6f}")
    print(f"[OK] Gradients exist: {has_grad}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Hidden Retention Loss Math Tests")
    print("=" * 60)
    print()
    
    try:
        test_mse_loss_correctness()
        test_cosine_loss_correctness()
        test_loss_range()
        test_gradient_flow()
        test_pooling_strategies()
        test_float32_stability()
        test_with_tiny_transformer()
        
        print("=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

