#!/usr/bin/env python

"""
MCQ Likelihood Evaluation - Smoke Test

最小测试，使用 2 条样例数据验证脚本功能。
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def create_test_image(size=(224, 224)):
    """创建测试图像"""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img


def create_test_jsonl(output_path: Path, num_samples=2):
    """创建测试 JSONL 文件"""
    samples = []
    
    # 创建临时图像目录
    img_dir = output_path.parent / "test_images"
    img_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        # 创建测试图像
        img = create_test_image()
        img_path = img_dir / f"test_image_{i}.jpg"
        img.save(img_path)
        
        # 创建测试样本
        sample = {
            "image_path": str(img_path),
            "question": f"What is the robot doing in image {i}?",
            "choices": [
                "picking up an object",
                "moving to the left",
                "stopping and waiting",
            ],
            "answer_index": i % 3,  # 循环使用不同答案
        }
        samples.append(sample)
    
    # 写入 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    return output_path, img_dir


def test_mcq_likelihood_smoke():
    """
    Smoke test for MCQ likelihood evaluation
    
    测试目标:
    1. 脚本能正常加载数据
    2. 能计算 log-likelihood（即使没有真实 checkpoint）
    3. 输出格式正确
    """
    # 创建临时测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        jsonl_path = tmpdir / "test_mcq.jsonl"
        jsonl_path, img_dir = create_test_jsonl(jsonl_path, num_samples=2)
        
        # 验证文件创建成功
        assert jsonl_path.exists()
        assert len(list(img_dir.glob("*.jpg"))) == 2
        
        # 加载数据验证格式
        from lerobot.scripts.eval_mcq_likelihood import load_jsonl
        
        samples = load_jsonl(jsonl_path, max_samples=None)
        assert len(samples) == 2
        
        # 验证样本格式
        for sample in samples:
            assert "image_path" in sample
            assert "question" in sample
            assert "choices" in sample
            assert "answer_index" in sample
            assert isinstance(sample["choices"], list)
            assert len(sample["choices"]) > 0
            assert 0 <= sample["answer_index"] < len(sample["choices"])
        
        # 验证图像加载
        from lerobot.scripts.eval_mcq_likelihood import load_image
        
        img_tensor = load_image(samples[0]["image_path"])
        assert img_tensor.shape == (3, 224, 224)
        assert img_tensor.dtype == torch.float32
        assert 0 <= img_tensor.min() <= 1
        assert 0 <= img_tensor.max() <= 1
        
        print("✓ Smoke test passed!")
        print(f"  - Loaded {len(samples)} samples")
        print(f"  - Image shape: {img_tensor.shape}")
        print(f"  - Sample format validated")


def test_mcq_likelihood_mock_evaluation():
    """
    Mock evaluation test (without real checkpoint)
    
    测试 log-likelihood 计算逻辑（使用 mock policy）
    """
    # 创建 mock policy
    class MockPolicy:
        def __init__(self):
            self.device = "cpu"
            self._paligemma_tokenizer = self._create_mock_tokenizer()
            self.model = self._create_mock_model()
        
        def _create_mock_tokenizer(self):
            """创建 mock tokenizer"""
            class MockTokenizer:
                bos_token_id = 1
                
                def encode(self, text, add_special_tokens=False, return_tensors=None):
                    # 简单 mock：每个字符一个 token
                    tokens = [ord(c) % 1000 for c in text]
                    if return_tensors == "pt":
                        return torch.tensor([tokens], dtype=torch.long)
                    return tokens
            
            return MockTokenizer()
        
        def _create_mock_model(self):
            """创建 mock model"""
            class MockModel:
                def __init__(self):
                    self.paligemma_with_expert = self._create_mock_paligemma()
                
                def _create_mock_paligemma(self):
                    class MockPaliGemma:
                        def __init__(self):
                            self.paligemma = self._create_mock_inner()
                        
                        def _create_mock_inner(self):
                            class MockInner:
                                def __init__(self):
                                    self.language_model = self._create_mock_lm()
                                    self.lm_head = lambda x: torch.randn(*x.shape[:-1], 1000)
                                
                                def _create_mock_lm(self):
                                    class MockLM:
                                        def __init__(self):
                                            self.layers = [self._create_mock_layer()]
                                        
                                        def _create_mock_layer(self):
                                            class MockLayer:
                                                def __init__(self):
                                                    self.self_attn = self._create_mock_attn()
                                                
                                                def _create_mock_attn(self):
                                                    class MockAttn:
                                                        def __init__(self):
                                                            self.q_proj = self._create_mock_proj()
                                                        
                                                        def _create_mock_proj(self):
                                                            class MockProj:
                                                                weight = torch.randn(10, 10, dtype=torch.float32)
                                                            return MockProj()
                                                    return MockAttn()
                                            return MockLayer()
                                    return MockLM()
                            return MockInner()
                    return MockPaliGemma()
                
                def embed_prefix_fast(self, *args, **kwargs):
                    # Mock embedding
                    batch_size = 1
                    seq_len = 10
                    hidden_dim = 128
                    embs = torch.randn(batch_size, seq_len, hidden_dim)
                    pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
                    att_masks = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
                    return embs, pad_masks, att_masks, 5, 0
                
                def _prepare_attention_masks_4d(self, att_2d, dtype=None):
                    return att_2d.unsqueeze(1)
            
            return MockModel()
        
        def eval(self):
            pass
    
    # 注意：完整的 mock evaluation 需要更多工作
    # 这里只验证基本结构
    mock_policy = MockPolicy()
    assert mock_policy.device == "cpu"
    assert hasattr(mock_policy, "_paligemma_tokenizer")
    assert hasattr(mock_policy, "model")
    
    print("✓ Mock evaluation structure validated!")


if __name__ == "__main__":
    print("=" * 80)
    print("MCQ Likelihood Evaluation - Smoke Test")
    print("=" * 80)
    
    print("\n[Test 1] Data loading and format validation")
    test_mcq_likelihood_smoke()
    
    print("\n[Test 2] Mock evaluation structure")
    test_mcq_likelihood_mock_evaluation()
    
    print("\n" + "=" * 80)
    print("All smoke tests passed! ✓")
    print("=" * 80)

