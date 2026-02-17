#!/usr/bin/env python

"""
MCQ Likelihood Evaluation Script for Pi0Fast

评测多选题（Multiple Choice Question）的答案概率，不使用 generate，仅用 forward logits。

用法:
    python -m lerobot.scripts.eval_mcq_likelihood \
        --checkpoint_path=outputs/model_checkpoint \
        --data_jsonl=data/mcq_test.jsonl \
        --max_samples=100 \
        --batch_size=4

JSONL 格式:
    每行一个 JSON 对象:
    {
        "image_path": "path/to/image.jpg",
        "question": "What action should the robot take?",
        "choices": ["pick up the cup", "move left", "stop"],
        "answer_index": 0
    }

输出:
    - Accuracy: 正确率
    - Average Margin: top1 和 top2 选项的 log-likelihood 差值平均值
    - 可选：对比两个 checkpoint 的性能差异
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from lerobot.policies.factory import make_policy


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_jsonl(jsonl_path: Path, max_samples: int = None):
    """加载 JSONL 数据"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def load_image(image_path: str, image_size=(224, 224)):
    """加载并预处理图像"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    
    # 转换为 tensor [C, H, W]，归一化到 [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    return img_tensor


def compute_choice_loglikelihood(
    policy,
    image_tensor,
    question_text,
    choice_text,
    device,
):
    """
    计算单个 choice 的 log-likelihood
    
    使用 teacher forcing：将 choice 作为后缀，计算其 token 的累加 log-probability
    
    参数:
        policy: Pi0FastPolicy 实例
        image_tensor: 图像 tensor [C, H, W]
        question_text: 问题文本
        choice_text: 选项文本
        device: 设备
    
    返回:
        log_likelihood: 该 choice 的 log-likelihood（标量）
    """
    # 构造完整 prompt
    prompt = f"{question_text}\nAnswer: {choice_text}"
    
    # Tokenize
    tokenizer = policy._paligemma_tokenizer
    tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    
    # 创建 attention mask
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    
    # 准备图像输入
    image_batch = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
    images = [image_batch]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device)]
    
    # 添加 BOS token
    bos_token = torch.full(
        (1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
    )
    tokens_with_bos = torch.cat([tokens, bos_token], dim=1)
    masks_with_bos = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.bool, device=device)], dim=1)
    
    # Embed prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = policy.model.embed_prefix_fast(
        images,
        img_masks,
        tokens_with_bos,
        masks_with_bos,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    
    # 转换 dtype
    if policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
    
    # Forward pass
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = policy.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
    
    (prefix_out, _), _ = policy.model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        adarms_cond=[None, None],
    )
    
    # 获取 logits
    lm_head = policy.model.paligemma_with_expert.paligemma.lm_head
    logits = lm_head(prefix_out)  # [1, seq_len, vocab_size]
    
    # 计算 choice tokens 的 log-likelihood
    # 找到 "Answer: " 之后的 tokens
    answer_prefix = "Answer: "
    answer_prefix_tokens = tokenizer.encode(answer_prefix, add_special_tokens=False)
    choice_tokens = tokenizer.encode(choice_text, add_special_tokens=False)
    
    # 定位 choice tokens 在序列中的位置
    # 简化：假设 choice tokens 在最后
    num_choice_tokens = len(choice_tokens)
    
    # logits[:, i] 预测 tokens[:, i+1]
    # 我们需要 logits[:, -(num_choice_tokens+1):-1] 来预测 choice_tokens
    choice_logits = logits[:, -(num_choice_tokens+1):-1, :]  # [1, num_choice_tokens, vocab_size]
    choice_targets = torch.tensor(choice_tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, num_choice_tokens]
    
    # 计算 log probabilities
    log_probs = F.log_softmax(choice_logits, dim=-1)  # [1, num_choice_tokens, vocab_size]
    
    # 提取目标 token 的 log prob
    target_log_probs = log_probs.gather(dim=-1, index=choice_targets.unsqueeze(-1)).squeeze(-1)  # [1, num_choice_tokens]
    
    # 累加得到总 log-likelihood
    log_likelihood = target_log_probs.sum().item()
    
    return log_likelihood


def evaluate_sample(policy, sample, device, image_size=(224, 224)):
    """
    评测单个样本
    
    返回:
        predicted_index: 预测的选项索引
        log_likelihoods: 所有选项的 log-likelihood 列表
        correct: 是否预测正确
        margin: top1 和 top2 的差值
    """
    image_path = sample["image_path"]
    question = sample["question"]
    choices = sample["choices"]
    answer_index = sample["answer_index"]
    
    # 加载图像
    image_tensor = load_image(image_path, image_size)
    
    # 计算每个 choice 的 log-likelihood
    log_likelihoods = []
    for choice in choices:
        log_lik = compute_choice_loglikelihood(
            policy,
            image_tensor,
            question,
            choice,
            device,
        )
        log_likelihoods.append(log_lik)
    
    # 选择 log-likelihood 最大的
    predicted_index = int(np.argmax(log_likelihoods))
    correct = (predicted_index == answer_index)
    
    # 计算 margin (top1 - top2)
    sorted_logliks = sorted(log_likelihoods, reverse=True)
    margin = sorted_logliks[0] - sorted_logliks[1] if len(sorted_logliks) > 1 else 0.0
    
    return predicted_index, log_likelihoods, correct, margin


def evaluate_checkpoint(checkpoint_path, data_jsonl, max_samples, batch_size, device):
    """
    评测单个 checkpoint
    
    注意：当前实现为逐样本评测（batch_size 参数保留用于未来优化）
    """
    logging.info(f"加载 checkpoint: {checkpoint_path}")
    
    # 加载 policy
    policy = make_policy(
        pretrained_path=checkpoint_path,
        device=device,
    )
    policy.eval()
    
    # 加载数据
    logging.info(f"加载数据: {data_jsonl}")
    samples = load_jsonl(data_jsonl, max_samples)
    logging.info(f"总样本数: {len(samples)}")
    
    # 评测
    results = []
    correct_count = 0
    margins = []
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="评测中"):
            try:
                predicted_index, log_likelihoods, correct, margin = evaluate_sample(
                    policy, sample, device
                )
                
                results.append({
                    "predicted_index": predicted_index,
                    "log_likelihoods": log_likelihoods,
                    "correct": correct,
                    "margin": margin,
                    "answer_index": sample["answer_index"],
                })
                
                if correct:
                    correct_count += 1
                margins.append(margin)
                
            except Exception as e:
                logging.error(f"评测样本失败: {e}")
                logging.error(f"样本: {sample}")
                continue
    
    # 计算指标
    accuracy = correct_count / len(results) if results else 0.0
    avg_margin = np.mean(margins) if margins else 0.0
    
    return {
        "accuracy": accuracy,
        "avg_margin": avg_margin,
        "correct_count": correct_count,
        "total_count": len(results),
        "results": results,
    }


def compare_checkpoints(checkpoint_a, checkpoint_b, data_jsonl, max_samples, batch_size, device):
    """对比两个 checkpoint"""
    logging.info("=" * 80)
    logging.info("对比两个 checkpoint")
    logging.info("=" * 80)
    
    # 评测 checkpoint A
    logging.info(f"\n评测 Checkpoint A: {checkpoint_a}")
    results_a = evaluate_checkpoint(checkpoint_a, data_jsonl, max_samples, batch_size, device)
    
    # 评测 checkpoint B
    logging.info(f"\n评测 Checkpoint B: {checkpoint_b}")
    results_b = evaluate_checkpoint(checkpoint_b, data_jsonl, max_samples, batch_size, device)
    
    # 输出对比结果
    logging.info("\n" + "=" * 80)
    logging.info("对比结果")
    logging.info("=" * 80)
    logging.info(f"Checkpoint A: {checkpoint_a}")
    logging.info(f"  Accuracy: {results_a['accuracy']:.2%}")
    logging.info(f"  Avg Margin: {results_a['avg_margin']:.4f}")
    logging.info(f"  Correct: {results_a['correct_count']}/{results_a['total_count']}")
    
    logging.info(f"\nCheckpoint B: {checkpoint_b}")
    logging.info(f"  Accuracy: {results_b['accuracy']:.2%}")
    logging.info(f"  Avg Margin: {results_b['avg_margin']:.4f}")
    logging.info(f"  Correct: {results_b['correct_count']}/{results_b['total_count']}")
    
    logging.info(f"\n差异:")
    logging.info(f"  Accuracy: {(results_b['accuracy'] - results_a['accuracy']):.2%}")
    logging.info(f"  Avg Margin: {(results_b['avg_margin'] - results_a['avg_margin']):.4f}")
    
    return results_a, results_b


def main():
    parser = argparse.ArgumentParser(description="MCQ Likelihood Evaluation for Pi0Fast")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently unused, for future optimization)")
    parser.add_argument("--checkpoint_path_b", type=str, default=None, help="Optional second checkpoint for comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save detailed results as JSON")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # 检查文件存在
    if not Path(args.data_jsonl).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_jsonl}")
    
    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # 单 checkpoint 评测
    if args.checkpoint_path_b is None:
        logging.info("=" * 80)
        logging.info("MCQ Likelihood Evaluation")
        logging.info("=" * 80)
        
        results = evaluate_checkpoint(
            args.checkpoint_path,
            args.data_jsonl,
            args.max_samples,
            args.batch_size,
            args.device,
        )
        
        # 输出结果
        logging.info("\n" + "=" * 80)
        logging.info("评测结果")
        logging.info("=" * 80)
        logging.info(f"Checkpoint: {args.checkpoint_path}")
        logging.info(f"Accuracy: {results['accuracy']:.2%}")
        logging.info(f"Average Margin (top1 - top2): {results['avg_margin']:.4f}")
        logging.info(f"Correct: {results['correct_count']}/{results['total_count']}")
        
        # 保存详细结果
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"\n详细结果已保存到: {output_path}")
    
    # 对比两个 checkpoint
    else:
        if not Path(args.checkpoint_path_b).exists():
            raise FileNotFoundError(f"Checkpoint B not found: {args.checkpoint_path_b}")
        
        results_a, results_b = compare_checkpoints(
            args.checkpoint_path,
            args.checkpoint_path_b,
            args.data_jsonl,
            args.max_samples,
            args.batch_size,
            args.device,
        )
        
        # 保存对比结果
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_results = {
                "checkpoint_a": {
                    "path": args.checkpoint_path,
                    "results": results_a,
                },
                "checkpoint_b": {
                    "path": args.checkpoint_path_b,
                    "results": results_b,
                },
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            logging.info(f"\n对比结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

