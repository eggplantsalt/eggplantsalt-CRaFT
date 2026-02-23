#!/usr/bin/env python3

"""
快速测试脚本

使用少量数据和步数快速测试整个流程是否正常工作。
预计运行时间: 10-15 分钟
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ 失败: {description}")
        return False
    
    print(f"\n✓ 完成: {description}")
    return True


def main():
    """主函数"""
    
    print("=" * 60)
    print("CRaFT 跨 Suite 泛化实验 - 快速测试")
    print("=" * 60)
    print()
    print("本测试使用少量数据和步数验证流程是否正常。")
    print("预计运行时间: 10-15 分钟")
    print()
    
    input("按 Enter 开始测试...")
    
    experiment_dir = Path("experiments/cross_suite_generalization")
    
    # 创建测试输出目录
    test_output_dir = experiment_dir / "outputs_test"
    test_output_dir.mkdir(exist_ok=True)
    
    # 步骤 1: 训练 Baseline (100 步)
    cmd = f"""
    python -m lerobot.scripts.lerobot_train \
        --policy.path=lerobot/pi0_fast \
        --env.type=libero \
        --env.task=libero_spatial \
        --dataset.repo_id=lerobot/libero_spatial_no_noops \
        --output_dir={test_output_dir}/baseline_spatial \
        --steps=100 \
        --batch_size=8 \
        --eval_freq=50 \
        --save_freq=100 \
        --log_freq=10 \
        --seed=42
    """
    
    if not run_command(cmd, "步骤 1: 训练 Baseline (100 步)"):
        return 1
    
    # 步骤 2: 构建 Anchor Cache (50 样本)
    cmd = f"""
    python -m lerobot.scripts.build_anchor_hidden_cache \
        --teacher_policy_path=lerobot/pi0_fast \
        --dataset_repo_id=lerobot/libero_spatial_no_noops \
        --output_dir={test_output_dir}/anchor_cache \
        --num_samples=50 \
        --hidden_layer=-2 \
        --pooling=mean_image_tokens \
        --dtype=float16 \
        --shard_size=25 \
        --seed=42
    """
    
    if not run_command(cmd, "步骤 2: 构建 Anchor Cache (50 样本)"):
        return 1
    
    # 步骤 3: 训练 CRaFT (100 步)
    cmd = f"""
    python -m lerobot.scripts.lerobot_train_craft \
        --policy.path=lerobot/pi0_fast \
        --env.type=libero \
        --env.task=libero_spatial \
        --dataset.repo_id=lerobot/libero_spatial_no_noops \
        --output_dir={test_output_dir}/craft_spatial \
        --steps=100 \
        --batch_size=8 \
        --eval_freq=50 \
        --save_freq=100 \
        --log_freq=10 \
        --seed=42 \
        craft.enabled=true \
        craft.anchor_cache_dir={test_output_dir}/anchor_cache \
        craft.retention_freq=5 \
        craft.initial_lambda=1.0 \
        craft.lambda_lr=0.001 \
        craft.epsilon_start=1.0 \
        craft.epsilon_end=0.5
    """
    
    if not run_command(cmd, "步骤 3: 训练 CRaFT (100 步)"):
        return 1
    
    # 步骤 4: 评测 (仅 libero_spatial, 5 episodes)
    baseline_checkpoint = f"{test_output_dir}/baseline_spatial/checkpoints/000100/pretrained_model"
    craft_checkpoint = f"{test_output_dir}/craft_spatial/checkpoints/000100/pretrained_model"
    
    cmd = f"""
    python -m lerobot.scripts.lerobot_eval \
        --policy.path={baseline_checkpoint} \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.n_episodes=5 \
        --eval.batch_size=5 \
        --policy.device=cuda \
        --output_dir={test_output_dir}/results/baseline_test
    """
    
    if not run_command(cmd, "步骤 4a: 评测 Baseline"):
        return 1
    
    cmd = f"""
    python -m lerobot.scripts.lerobot_eval \
        --policy.path={craft_checkpoint} \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.n_episodes=5 \
        --eval.batch_size=5 \
        --policy.device=cuda \
        --output_dir={test_output_dir}/results/craft_test
    """
    
    if not run_command(cmd, "步骤 4b: 评测 CRaFT"):
        return 1
    
    print("\n" + "=" * 60)
    print("✓ 快速测试完成！")
    print("=" * 60)
    print()
    print("所有步骤都正常工作，可以运行完整实验。")
    print()
    print("测试输出位置:")
    print(f"  {test_output_dir}")
    print()
    print("运行完整实验:")
    print("  bash experiments/cross_suite_generalization/run_full_experiment.sh")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

