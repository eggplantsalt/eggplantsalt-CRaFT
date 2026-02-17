#!/bin/bash

# CRaFT Hidden Retention Dry-Run 测试脚本
# 用法: bash scripts/train_craft_hidden_dryrun.sh

echo "=========================================="
echo "CRaFT Hidden Retention Dry-Run (3 steps)"
echo "=========================================="

python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --policy.path=lerobot/pi0_fast \
    --output_dir=outputs/craft_hidden_dryrun \
    --steps=3 \
    --batch_size=2 \
    --eval_freq=0 \
    --log_freq=1 \
    --save_checkpoint=false \
    --num_workers=0 \
    --wandb.enable=false \
    craft.enabled=true \
    craft.retention_mode=hidden \
    craft.anchor_cache_dir=data/anchor_hidden_cache \
    craft.anchor_batch_size=2 \
    craft.retention_freq=1 \
    craft.initial_lambda=1.0 \
    craft.lambda_lr=0.01 \
    craft.epsilon_start=1.0 \
    craft.epsilon_end=0.5

echo ""
echo "=========================================="
echo "Dry-Run 完成！"
echo ""
echo "预期日志输出："
echo "  - Retention Mode: hidden"
echo "  - ✓ AnchorCache 加载成功"
echo "  - mode=hidden | L_ret=X.XXX"
echo "  - λ=X.XXX | ε=X.XXX"
echo "  - dot=X.XXX | cos=X.XXX"
echo "=========================================="

