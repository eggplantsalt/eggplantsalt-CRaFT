#!/bin/bash

# CRaFT 训练 Dry-Run 脚本
# 用于快速验证训练流程（只运行 3 步）
# 用法: bash scripts/train_craft_dryrun.sh

echo "=========================================="
echo "CRaFT Dry-Run (3 steps)"
echo "=========================================="

python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id="lerobot/aloha_sim_insertion_human" \
    --policy.path="lerobot/pi0_fast" \
    --output_dir="outputs/craft_dryrun" \
    --steps=3 \
    --batch_size=2 \
    --eval_freq=0 \
    --log_freq=1 \
    --save_checkpoint=false \
    --num_workers=0 \
    --wandb.enable=false

echo ""
echo "=========================================="
echo "Dry-Run 完成！"
echo "=========================================="

