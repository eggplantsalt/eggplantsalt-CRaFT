#!/bin/bash

# CRaFT 训练启动脚本
# 用法: bash scripts/train_craft.sh

# ============================================================================
# 配置参数
# ============================================================================

# 基础训练配置
DATASET="lerobot/aloha_sim_insertion_human"
POLICY_PATH="lerobot/pi0_fast"
OUTPUT_DIR="outputs/craft_train"
STEPS=1000
BATCH_SIZE=8
EVAL_FREQ=500
LOG_FREQ=50

# CRaFT 配置
CRAFT_ENABLED=true
ANCHOR_CACHE_DIR="data/anchor_cache"  # 由 build_anchor_cache.py 生成
ANCHOR_BATCH_SIZE=8
RETENTION_FREQ=5  # K-step: 每 5 步计算一次保留损失
INITIAL_LAMBDA=1.0
LAMBDA_LR=0.01
LAMBDA_MAX=10.0
EPSILON_START=1.0
EPSILON_END=0.1
USE_GRAD_PROJECTION=true
CONFLICT_THRESHOLD=-0.1
PROJECTION_MODE="weighted"

# ============================================================================
# 训练命令
# ============================================================================

echo "=========================================="
echo "CRaFT 训练"
echo "=========================================="
echo "数据集: $DATASET"
echo "策略: $POLICY_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "训练步数: $STEPS"
echo "CRaFT 启用: $CRAFT_ENABLED"
if [ "$CRAFT_ENABLED" = true ]; then
    echo "AnchorCache: $ANCHOR_CACHE_DIR"
    echo "保留频率: 每 $RETENTION_FREQ 步"
    echo "初始 λ: $INITIAL_LAMBDA"
fi
echo "=========================================="

python -m lerobot.scripts.lerobot_train_craft \
    --dataset.repo_id="$DATASET" \
    --policy.path="$POLICY_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --steps=$STEPS \
    --batch_size=$BATCH_SIZE \
    --eval_freq=$EVAL_FREQ \
    --log_freq=$LOG_FREQ \
    --save_freq=500 \
    --num_workers=4 \
    --wandb.enable=false

# 注意：CRaFT 配置目前通过代码传递
# 未来可以扩展为命令行参数

echo ""
echo "=========================================="
echo "训练完成！"
echo "检查点保存在: $OUTPUT_DIR"
echo "=========================================="

