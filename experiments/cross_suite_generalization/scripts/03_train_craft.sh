#!/bin/bash

################################################################################
# 步骤 3: 训练 CRaFT
################################################################################
#
# 使用 CRaFT 约束在 libero_spatial 上微调
#
################################################################################

set -e

echo "=========================================="
echo "训练 CRaFT"
echo "=========================================="

# 配置参数
POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
ENV_TYPE="libero"
ENV_TASK="libero_spatial"  # 通过 env.task 过滤数据集
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集，通过 env.task 自动过滤
OUTPUT_DIR="experiments/cross_suite_generalization/outputs/craft_spatial"
ANCHOR_CACHE_DIR="experiments/cross_suite_generalization/outputs/anchor_cache"
STEPS=10000
BATCH_SIZE=32
EVAL_FREQ=2000
SAVE_FREQ=2000
LOG_FREQ=100
SEED=42

# CRaFT 超参数
CRAFT_ENABLED=true
RETENTION_FREQ=5
INITIAL_LAMBDA=1.0
LAMBDA_LR=0.001
EPSILON_START=1.0
EPSILON_END=0.05
USE_GRAD_PROJECTION=true
CONFLICT_THRESHOLD=-0.1

echo "配置信息："
echo "  Policy: ${POLICY_PATH}"
echo "  Environment: ${ENV_TYPE} / ${ENV_TASK}"
echo "  Dataset: ${DATASET_REPO_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Anchor Cache: ${ANCHOR_CACHE_DIR}"
echo "  Steps: ${STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo ""
echo "CRaFT 超参数："
echo "  Enabled: ${CRAFT_ENABLED}"
echo "  Retention Freq: ${RETENTION_FREQ}"
echo "  Initial Lambda: ${INITIAL_LAMBDA}"
echo "  Lambda LR: ${LAMBDA_LR}"
echo "  Epsilon: ${EPSILON_START} -> ${EPSILON_END}"
echo "  Grad Projection: ${USE_GRAD_PROJECTION}"
echo ""

# 检查 Anchor Cache
if [ ! -d "${ANCHOR_CACHE_DIR}" ] || [ ! -f "${ANCHOR_CACHE_DIR}/metadata.json" ]; then
    echo "错误: Anchor Cache 未找到，请先运行步骤 2"
    exit 1
fi

# 检查输出目录
if [ -d "${OUTPUT_DIR}/checkpoints/010000" ]; then
    echo "警告: 检测到已存在的 checkpoint，将覆盖"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消训练"
        exit 1
    fi
fi

# 开始训练
echo "开始训练 CRaFT..."
echo ""

python -m lerobot.scripts.lerobot_train_craft \
    --policy.path="${POLICY_PATH}" \
    --policy.repo_id="local/pi0fast_craft" \
    --env.type="${ENV_TYPE}" \
    --env.task="${ENV_TASK}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --eval_freq="${EVAL_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --seed="${SEED}" \
    --num_workers=4 \
    --save_checkpoint=true \
    --rename_map '{"observation.images.agentview_rgb": "observation.images.image", "observation.images.eye_in_hand_rgb": "observation.images.image2"}' \
    craft.enabled="${CRAFT_ENABLED}" \
    craft.anchor_cache_dir="${ANCHOR_CACHE_DIR}" \
    craft.retention_freq="${RETENTION_FREQ}" \
    craft.initial_lambda="${INITIAL_LAMBDA}" \
    craft.lambda_lr="${LAMBDA_LR}" \
    craft.epsilon_start="${EPSILON_START}" \
    craft.epsilon_end="${EPSILON_END}" \
    craft.use_grad_projection="${USE_GRAD_PROJECTION}" \
    craft.conflict_threshold="${CONFLICT_THRESHOLD}"

echo ""
echo "=========================================="
echo "CRaFT 训练完成！"
echo "=========================================="
echo ""
echo "Checkpoint 位置: ${OUTPUT_DIR}/checkpoints/010000/"
echo ""

# 验证 checkpoint 存在
if [ ! -d "${OUTPUT_DIR}/checkpoints/010000/pretrained_model" ]; then
    echo "错误: Checkpoint 未找到"
    exit 1
fi

echo "✓ Checkpoint 验证通过"
echo ""

