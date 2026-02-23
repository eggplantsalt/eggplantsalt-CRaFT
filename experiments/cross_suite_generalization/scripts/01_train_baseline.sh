#!/bin/bash

################################################################################
# 步骤 1: 训练 Baseline (Naive SFT)
################################################################################
#
# 在 libero_spatial 上进行标准微调，不使用任何 CRaFT 约束
#
################################################################################

set -e

echo "=========================================="
echo "训练 Baseline (Naive SFT)"
echo "=========================================="

# 配置参数
POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
ENV_TYPE="libero"
ENV_TASK="libero_spatial"  # 通过 env.task 过滤数据集
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集，通过 env.task 自动过滤
OUTPUT_DIR="experiments/cross_suite_generalization/outputs/baseline_spatial"
STEPS=10000
BATCH_SIZE=32
SAVE_FREQ=2000
LOG_FREQ=100
SEED=42

echo "配置信息："
echo "  Policy: ${POLICY_PATH}"
echo "  Environment: ${ENV_TYPE} / ${ENV_TASK}"
echo "  Dataset: ${DATASET_REPO_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Steps: ${STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  注意: 训练期间禁用在线评估（避免 Headless 服务器卡死）"
echo ""

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
echo "开始训练 Baseline..."
echo ""

# 设置 Headless 渲染环境变量（使用 OSMesa 作为备用方案）
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

python -m lerobot.scripts.lerobot_train \
    --policy.path="${POLICY_PATH}" \
    --policy.repo_id="local/pi0fast_baseline" \
    --env.type="${ENV_TYPE}" \
    --env.task="${ENV_TASK}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.use_imagenet_stats=false \
    --rename_map '{"observation.images.agentview_rgb": "observation.images.image", "observation.images.eye_in_hand_rgb": "observation.images.image2"}' \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --eval_freq=0 \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --seed="${SEED}" \
    --num_workers=4 \
    --save_checkpoint=true

echo ""
echo "=========================================="
echo "Baseline 训练完成！"
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

