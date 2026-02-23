#!/bin/bash

################################################################################
# 步骤 2: 构建 Anchor Cache
################################################################################
#
# 为 CRaFT 训练生成离线特征锚点缓存
#
################################################################################

set -e

echo "=========================================="
echo "构建 Anchor Cache"
echo "=========================================="

# 配置参数
TEACHER_POLICY_PATH="lerobot/pi0_fast"
DATASET_REPO_ID="lerobot/libero_spatial_no_noops"
OUTPUT_DIR="experiments/cross_suite_generalization/outputs/anchor_cache"
NUM_SAMPLES=1000
HIDDEN_LAYER=-2
POOLING="mean_image_tokens"
DTYPE="float16"
SHARD_SIZE=100
SEED=42

echo "配置信息："
echo "  Teacher Policy: ${TEACHER_POLICY_PATH}"
echo "  Dataset: ${DATASET_REPO_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Hidden Layer: ${HIDDEN_LAYER}"
echo "  Pooling: ${POOLING}"
echo ""

# 检查输出目录
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR})" ]; then
    echo "警告: 输出目录已存在且非空，将覆盖"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消构建"
        exit 1
    fi
    rm -rf "${OUTPUT_DIR}"/*
fi

# 开始构建
echo "开始构建 Anchor Cache..."
echo ""

python -m lerobot.scripts.build_anchor_hidden_cache \
    --teacher_policy_path="${TEACHER_POLICY_PATH}" \
    --dataset_repo_id="${DATASET_REPO_ID}" \
    --output_dir="${OUTPUT_DIR}" \
    --num_samples="${NUM_SAMPLES}" \
    --hidden_layer="${HIDDEN_LAYER}" \
    --pooling="${POOLING}" \
    --dtype="${DTYPE}" \
    --shard_size="${SHARD_SIZE}" \
    --seed="${SEED}"

echo ""
echo "=========================================="
echo "Anchor Cache 构建完成！"
echo "=========================================="
echo ""
echo "Cache 位置: ${OUTPUT_DIR}"
echo ""

# 验证 cache 文件
if [ ! -f "${OUTPUT_DIR}/metadata.json" ]; then
    echo "错误: metadata.json 未找到"
    exit 1
fi

SHARD_COUNT=$(ls -1 "${OUTPUT_DIR}"/shard_*.pt 2>/dev/null | wc -l)
echo "✓ 找到 ${SHARD_COUNT} 个 shard 文件"

if [ "${SHARD_COUNT}" -eq 0 ]; then
    echo "错误: 未找到任何 shard 文件"
    exit 1
fi

echo "✓ Anchor Cache 验证通过"
echo ""

