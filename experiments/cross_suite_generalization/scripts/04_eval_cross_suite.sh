#!/bin/bash

################################################################################
# 步骤 4: 跨 Suite 评测
################################################################################
#
# 在 4 个不同的 Suites 上评测 Baseline 和 CRaFT
# - libero_spatial (ID)
# - libero_object (OOD)
# - libero_goal (OOD)
# - libero_10 (OOD)
#
################################################################################

set -e

echo "=========================================="
echo "跨 Suite 评测"
echo "=========================================="

# 配置参数
BASELINE_CHECKPOINT="experiments/cross_suite_generalization/outputs/baseline_spatial/checkpoints/010000/pretrained_model"
CRAFT_CHECKPOINT="experiments/cross_suite_generalization/outputs/craft_spatial/checkpoints/010000/pretrained_model"
RESULTS_DIR="experiments/cross_suite_generalization/results"
N_EPISODES=50
BATCH_SIZE=10
DEVICE="cuda"

# 测试的 Suites
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

echo "配置信息："
echo "  Baseline Checkpoint: ${BASELINE_CHECKPOINT}"
echo "  CRaFT Checkpoint: ${CRAFT_CHECKPOINT}"
echo "  Results Dir: ${RESULTS_DIR}"
echo "  Episodes per Suite: ${N_EPISODES}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Device: ${DEVICE}"
echo ""

# 检查 checkpoints
if [ ! -d "${BASELINE_CHECKPOINT}" ]; then
    echo "错误: Baseline checkpoint 未找到"
    exit 1
fi

if [ ! -d "${CRAFT_CHECKPOINT}" ]; then
    echo "错误: CRaFT checkpoint 未找到"
    exit 1
fi

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

################################################################################
# 评测 Baseline
################################################################################

echo "=========================================="
echo "评测 Baseline"
echo "=========================================="
echo ""

for suite in "${SUITES[@]}"; do
    echo "----------------------------------------"
    echo "评测 Baseline on ${suite}"
    echo "----------------------------------------"
    
    OUTPUT_DIR="${RESULTS_DIR}/baseline_spatial_on_${suite}"
    
    python -m lerobot.scripts.lerobot_eval \
        --policy.path="${BASELINE_CHECKPOINT}" \
        --env.type=libero \
        --env.task="${suite}" \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.batch_size="${BATCH_SIZE}" \
        --policy.device="${DEVICE}" \
        --output_dir="${OUTPUT_DIR}" \
        --seed=42
    
    echo ""
    echo "✓ Baseline on ${suite} 完成"
    echo ""
done

################################################################################
# 评测 CRaFT
################################################################################

echo "=========================================="
echo "评测 CRaFT"
echo "=========================================="
echo ""

for suite in "${SUITES[@]}"; do
    echo "----------------------------------------"
    echo "评测 CRaFT on ${suite}"
    echo "----------------------------------------"
    
    OUTPUT_DIR="${RESULTS_DIR}/craft_spatial_on_${suite}"
    
    python -m lerobot.scripts.lerobot_eval \
        --policy.path="${CRAFT_CHECKPOINT}" \
        --env.type=libero \
        --env.task="${suite}" \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.batch_size="${BATCH_SIZE}" \
        --policy.device="${DEVICE}" \
        --output_dir="${OUTPUT_DIR}" \
        --seed=42
    
    echo ""
    echo "✓ CRaFT on ${suite} 完成"
    echo ""
done

################################################################################
# 评测完成
################################################################################

echo "=========================================="
echo "跨 Suite 评测完成！"
echo "=========================================="
echo ""
echo "结果位置: ${RESULTS_DIR}"
echo ""

# 验证所有结果文件存在
echo "验证结果文件..."
ALL_FOUND=true

for suite in "${SUITES[@]}"; do
    BASELINE_RESULT="${RESULTS_DIR}/baseline_spatial_on_${suite}/eval_info.json"
    CRAFT_RESULT="${RESULTS_DIR}/craft_spatial_on_${suite}/eval_info.json"
    
    if [ ! -f "${BASELINE_RESULT}" ]; then
        echo "✗ 缺失: ${BASELINE_RESULT}"
        ALL_FOUND=false
    else
        echo "✓ 找到: baseline_spatial_on_${suite}/eval_info.json"
    fi
    
    if [ ! -f "${CRAFT_RESULT}" ]; then
        echo "✗ 缺失: ${CRAFT_RESULT}"
        ALL_FOUND=false
    else
        echo "✓ 找到: craft_spatial_on_${suite}/eval_info.json"
    fi
done

echo ""

if [ "${ALL_FOUND}" = true ]; then
    echo "✓ 所有结果文件验证通过"
else
    echo "✗ 部分结果文件缺失"
    exit 1
fi

echo ""

