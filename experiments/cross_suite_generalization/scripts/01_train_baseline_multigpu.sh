#!/bin/bash

################################################################################
# 步骤 1: 训练 Baseline (Naive SFT) - 多 GPU 版本
################################################################################
#
# 在 libero_spatial 上进行标准微调，不使用任何 CRaFT 约束
# 支持多 GPU 并行训练（使用 accelerate）
#
################################################################################

set -e

echo "=========================================="
echo "训练 Baseline (Naive SFT) - 多 GPU"
echo "=========================================="

# 配置参数
POLICY_PATH="lerobot/pi0fast-base"  # 官方预训练模型
ENV_TYPE="libero"
ENV_TASK="libero_spatial"  # 通过 env.task 过滤数据集
DATASET_REPO_ID="lerobot/libero"  # 使用全量数据集，通过 env.task 自动过滤
OUTPUT_DIR="experiments/cross_suite_generalization/outputs/baseline_spatial"
STEPS=10000
SAVE_FREQ=2000
LOG_FREQ=100
SEED=42

# 多 GPU 配置
NUM_GPUS=8  # V100 数量
BATCH_SIZE_PER_GPU=4  # 每个 GPU 的批次大小
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE_PER_GPU))  # 总批次大小 = 32

echo "配置信息："
echo "  Policy: ${POLICY_PATH}"
echo "  Environment: ${ENV_TYPE} / ${ENV_TASK}"
echo "  Dataset: ${DATASET_REPO_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Steps: ${STEPS}"
echo "  Total Batch Size: ${TOTAL_BATCH_SIZE} (${NUM_GPUS} GPUs × ${BATCH_SIZE_PER_GPU})"
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

# 检查 accelerate 配置
if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
    echo "警告: 未找到 accelerate 配置，将自动生成"
    echo ""
    
    # 创建 accelerate 配置目录
    mkdir -p "$HOME/.cache/huggingface/accelerate"
    
    # 生成多 GPU 配置
    cat > "$HOME/.cache/huggingface/accelerate/default_config.yaml" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    
    echo "✓ 已生成 accelerate 配置"
    echo ""
fi

# 开始训练
echo "开始训练 Baseline (使用 ${NUM_GPUS} 个 GPU)..."
echo ""

# 设置 Headless 渲染环境变量（使用 OSMesa 作为备用方案）
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# 使用 accelerate launch 启动多 GPU 训练
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --multi_gpu \
    -m lerobot.scripts.lerobot_train \
    --policy.path="${POLICY_PATH}" \
    --policy.repo_id="local/pi0fast_baseline" \
    --env.type="${ENV_TYPE}" \
    --env.task="${ENV_TASK}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.use_imagenet_stats=false \
    --rename_map '{"observation.images.agentview_rgb": "observation.images.image", "observation.images.eye_in_hand_rgb": "observation.images.image2"}' \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE_PER_GPU}" \
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

