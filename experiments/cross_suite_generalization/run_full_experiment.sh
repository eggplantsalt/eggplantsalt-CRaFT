#!/bin/bash

################################################################################
# CRaFT 跨 Suite 泛化能力验证 - 完整实验流程
################################################################################
#
# 本脚本自动执行完整的实验流程：
# 1. 训练 Baseline (Naive SFT)
# 2. 构建 Anchor Cache
# 3. 训练 CRaFT
# 4. 跨 Suite 评测（4 个 Suites × 2 个模型 = 8 个评测任务）
# 5. 生成对比报告
#
# 预计运行时间：6-9 小时（取决于硬件）
#
# 使用方法：
#   bash run_full_experiment.sh
#
################################################################################

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 计时函数
start_time=$(date +%s)

print_elapsed_time() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    log_info "总耗时: ${hours}h ${minutes}m ${seconds}s"
}

# 错误处理
trap 'log_error "实验在第 $LINENO 行失败"; print_elapsed_time; exit 1' ERR

################################################################################
# 环境检查
################################################################################

log_info "=========================================="
log_info "步骤 0: 环境检查"
log_info "=========================================="

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    log_error "Python 未安装或不在 PATH 中"
    exit 1
fi

# 检查 CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_success "CUDA 可用"
else
    log_warning "CUDA 不可用，将使用 CPU（速度会很慢）"
fi

# 检查 lerobot 安装
if ! python -c "import lerobot" 2>/dev/null; then
    log_error "lerobot 未安装，请先运行: pip install -e ."
    exit 1
fi

# 检查 LIBERO 安装
if ! python -c "import libero" 2>/dev/null; then
    log_error "LIBERO 未安装，请先运行: pip install libero-robotics"
    exit 1
fi

log_success "环境检查通过"

################################################################################
# 创建输出目录
################################################################################

log_info "创建输出目录..."

EXPERIMENT_DIR="experiments/cross_suite_generalization"
mkdir -p "${EXPERIMENT_DIR}/outputs/baseline_spatial"
mkdir -p "${EXPERIMENT_DIR}/outputs/craft_spatial"
mkdir -p "${EXPERIMENT_DIR}/outputs/anchor_cache"
mkdir -p "${EXPERIMENT_DIR}/results"

log_success "输出目录创建完成"

################################################################################
# 步骤 1: 训练 Baseline
################################################################################

log_info "=========================================="
log_info "步骤 1: 训练 Baseline (Naive SFT)"
log_info "=========================================="

step1_start=$(date +%s)

bash "${EXPERIMENT_DIR}/scripts/01_train_baseline.sh"

step1_end=$(date +%s)
step1_elapsed=$((step1_end - step1_start))
log_success "Baseline 训练完成 (耗时: $((step1_elapsed / 60)) 分钟)"

################################################################################
# 步骤 2: 构建 Anchor Cache
################################################################################

log_info "=========================================="
log_info "步骤 2: 构建 Anchor Cache"
log_info "=========================================="

step2_start=$(date +%s)

bash "${EXPERIMENT_DIR}/scripts/02_build_anchor_cache.sh"

step2_end=$(date +%s)
step2_elapsed=$((step2_end - step2_start))
log_success "Anchor Cache 构建完成 (耗时: $((step2_elapsed / 60)) 分钟)"

################################################################################
# 步骤 3: 训练 CRaFT
################################################################################

log_info "=========================================="
log_info "步骤 3: 训练 CRaFT"
log_info "=========================================="

step3_start=$(date +%s)

bash "${EXPERIMENT_DIR}/scripts/03_train_craft.sh"

step3_end=$(date +%s)
step3_elapsed=$((step3_end - step3_start))
log_success "CRaFT 训练完成 (耗时: $((step3_elapsed / 60)) 分钟)"

################################################################################
# 步骤 4: 跨 Suite 评测
################################################################################

log_info "=========================================="
log_info "步骤 4: 跨 Suite 评测"
log_info "=========================================="

step4_start=$(date +%s)

bash "${EXPERIMENT_DIR}/scripts/04_eval_cross_suite.sh"

step4_end=$(date +%s)
step4_elapsed=$((step4_end - step4_start))
log_success "跨 Suite 评测完成 (耗时: $((step4_elapsed / 60)) 分钟)"

################################################################################
# 步骤 5: 生成对比报告
################################################################################

log_info "=========================================="
log_info "步骤 5: 生成对比报告"
log_info "=========================================="

step5_start=$(date +%s)

python "${EXPERIMENT_DIR}/scripts/05_generate_report.py"

step5_end=$(date +%s)
step5_elapsed=$((step5_end - step5_start))
log_success "对比报告生成完成 (耗时: $((step5_elapsed / 60)) 分钟)"

################################################################################
# 实验完成
################################################################################

log_info "=========================================="
log_success "🎉 实验完成！"
log_info "=========================================="

print_elapsed_time

log_info ""
log_info "结果文件位置："
log_info "  - Baseline Checkpoint: ${EXPERIMENT_DIR}/outputs/baseline_spatial/checkpoints/010000/"
log_info "  - CRaFT Checkpoint: ${EXPERIMENT_DIR}/outputs/craft_spatial/checkpoints/010000/"
log_info "  - 对比报告: ${EXPERIMENT_DIR}/results/comparison_report.md"
log_info "  - 成功率对比图: ${EXPERIMENT_DIR}/results/success_rate_comparison.png"
log_info ""
log_info "查看报告："
log_info "  cat ${EXPERIMENT_DIR}/results/comparison_report.md"
log_info ""

exit 0

