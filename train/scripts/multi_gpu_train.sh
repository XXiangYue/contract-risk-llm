#!/bin/bash
# 多GPU训练脚本
# 合同合规审查与风险预警大模型

set -e

# 配置
MODEL_NAME="Qwen/Qwen2.5-7B"
OUTPUT_DIR="outputs/qwen2.5_7b_lora_contract"
CONFIG_PATH="train/configs/qwen2.5_7b_lora.yaml"
GPUS=${GPUS:-4}

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "开始多GPU训练 - 合同合规审查大模型"
echo "=========================================="
echo "模型: ${MODEL_NAME}"
echo "GPU数量: ${GPUS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "配置文件: ${CONFIG_PATH}"
echo "=========================================="

# 检查数据目录
if [ ! -d "data/processed/contract_sft" ]; then
    echo "警告: 训练数据目录不存在，请先运行数据处理脚本"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p outputs/logs

# 使用DeepSpeed多节点训练
accelerate launch \
    --num_processes ${GPUS} \
    --mixed_precision bf16 \
    --deepspeed train/configs/deepspeed.json \
    -m axolotl.cli.train \
    ${CONFIG_PATH}

echo "=========================================="
echo "训练完成!"
echo "模型保存在: ${OUTPUT_DIR}"
echo "=========================================="
