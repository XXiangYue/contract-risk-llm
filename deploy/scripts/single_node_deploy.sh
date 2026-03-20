#!/bin/bash
# vLLM单节点部署脚本
# 合同合规审查与风险预警大模型

set -e

# 配置
MODEL_PATH=${MODEL_PATH:-"outputs/qwen2.5_7b_lora_contract"}  # LoRA适配器路径或基础模型
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B"}  # 基础模型
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}
GPUS=${GPUS:-1}

# vLLM参数
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${GPUS}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQ=${MAX_NUM_SEQ:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

echo "=========================================="
echo "vLLM模型部署 - 合同合规审查大模型"
echo "=========================================="
echo "基础模型: ${BASE_MODEL}"
echo "LoRA模型: ${MODEL_PATH}"
echo "GPU数量: ${GPUS}"
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "服务地址: ${HOST}:${PORT}"
echo "最大序列长度: ${MAX_MODEL_LEN}"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "${MODEL_PATH}" ] && [ ! -d "${BASE_MODEL}" ]; then
    echo "错误: 模型目录不存在: ${MODEL_PATH} 或 ${BASE_MODEL}"
    exit 1
fi

# 启动vLLM服务器
# 如果是LoRA微调模型，使用--lora参数
if [ -d "${MODEL_PATH}/adapter_config.json" ]; then
    echo "检测到LoRA适配器，使用LoRA部署模式..."
    python -m vllm.entrypoints.openai.api_server \
        --model ${BASE_MODEL} \
        --lora-modules ${MODEL_PATH} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seq ${MAX_NUM_SEQ} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --host ${HOST} \
        --port ${PORT} \
        --dtype half \
        --enforce-eager
else
    echo "使用基础模型部署..."
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seq ${MAX_NUM_SEQ} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --host ${HOST} \
        --port ${PORT} \
        --dtype half \
        --enforce-eager
fi
