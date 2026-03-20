#!/bin/bash
# vLLM多节点部署脚本
# 合同合规审查与风险预警大模型

set -e

# 配置
MODEL_PATH=${MODEL_PATH:-"outputs/qwen2.5_7b_lora_contract"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B"}
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}
GPUS=${GPUS:-4}

# vLLM参数
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${GPUS}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQ=${MAX_NUM_SEQ:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# 多节点配置
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "vLLM多节点部署 - 合同合规审查大模型"
echo "=========================================="
echo "基础模型: ${BASE_MODEL}"
echo "LoRA模型: ${MODEL_PATH}"
echo "GPU数量: ${GPUS}"
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "服务地址: ${HOST}:${PORT}"
echo "=========================================="

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动vLLM
python -m vllm.entrypoints.openapi_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seq ${MAX_NUM_SEQ} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --host ${HOST} \
    --port ${PORT} \
    --dtype half \
    --enforce-eager \
    --distributed-init-address ${MASTER_ADDR}:${MASTER_PORT}

echo "=========================================="
echo "部署完成! 服务地址: http://${HOST}:${PORT}"
echo "=========================================="
