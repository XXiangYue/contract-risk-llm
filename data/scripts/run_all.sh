#!/bin/bash
# 数据处理一键完成脚本
# 从原始数据到训练数据集的完整流程

set -e

echo "=========================================="
echo "合同合规审查大模型 - 数据处理流程"
echo "=========================================="

# 配置
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd ${PROJECT_ROOT}

# 1. 下载开源数据集
echo ""
echo "步骤1: 下载开源数据集..."
echo "=========================================="
if [ -d "data/download" ]; then
    python data/download/download_open_source.py --help > /dev/null 2>&1 && \
    python data/download/download_open_source.py \
        --datasets cail disc-law chatlaw cuad \
        --output ./data/download/ || \
    echo "跳过：开源数据集下载脚本需要配置"
else
    mkdir -p data/download
fi

# 2. 爬取招投标数据
echo ""
echo "步骤2: 爬取招投标数据..."
echo "=========================================="
if [ -d "data/scripts/crawler" ]; then
    python data/scripts/crawler/chinabidding_crawler.py --help > /dev/null 2>&1 && \
    echo "请手动运行爬虫脚本：python data/scripts/crawler/chinabidding_crawler.py" || \
    echo "跳过"
fi

# 3. PDF解析
echo ""
echo "步骤3: PDF解析..."
echo "=========================================="
if [ -d "data/raw/pdfs" ] && [ -n "$(ls -A data/raw/pdfs 2>/dev/null)" ]; then
    python data/scripts/parser/pdf_parser.py \
        --input ./data/raw/pdfs \
        --output ./data/processed/parsed
else
    echo "跳过：无PDF文件"
fi

# 4. 数据清洗
echo ""
echo "步骤4: 数据清洗..."
echo "=========================================="
if [ -d "data/processed/parsed" ] && [ -n "$(ls -A data/processed/parsed 2>/dev/null)" ]; then
    python data/scripts/cleaner/data_cleaner.py \
        --input ./data/processed/parsed \
        --output ./data/processed/cleaned
else
    echo "跳过：无解析数据"
fi

# 5. 数据增强（需要API Key）
echo ""
echo "步骤5: 数据增强..."
echo "=========================================="
if [ -n "$DEEPSEEK_API_KEY" ]; then
    if [ -d "data/processed/cleaned" ]; then
        python data/scripts/augmentation/augmentation.py \
            --api-key $DEEPSEEK_API_KEY \
            --input ./data/processed/cleaned/contracts.json \
            --output ./data/processed/augmented.json \
            --samples 2
    fi
else
    echo "跳过：未设置 DEEPSEEK_API_KEY 环境变量"
    echo "设置方式: export DEEPSEEK_API_KEY=your_key"
fi

# 6. 数据集整合
echo ""
echo "步骤6: 数据集整合..."
echo "=========================================="
python data/scripts/prepare_dataset.py \
    --output ./data/processed/contract_sft

echo ""
echo "=========================================="
echo "✅ 数据处理完成！"
echo "=========================================="
echo "训练数据: ./data/processed/contract_sft/train.json"
echo "验证数据: ./data/processed/contract_sft/val.json"
echo ""
echo "下一步：开始训练"
echo "  bash train/scripts/single_gpu_train.sh"
echo "=========================================="
