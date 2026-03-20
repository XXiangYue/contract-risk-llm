# 【智审】企业级长文本招投标文件及合同合规性审查与风险预警大模型

基于 Qwen2.5-7B 的企业合同合规审查大模型项目。

## 项目概述

利用大模型进行长文本的深度推理、信息抽取和逻辑比对，将合同审查时间从"天"级别压缩到"分钟"级别。

## 技术栈

- **基础模型**: Qwen2.5-7B
- **训练框架**: Axolotl + DeepSpeed + FlashAttention-2
- **微调技术**: LoRA / QLoRA
- **推理部署**: vLLM (高吞吐批量处理)

## 项目结构

```
合同合规审查与风险预警大模型/
├── data/                      # 数据处理
│   ├── download/              # 开源数据集下载
│   │   └── download_open_source.py
│   ├── scripts/
│   │   ├── crawler/           # 招投标爬虫
│   │   │   └── chinabidding_crawler.py
│   │   ├── parser/            # PDF解析
│   │   │   └── pdf_parser.py
│   │   ├── cleaner/           # 数据清洗
│   │   │   └── data_cleaner.py
│   │   └── augmentation/      # 数据增强
│   │       └── augmentation.py
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
├── train/                     # 训练模块
│   ├── configs/               # Axolotl配置
│   │   ├── qwen2.5_7b_lora.yaml
│   │   └── deepspeed.json
│   └── scripts/               # 训练脚本
│       ├── single_gpu_train.sh
│       └── multi_gpu_train.sh
├── deploy/                    # 部署模块
│   ├── scripts/               # 部署脚本
│   │   ├── single_node_deploy.sh
│   │   └── multi_node_deploy.sh
│   └── api/                   # API服务
│       └── main.py
└── docs/                      # 文档
    └── ISSUES.md
```

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repo-url>
cd 合同合规审查与风险预警大模型

# 创建虚拟环境
conda create -n contract python=3.10
conda activate contract

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据处理

#### 2.1 下载开源数据集

```bash
python data/download/download_open_source.py \
    --datasets cail disc-law chatlaw cuad \
    --output ./data/download/
```

#### 2.2 爬取招投标文件

```bash
python data/scripts/crawler/chinabidding_crawler.py \
    --keyword "采购" \
    --pages 10 \
    --output ./data/raw/
```

#### 2.3 PDF解析

```bash
python data/scripts/parser/pdf_parser.py \
    --input ./data/raw/pdfs \
    --output ./data/processed/
```

#### 2.4 数据清洗

```bash
python data/scripts/cleaner/data_cleaner.py \
    --input ./data/processed/parsed \
    --output ./data/processed/cleaned/
```

#### 2.5 数据增强

```bash
python data/scripts/augmentation/augmentation.py \
    --api-key YOUR_API_KEY \
    --input ./data/processed/cleaned/contracts.json \
    --output ./data/processed/augmented.json \
    --samples 3
```

#### 2.6 数据集整合

将所有数据源整合为训练格式：

```bash
python data/scripts/prepare_dataset.py \
    --output ./data/processed/contract_sft
```

支持的输出：
- 开源数据集：CAIL、DISC-LawLLM、ChatLaw、CUAD
- 爬取数据：data/processed/cleaned/
- 增强数据：data/processed/augmented/

输出格式：
- `train.json` - 训练集 (90%)
- `val.json` - 验证集 (10%)
- `all.json` - 全部数据

### 3. 模型训练

#### 单GPU训练

```bash
bash train/scripts/single_gpu_train.sh
```

#### 多GPU训练

```bash
GPUS=4 bash train/scripts/multi_gpu_train.sh
```

### 4. 模型部署

#### 单节点部署

```bash
bash deploy/scripts/single_node_deploy.sh
```

#### 多节点部署

```bash
GPUS=4 bash deploy/scripts/multi_node_deploy.sh
```

#### 启动API服务

```bash
python deploy/api/main.py
```

## API使用

### 合同审查

```bash
curl -X POST http://localhost:8080/api/contract/review \
  -H "Content-Type: application/json" \
  -d '{
    "contract_text": "合同内容..."
  }'
```

### 批量审查

```bash
curl -X POST http://localhost:8080/api/contract/batch-review \
  -H "Content-Type: application/json" \
  -d '{
    "contracts": ["合同1...", "合同2..."]
  }'
```

## 配置说明

### 训练配置

主要配置项在 `train/configs/qwen2.5_7b_lora.yaml`：

- `sequence_len`: 序列长度 (默认8192)
- `lora_config`: LoRA参数配置
- `deepspeed`: DeepSpeed优化配置

### 部署配置

主要配置在 `deploy/scripts/single_node_deploy.sh`：

- `MODEL_PATH`: 模型路径
- `TENSOR_PARALLEL_SIZE`: Tensor并行数
- `MAX_MODEL_LEN`: 最大序列长度

## 数据集说明

### 开源数据集

- **CAIL**: 中国法研杯司法数据集
- **DISC-LawLLM**: 复旦大学法律大模型数据集
- **ChatLaw**: 北大法律大模型数据集
- **CUAD**: 合同理解Atticus数据集

### 数据增强策略

使用高阶模型进行逆向工程，生成含风险的Negative样本：
- 违约金比例异常
- 管辖法院偏远
- 付款条件苛刻
- 违约责任不对等
- 保密义务过重

## 常见问题

见 [ISSUES.md](docs/ISSUES.md)

## License

MIT License
