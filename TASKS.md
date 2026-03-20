# 项目任务拆解

## 项目概述
- 名称：【智审】企业级长文本招投标文件及合同合规性审查与风险预警大模型
- 基础模型：Qwen-2.5-7B
- 训练框架：Axlotl + LoRA
- 部署：vLLM

## 任务拆解

### 1. 项目结构搭建
- 创建模块化目录结构
- 编写README.md

### 2. 数据处理模块 (data/)
- [ ] 2.1 开源数据集下载脚本 (data/scripts/download_open_source.py)
  - CAIL、 DISC-LawLLM、 ChatLaw、 CUAD 下载脚本
- [ ] 2.2 招投标文件爬虫 (data/scripts/crawler/)
  - 中国政府采购网爬虫
  - 公共资源交易平台爬虫
- [ ] 2.3 PDF解析模块 (data/scripts/parser/)
  - 使用MinerU进行版面分析
  - PDF转Markdown
- [ ] 2.4 数据清洗 (data/scripts/cleaner/)
  - 去除页眉页脚、噪声
  - 文本标准化
- [ ] 2.5 数据增强 (data/scripts/augmentation/)
  - 逆向工程：使用高阶模型生成Negative样本
  - 恶意修改：违约金比例、管辖法院、违约条款等

### 3. 训练模块 (train/)
- [ ] 3.1 Axolotl配置文件 (train/configs/)
  - qwen2.5_7b_lora.yaml
- [ ] 3.2 训练脚本 (train/scripts/)
  - single_gpu_train.sh
  - multi_gpu_train.sh
- [ ] 3.3 数据集配置 (train/datasets/)
  - 指令微调格式转换

### 4. 部署模块 (deploy/)
- [ ] 4.1 vLLM部署脚本 (deploy/scripts/)
  - single_node_deploy.sh
  - multi_node_deploy.sh
- [ ] 4.2 API服务 (deploy/api/)
  - FastAPI封装

### 5. 文档与维护
- [ ] 5.1 问题记录 (docs/ISSUES.md)
- [ ] 5.2 环境配置 (requirements.txt)
- [ ] 5.3 使用说明 (README.md)

## 状态
- [x] 进行中

## GitHub仓库
https://github.com/XXiangYue/contract-risk-llm

## 完成进度
- [x] 1. 项目结构搭建
- [x] 2. 开源数据集下载脚本
- [x] 3. 招投标爬虫
- [x] 4. PDF解析模块
- [x] 5. 数据清洗
- [x] 6. 数据增强
- [x] 7. Axolotl训练配置
- [x] 8. vLLM部署脚本
- [x] 9. API服务
- [x] 10. 文档与维护
- [x] 11. GitHub仓库创建
