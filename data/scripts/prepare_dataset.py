#!/usr/bin/env python3
"""
数据集整合模块
将开源数据集 + 爬取数据 + 增强数据 整合成统一的训练数据格式

使用方法：
    python prepare_dataset.py --source all --output ./data/processed/contract_sft
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 开源数据集路径
    cail_dir: str = "data/download/CAIL"
    discllaw_dir: str = "data/download/DISC-LawLLM"
    chatlaw_dir: str = "data/download/ChatLaw"
    cuad_dir: str = "data/download/CUAD"
    
    # 爬取数据路径
    crawled_dir: str = "data/processed/cleaned"
    
    # 增强数据路径
    augmented_dir: str = "data/processed/augmented"
    
    # 输出路径
    output_dir: str = "data/processed/contract_sft"


def load_json(path: str) -> List[Dict]:
    """加载JSON文件"""
    if not Path(path).exists():
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'data' in data:
                return data['data']
            else:
                return []
    except Exception as e:
        logger.warning(f"加载失败 {path}: {e}")
        return []


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    if not Path(path).exists():
        return []
    
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        logger.warning(f"加载失败 {path}: {e}")
    
    return data


def convert_to_sft(sample: Dict, source: str) -> Dict:
    """
    转换为指令微调格式
    
    Args:
        sample: 原始样本
        source: 数据来源
    
    Returns:
        SFT格式样本
    """
    # 提取文本
    text = sample.get('text') or sample.get('content') or sample.get('contract_text', '')
    if not text:
        return None
    
    # 提取标签
    label = sample.get('label', 'positive')
    risk_type = sample.get('risk_type', '')
    
    # 构建对话
    if label == 'negative':
        # 风险样本
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的企业合同合规审查专家。"
            },
            {
                "role": "user",
                "content": f"请审查以下合同是否有风险：\n\n{text}"
            },
            {
                "role": "assistant",
                "content": f"⚠️ 风险提示：该合同存在风险！\n\n风险类型：{risk_type}\n\n建议审查并修改相关条款。"
            }
        ]
    else:
        # 正常样本
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的企业合同合规审查专家。"
            },
            {
                "role": "user",
                "content": f"请审查以下合同是否有风险：\n\n{text}"
            },
            {
                "role": "assistant",
                "content": "✅ 合同审查结果：\n\n该合同未发现明显风险点，条款较为合理。"
            }
        ]
    
    return {
        "messages": messages,
        "source": source,
        "label": label,
        "risk_type": risk_type if risk_type else None
    }


def process_cail(data_dir: str) -> List[Dict]:
    """处理CAIL数据集"""
    logger.info("处理 CAIL 数据集...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"CAIL 数据集不存在: {data_dir}")
        return samples
    
    # 查找所有json/jsonl文件
    for file in data_path.rglob("*.json") + data_path.rglob("*.jsonl"):
        data = load_json(str(file)) if str(file).endswith('.json') else load_jsonl(str(file))
        
        for item in data:
            # CAIL格式转换
            if 'fact' in item and 'law' in item:
                # 司法问答格式，转为合同审查
                text = item.get('fact', '')
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'cail')
                if sft_sample:
                    samples.append(sft_sample)
            elif 'question' in item and 'answer' in item:
                text = item.get('question', '') + ' ' + item.get('answer', '')
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'cail')
                if sft_sample:
                    samples.append(sft_sample)
    
    logger.info(f"  CAIL: {len(samples)} 条")
    return samples


def process_disc_law(data_dir: str) -> List[Dict]:
    """处理DISC-LawLLM数据集"""
    logger.info("处理 DISC-LawLLM 数据集...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"DISC-LawLLM 数据集不存在: {data_dir}")
        return samples
    
    for file in data_path.rglob("*.json") + data_path.rglob("*.jsonl"):
        data = load_json(str(file)) if str(file).endswith('.json') else load_jsonl(str(file))
        
        for item in data:
            # 通用格式
            if 'conversation' in item:
                # 对话格式
                text = ' '.join([c.get('content', '') for c in item['conversation']])
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'disc-law')
                if sft_sample:
                    samples.append(sft_sample)
            elif 'input' in item and 'output' in item:
                text = item.get('input', '') + ' ' + item.get('output', '')
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'disc-law')
                if sft_sample:
                    samples.append(sft_sample)
    
    logger.info(f"  DISC-LawLLM: {len(samples)} 条")
    return samples


def process_chatlaw(data_dir: str) -> List[Dict]:
    """处理ChatLaw数据集"""
    logger.info("处理 ChatLaw 数据集...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"ChatLaw 数据集不存在: {data_dir}")
        return samples
    
    for file in data_path.rglob("*.json") + data_path.rglob("*.jsonl"):
        data = load_json(str(file)) if str(file).endswith('.json') else load_jsonl(str(file))
        
        for item in data:
            if 'text' in item:
                sft_sample = convert_to_sft(item, 'chatlaw')
                if sft_sample:
                    samples.append(sft_sample)
    
    logger.info(f"  ChatLaw: {len(samples)} 条")
    return samples


def process_cuad(data_dir: str) -> List[Dict]:
    """处理CUAD数据集"""
    logger.info("处理 CUAD 数据集...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"CUAD 数据集不存在: {data_dir}")
        return samples
    
    for file in data_path.rglob("*.json"):
        data = load_json(str(file))
        
        # CUAD格式：{paragraphs: [...], qas: [...]}
        for item in data.get('paragraphs', []):
            text = item.get('context', '')
            if text:
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'cuad')
                if sft_sample:
                    samples.append(sft_sample)
    
    logger.info(f"  CUAD: {len(samples)} 条")
    return samples


def process_crawled_data(data_dir: str) -> List[Dict]:
    """处理爬取的招投标数据"""
    logger.info("处理爬取数据...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"爬取数据不存在: {data_dir}")
        return samples
    
    # 查找所有txt和json文件
    for file in data_path.rglob("*.txt") + data_path.rglob("*.json"):
        if file.name.startswith('.'):
            continue
        
        try:
            if file.suffix == '.txt':
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                data = load_json(str(file))
                text = data.get('text', '') if isinstance(data, dict) else ''
            
            if text and len(text) > 100:
                sft_sample = convert_to_sft({'text': text, 'label': 'positive'}, 'crawled')
                if sft_sample:
                    samples.append(sft_sample)
        except Exception as e:
            logger.warning(f"处理失败 {file}: {e}")
    
    logger.info(f"  爬取数据: {len(samples)} 条")
    return samples


def process_augmented_data(data_dir: str) -> List[Dict]:
    """处理增强数据"""
    logger.info("处理增强数据...")
    
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"增强数据不存在: {data_dir}")
        return samples
    
    for file in data_path.rglob("*.json"):
        try:
            data = load_json(str(file))
            
            # 支持两种格式：{data: [...]} 或 直接list
            items = data.get('data', []) if isinstance(data, dict) else data
            
            for item in items:
                sft_sample = convert_to_sft(item, 'augmented')
                if sft_sample:
                    samples.append(sft_sample)
        except Exception as e:
            logger.warning(f"处理失败 {file}: {e}")
    
    logger.info(f"  增强数据: {len(samples)} 条")
    return samples


def split_dataset(samples: List[Dict], train_ratio: float = 0.9) -> tuple:
    """划分训练集和验证集"""
    random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    return train_samples, val_samples


def save_dataset(samples: List[Dict], output_path: str):
    """保存数据集"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"保存 {len(samples)} 条到 {output_path}")


def prepare_dataset(config: DatasetConfig):
    """整合所有数据源"""
    
    all_samples = []
    
    # 1. 开源数据集
    if Path(config.cail_dir).exists():
        all_samples.extend(process_cail(config.cail_dir))
    
    if Path(config.discllaw_dir).exists():
        all_samples.extend(process_disc_law(config.discllaw_dir))
    
    if Path(config.chatlaw_dir).exists():
        all_samples.extend(process_chatlaw(config.chatlaw_dir))
    
    if Path(config.cuad_dir).exists():
        all_samples.extend(process_cuad(config.cuad_dir))
    
    # 2. 爬取数据
    if Path(config.crawled_dir).exists():
        all_samples.extend(process_crawled_data(config.crawled_dir))
    
    # 3. 增强数据
    if Path(config.augmented_dir).exists():
        all_samples.extend(process_augmented_data(config.augmented_dir))
    
    if not all_samples:
        logger.error("没有找到任何数据！请先运行数据下载和爬取脚本")
        return
    
    logger.info(f"\n总计: {len(all_samples)} 条样本")
    
    # 统计
    labels = {}
    for s in all_samples:
        label = s.get('label', 'unknown')
        labels[label] = labels.get(label, 0) + 1
    
    logger.info(f"标签分布: {labels}")
    
    # 划分
    train_samples, val_samples = split_dataset(all_samples)
    
    logger.info(f"\n训练集: {len(train_samples)} 条")
    logger.info(f"验证集: {len(val_samples)} 条")
    
    # 保存
    save_dataset(train_samples, f"{config.output_dir}/train.json")
    save_dataset(val_samples, f"{config.output_dir}/val.json")
    
    # 同时保存合并文件
    save_dataset(all_samples, f"{config.output_dir}/all.json")
    
    logger.info("\n✅ 数据集整合完成！")
    logger.info(f"训练数据路径: {config.output_dir}/train.json")
    logger.info(f"验证数据路径: {config.output_dir}/val.json")


def main():
    parser = argparse.ArgumentParser(description="数据集整合脚本")
    
    parser.add_argument("--source", type=str, default="all",
                        help="数据源: all/cail/disc-law/chatlaw/crawled/augmented")
    parser.add_argument("--output", type=str, 
                        default="data/processed/contract_sft",
                        help="输出路径")
    
    # 数据源路径（可选）
    parser.add_argument("--cail-dir", type=str, default="data/download/CAIL")
    parser.add_argument("--disc-law-dir", type=str, default="data/download/DISC-LawLLM")
    parser.add_argument("--chatlaw-dir", type=str, default="data/download/ChatLaw")
    parser.add_argument("--cuad-dir", type=str, default="data/download/CUAD")
    parser.add_argument("--crawled-dir", type=str, default="data/processed/cleaned")
    parser.add_argument("--augmented-dir", type=str, default="data/processed/augmented")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        cail_dir=args.cail_dir,
        discllaw_dir=args.disc_law_dir,
        chatlaw_dir=args.chatlaw_dir,
        cuad_dir=args.cuad_dir,
        crawled_dir=args.crawled_dir,
        augmented_dir=args.augmented_dir,
        output_dir=args.output
    )
    
    prepare_dataset(config)


if __name__ == "__main__":
    main()
