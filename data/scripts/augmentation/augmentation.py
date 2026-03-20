#!/usr/bin/env python3
"""
数据增强模块
使用大模型API生成Negative样本，用于合同合规审查的指令微调训练

支持的增强方式：
1. 修改违约金比例
2. 改变管辖法院
3. 修改违约条款
4. 调整付款条件

使用方法：
    python augmentation.py --input ./processed/cleaned --output ./processed/augmented
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('augmentation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """增强类型"""
    PENALTY_RATE = "penalty_rate"          # 修改违约金比例
    JURISDICTION = "jurisdiction"           # 改变管辖法院
    BREACH_TERMS = "breach_terms"           # 修改违约条款
    PAYMENT_TERMS = "payment_terms"         # 调整付款条件


@dataclass
class AugmentationConfig:
    """增强配置"""
    # API配置
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # 增强参数
    negative_ratio: float = 1.0  # 负样本比例
    batch_size: int = 10
    max_workers: int = 3
    retry_times: int = 3
    retry_delay: float = 1.0
    
    # 输出格式
    output_format: str = "alpaca"  # alpaca, sharegpt, raw
    
    # 合同字段
    contract_fields: Dict[str, str] = None
    
    def __post_init__(self):
        if self.contract_fields is None:
            self.contract_fields = {
                "text": "text",
                "question": "question",
                "answer": "answer",
                "label": "label"
            }


class ContractAugmenter:
    """合同数据增强器"""
    
    # 常见的违约金比例（用于替换）
    PENALTY_RATES = [
        "5%", "10%", "15%", "20%", "30%", "50%", "0.5‰", "1‰", "2‰", "5‰"
    ]
    
    # 常见的管辖法院（用于替换）
    JURISDICTIONS = [
        "北京市朝阳区人民法院",
        "上海市浦东新区人民法院",
        "广州市天河区人民法院",
        "深圳市南山区人民法院",
        "杭州市西湖区人民法院",
        "成都市锦江区人民法院",
        "武汉市江汉区人民法院",
        "南京市鼓楼区人民法院",
    ]
    
    # 违约条款替换词
    BREACH_TERMS = {
        "解除合同": "继续履行合同",
        "赔偿损失": "支付违约金",
        "违约金": "滞纳金",
        "继续履行": "解除合同",
        "双倍返还定金": "没收定金",
    }
    
    # 付款条件替换词
    PAYMENT_TERMS = {
        "货到付款": "预付款",
        "预付款": "货到付款",
        "30天内付款": "90天内付款",
        "验收后付款": "发货前付款",
        "月结30天": "月结90天",
    }
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.api_client = None
        
        if config.api_key:
            self._init_api_client()
    
    def _init_api_client(self):
        """初始化API客户端"""
        try:
            import openai
            openai.api_key = self.config.api_key
            openai.api_base = self.config.api_base
            self.api_client = openai
            logger.info(f"API客户端已初始化: {self.config.model}")
        except ImportError:
            logger.warning("openai库未安装，将使用规则增强")
    
    def augment(self, sample: Dict, target_types: List[AugmentationType] = None) -> List[Dict]:
        """
        增强单条样本
        
        Args:
            sample: 原始样本
            target_types: 目标增强类型列表
        
        Returns:
            增强后的样本列表
        """
        if target_types is None:
            # 随机选择1-2种增强类型
            num_types = random.randint(1, 2)
            target_types = random.sample(
                list(AugmentationType), 
                min(num_types, len(list(AugmentationType)))
            )
        
        augmented_samples = []
        
        for aug_type in target_types:
            if self.api_client and self.config.api_key:
                # 使用API增强
                augmented = self._augment_with_api(sample, aug_type)
            else:
                # 使用规则增强
                augmented = self._augment_with_rules(sample, aug_type)
            
            if augmented:
                augmented_samples.append(augmented)
        
        return augmented_samples
    
    def _augment_with_api(self, sample: Dict, aug_type: AugmentationType) -> Optional[Dict]:
        """使用API进行增强"""
        prompt = self._build_prompt(sample, aug_type)
        
        for attempt in range(self.config.retry_times):
            try:
                response = self.api_client.ChatCompletion.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的合同法律助手，擅长修改和生成合同条款。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                result_text = response.choices[0].message.content
                
                # 解析结果
                augmented = self._parse_api_result(sample, result_text, aug_type)
                return augmented
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.retry_times}): {e}")
                time.sleep(self.config.retry_delay)
        
        logger.error(f"API调用失败，超过重试次数")
        return None
    
    def _build_prompt(self, sample: Dict, aug_type: AugmentationType) -> str:
        """构建增强提示"""
        text = sample.get(self.config.contract_fields.get("text", "text"), "")
        
        prompts = {
            AugmentationType.PENALTY_RATE: f"""请修改以下合同文本中的违约金比例条款，使其与原文不同但仍然合理：

{text}

请直接返回修改后的合同文本，不要添加任何解释。""",
            
            AugmentationType.JURISDICTION: f"""请修改以下合同文本中的管辖法院条款，将其改为其他地区的法院：

{text}

请直接返回修改后的合同文本，不要添加任何解释。""",
            
            AugmentationType.BREACH_TERMS: f"""请修改以下合同文本中的违约责任条款，将其改为不同的表述方式：

{text}

请直接返回修改后的合同文本，不要添加任何解释。""",
            
            AugmentationType.PAYMENT_TERMS: f"""请修改以下合同文本中的付款条件条款，将其改为不同的付款方式或时间：

{text}

请直接返回修改后的合同文本，不要添加任何解释。""",
        }
        
        return prompts.get(aug_type, "")
    
    def _parse_api_result(self, original: Dict, result_text: str, 
                         aug_type: AugmentationType) -> Dict:
        """解析API返回的结果"""
        augmented = original.copy()
        
        # 更新文本
        text_field = self.config.contract_fields.get("text", "text")
        augmented[text_field] = result_text
        
        # 添加元数据
        augmented["augmented"] = True
        augmented["augmentation_type"] = aug_type.value
        augmented["original_text"] = original.get(text_field, "")
        
        # 生成标签（0表示有风险/违规，1表示正常）
        # 这里可以根据实际需求设置
        augmented["label"] = 0
        augmented["augmented_at"] = datetime.now().isoformat()
        
        # 构建指令微调格式
        augmented = self._format_output(augmented)
        
        return augmented
    
    def _augment_with_rules(self, sample: Dict, aug_type: AugmentationType) -> Optional[Dict]:
        """使用规则进行增强"""
        text_field = self.config.contract_fields.get("text", "text")
        original_text = sample.get(text_field, "")
        
        if not original_text:
            return None
        
        augmented = sample.copy()
        
        if aug_type == AugmentationType.PENALTY_RATE:
            # 修改违约金比例
            for rate in self.PENALTY_RATES:
                if rate in original_text:
                    new_rate = random.choice([r for r in self.PENALTY_RATES if r != rate])
                    augmented_text = original_text.replace(rate, new_rate)
                    break
            else:
                # 如果没有找到违约金比例，随机添加一个
                augmented_text = original_text + f"\n违约方应支付合同总金额的{random.choice(self.PENALTY_RATES)}作为违约金。"
                
        elif aug_type == AugmentationType.JURISDICTION:
            # 修改管辖法院
            found_jurisdiction = False
            augmented_text = original_text
            for jurisdiction in self.JURISDICTIONS:
                if jurisdiction in original_text:
                    new_jurisdiction = random.choice([j for j in self.JURISDICTIONS if j != jurisdiction])
                    augmented_text = original_text.replace(jurisdiction, new_jurisdiction)
                    found_jurisdiction = True
                    break
            
            if not found_jurisdiction:
                # 添加管辖法院条款
                augmented_text = original_text + f"\n因本合同产生的争议，由{random.choice(self.JURISDICTIONS)}管辖。"
                
        elif aug_type == AugmentationType.BREACH_TERMS:
            # 修改违约条款
            augmented_text = original_text
            for old_term, new_term in self.BREACH_TERMS.items():
                if old_term in augmented_text:
                    augmented_text = augmented_text.replace(old_term, new_term, 1)
                    break
                    
        elif aug_type == AugmentationType.PAYMENT_TERMS:
            # 修改付款条件
            augmented_text = original_text
            for old_term, new_term in self.PAYMENT_TERMS.items():
                if old_term in augmented_text:
                    augmented_text = augmented_text.replace(old_term, new_term, 1)
                    break
        
        if augmented_text == original_text:
            return None
        
        # 更新文本
        augmented[text_field] = augmented_text
        
        # 添加元数据
        augmented["augmented"] = True
        augmented["augmentation_type"] = aug_type.value
        augmented["original_text"] = original_text
        augmented["label"] = 0  # 负样本
        augmented["augmented_at"] = datetime.now().isoformat()
        
        # 格式化输出
        augmented = self._format_output(augmented)
        
        return augmented
    
    def _format_output(self, sample: Dict) -> Dict:
        """格式化输出为指令微调格式"""
        text_field = self.config.contract_fields.get("text", "text")
        text = sample.get(text_field, "")
        
        if self.config.output_format == "alpaca":
            return {
                "instruction": "请审查以下合同条款，识别潜在的法律风险：",
                "input": text,
                "output": "该合同条款存在风险。",
                "label": sample.get("label", 0)
            }
        elif self.config.output_format == "sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": f"请审查以下合同条款：\n{text}"},
                    {"from": "gpt", "value": "该合同条款存在风险。"}
                ],
                "label": sample.get("label", 0)
            }
        else:
            # raw格式
            return sample


class DataAugmentationPipeline:
    """数据增强流水线"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.augmenter = ContractAugmenter(config)
    
    def process(self, input_path: str, output_path: str) -> Dict:
        """
        处理数据
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        
        Returns:
            处理统计
        """
        # 读取输入数据
        samples = self._load_samples(input_path)
        
        if not samples:
            logger.error(f"未找到有效样本: {input_path}")
            return {"success": False, "error": "No valid samples found"}
        
        logger.info(f"加载了 {len(samples)} 个样本")
        
        # 增强数据
        augmented_samples = []
        original_samples = []
        
        for sample in samples:
            # 保留原始样本
            original = sample.copy()
            original["augmented"] = False
            original["label"] = 1  # 原始样本标记为正样本
            original_samples.append(original)
            
            # 生成负样本
            if random.random() < self.config.negative_ratio:
                augmented = self.augmenter.augment(sample)
                augmented_samples.extend(augmented)
        
        # 合并
        all_samples = original_samples + augmented_samples
        
        # 随机打乱
        random.shuffle(all_samples)
        
        # 保存结果
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "alpaca":
            output_file = output_p / "train.json"
        elif self.config.output_format == "sharegpt":
            output_file = output_p / "train.json"
        else:
            output_file = output_p / "augmented.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        stats = {
            "success": True,
            "total_samples": len(all_samples),
            "original_samples": len(original_samples),
            "augmented_samples": len(augmented_samples),
            "output_file": str(output_file)
        }
        
        logger.info("=" * 50)
        logger.info(f"数据增强完成:")
        logger.info(f"  原始样本: {len(original_samples)}")
        logger.info(f"  增强样本: {len(augmented_samples)}")
        logger.info(f"  总计: {len(all_samples)}")
        logger.info(f"  输出文件: {output_file}")
        logger.info("=" * 50)
        
        return stats
    
    def _load_samples(self, input_path: str) -> List[Dict]:
        """加载样本"""
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            if "data" in data:
                samples = data["data"]
            elif "samples" in data:
                samples = data["samples"]
            else:
                samples = [data]
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="数据增强模块 - 生成合同合规审查训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用API增强（需要配置API_KEY）
    python augmentation.py --input ./data.json --output ./augmented --api-key YOUR_KEY --model gpt-4
    
    # 使用规则增强（免费）
    python augmentation.py --input ./data.json --output ./augmented
    
    # 生成Alpaca格式
    python augmentation.py --input ./data.json --output ./augmented --format alpaca
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入JSON文件")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--api-key", type=str, default="",
                        help="API密钥")
    parser.add_argument("--api-base", type=str, 
                        default="https://api.openai.com/v1",
                        help="API基础URL")
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="使用的模型")
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="负样本生成比例")
    parser.add_argument("--format", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "raw"],
                        help="输出格式")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="批处理大小")
    parser.add_argument("--workers", type=int, default=3,
                        help="并行工作数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建配置
    config = AugmentationConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        negative_ratio=args.negative_ratio,
        output_format=args.format,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    # 创建流水线
    pipeline = DataAugmentationPipeline(config)
    
    # 处理数据
    stats = pipeline.process(args.input, args.output)
    
    if not stats.get("success"):
        logger.error(f"处理失败: {stats.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
