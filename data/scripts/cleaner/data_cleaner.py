#!/usr/bin/env python3
"""
数据清洗模块
去除页眉页脚、噪声，文本标准化，长度过滤

使用方法：
    python data_cleaner.py --input ./processed --output ./processed/cleaned
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import unicodedata

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CleanConfig:
    """清洗配置"""
    # 长度过滤
    min_length: int = 50           # 最小文本长度
    max_length: int = 500000       # 最大文本长度
    
    # 页眉页脚
    remove_headers_footers: bool = True
    header_footer_patterns: List[str] = None
    
    # 噪声去除
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_qq_wechat: bool = True
    remove_special_chars: bool = False
    
    # 文本标准化
    normalize_whitespace: bool = True
    normalize_punctuation: bool = True
    remove_empty_lines: bool = True
    
    # 编码处理
    fix_encoding: bool = True
    remove_non_chinese: bool = False  # 是否移除非中文字符
    
    def __post_init__(self):
        if self.header_footer_patterns is None:
            self.header_footer_patterns = [
                r'^第\s*\d+\s*页',           # 第 x 页
                r'^Page\s+\d+',              # Page x
                r'^\d+\s*/\s*\d+$',          # 1/10
                r'^\s*[-=]{3,}\s*$',         # 分隔线
                r'^\s*[\d\.]+\s*$',          # 页码数字
            ]


class DataCleaner:
    """数据清洗器"""
    
    # 常见的页眉页脚模式
    DEFAULT_HEADER_FOOTER_PATTERNS = [
        r'^第\s*\d+\s*页',
        r'^Page\s+\d+',
        r'^\d+\s*/\s*\d+$',
        r'^共\s*\d+\s*页',
        r'^\s*第\s*\d+\s*页\s*共\s*\d+\s*页',
    ]
    
    # 噪声模式
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    PHONE_PATTERN = re.compile(
        r'1[3-9]\d{9}',  # 手机号
    )
    PHONE_PATTERN_2 = re.compile(
        r'0\d{2,3}[-\s]?\d{7,8}',  # 固定电话
    )
    QQ_PATTERN = re.compile(
        r'QQ\s*[：:]\s*\d{5,11}',
        re.IGNORECASE
    )
    WECHAT_PATTERN = re.compile(
        r'微信\s*[：:]\s*[a-zA-Z0-9_-]+',
        re.IGNORECASE
    )
    
    # 需要去除的特殊字符
    SPECIAL_CHARS_PATTERN = re.compile(
        r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]',
        re.UNICODE
    )
    
    # 空白字符模式
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    # 重复标点
    PUNCTUATION_PATTERN = re.compile(r'([，。！？；：、]){2,}')
    
    def __init__(self, config: CleanConfig = None):
        self.config = config or CleanConfig()
        
        # 编译页眉页脚模式
        self.header_footer_re = [
            re.compile(p, re.MULTILINE) 
            for p in self.config.header_footer_patterns or self.DEFAULT_HEADER_FOOTER_PATTERNS
        ]
    
    def clean(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        cleaned = text
        
        # 1. 修复编码问题
        if self.config.fix_encoding:
            cleaned = self._fix_encoding(cleaned)
        
        # 2. 去除特殊字符
        cleaned = self.SPECIAL_CHARS_PATTERN.sub('', cleaned)
        
        # 3. 去除URL
        if self.config.remove_urls:
            cleaned = self.URL_PATTERN.sub('', cleaned)
        
        # 4. 去除邮箱
        if self.config.remove_emails:
            cleaned = self.EMAIL_PATTERN.sub('', cleaned)
        
        # 5. 去除电话号码
        if self.config.remove_phone_numbers:
            cleaned = self.PHONE_PATTERN.sub('', cleaned)
            cleaned = self.PHONE_PATTERN_2.sub('', cleaned)
        
        # 6. 去除QQ/微信
        if self.config.remove_qq_wechat:
            cleaned = self.QQ_PATTERN.sub('', cleaned)
            cleaned = self.WECHAT_PATTERN.sub('', cleaned)
        
        # 7. 去除页眉页脚
        if self.config.remove_headers_footers:
            cleaned = self._remove_headers_footers(cleaned)
        
        # 8. 标准化空白字符
        if self.config.normalize_whitespace:
            cleaned = self.WHITESPACE_PATTERN.sub(' ', cleaned)
            cleaned = cleaned.strip()
        
        # 9. 标准化标点
        if self.config.normalize_punctuation:
            cleaned = self._normalize_punctuation(cleaned)
        
        # 10. 去除空行
        if self.config.remove_empty_lines:
            lines = [line.strip() for line in cleaned.split('\n')]
            lines = [line for line in lines if line]
            cleaned = '\n'.join(lines)
        
        # 11. 去除非中文（可选）
        if self.config.remove_non_chinese:
            cleaned = self._remove_non_chinese(cleaned)
        
        # 12. 长度过滤
        if len(cleaned) < self.config.min_length:
            return ""
        
        if len(cleaned) > self.config.max_length:
            cleaned = cleaned[:self.config.max_length]
        
        return cleaned
    
    def _fix_encoding(self, text: str) -> str:
        """修复编码问题"""
        # 处理常见的编码错误
        # 这里可以做更多编码检测和修复
        
        # 统一Unicode
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """去除页眉页脚"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过匹配页眉页脚模式的行
            matched = False
            for pattern in self.header_footer_re:
                if pattern.match(line.strip()):
                    matched = True
                    break
            
            if not matched:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        # 去除重复标点
        text = self.PUNCTUATION_PATTERN.sub(r'\1', text)
        
        # 中文标点与英文标点的转换（可选）
        # 这里保持原样
        
        return text
    
    def _remove_non_chinese(self, text: str) -> str:
        """去除非中文字符"""
        # 保留中文、常用标点、数字、字母
        pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：、（）【】《》""''（）]')
        return pattern.sub('', text)
    
    def clean_file(self, input_path: str, output_path: str = None) -> Dict:
        """
        清洗单个文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        
        Returns:
            处理结果字典
        """
        result = {
            "input": input_path,
            "output": output_path,
            "success": False,
            "original_length": 0,
            "cleaned_length": 0,
            "error": ""
        }
        
        try:
            # 读取文件
            with open(input_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            result["original_length"] = len(original_text)
            
            # 清洗
            cleaned_text = self.clean(original_text)
            
            result["cleaned_length"] = len(cleaned_text)
            
            # 检查是否有效
            if not cleaned_text:
                result["error"] = "清洗后文本为空（可能长度不足）"
                return result
            
            # 保存
            if output_path:
                output_p = Path(output_path)
                output_p.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_p, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"清洗文件失败 {input_path}: {e}")
        
        return result
    
    def clean_json(self, input_path: str, output_path: str = None,
                   text_field: str = "text") -> Dict:
        """
        清洗JSON文件中的文本
        
        Args:
            input_path: 输入JSON文件路径
            output_path: 输出文件路径
            text_field: 文本字段名
        
        Returns:
            处理结果字典
        """
        result = {
            "input": input_path,
            "output": output_path,
            "success": False,
            "error": ""
        }
        
        try:
            # 读取JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同格式
            if isinstance(data, dict):
                # 单条记录
                if text_field in data:
                    original = data[text_field]
                    data[text_field] = self.clean(original)
                    result["original_length"] = len(original) if original else 0
                    result["cleaned_length"] = len(data[text_field])
                    
            elif isinstance(data, list):
                # 多条记录
                for item in data:
                    if isinstance(item, dict) and text_field in item:
                        original = item[text_field]
                        item[text_field] = self.clean(original)
                
                result["original_length"] = sum(
                    len(item.get(text_field, "")) for item in data if isinstance(item, dict)
                )
                result["cleaned_length"] = sum(
                    len(item.get(text_field, "")) for item in data if isinstance(item, dict)
                )
            
            # 检查是否有效
            if result.get("cleaned_length", 0) == 0:
                result["error"] = "清洗后文本为空"
                return result
            
            # 保存
            if output_path:
                output_p = Path(output_path)
                output_p.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_p, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"清洗JSON失败 {input_path}: {e}")
        
        return result


def process_directory(input_dir: str, output_dir: str = None,
                      config: CleanConfig = None) -> List[Dict]:
    """
    批量处理目录中的文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config: 清洗配置
    
    Returns:
        处理结果列表
    """
    input_path = Path(input_dir)
    cleaner = DataCleaner(config)
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.name}_cleaned"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    success_count = 0
    fail_count = 0
    
    # 处理所有文件
    for file_path in input_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        # 计算相对路径
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        
        if file_path.suffix == '.json':
            # JSON文件
            result = cleaner.clean_json(str(file_path), str(target_path))
        elif file_path.suffix == '.txt':
            # 文本文件
            result = cleaner.clean_file(str(file_path), str(target_path))
        else:
            # 其他文件直接复制
            if file_path.suffix.lower() in ['.pdf', '.doc', '.docx']:
                continue
            # 直接复制
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # import shutil
            # shutil.copy2(file_path, target_path)
            continue
        
        results.append(result)
        
        if result.get("success"):
            success_count += 1
        else:
            fail_count += 1
            logger.warning(f"处理失败: {file_path.name} - {result.get('error')}")
    
    # 保存统计
    stats = {
        "total": len(results),
        "success": success_count,
        "failed": fail_count,
        "results": results
    }
    
    stats_file = output_path / "cleaning_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"清洗完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info(f"统计文件: {stats_file}")
    logger.info("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="数据清洗模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python data_cleaner.py --input ./processed --output ./processed/cleaned
    python data_cleaner.py --input ./processed --min-length 100 --max-length 100000
    python data_cleaner.py --input ./data.json --output ./cleaned.json
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入文件或目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件或目录")
    parser.add_argument("--min-length", type=int, default=50,
                        help="最小文本长度 (默认: 50)")
    parser.add_argument("--max-length", type=int, default=500000,
                        help="最大文本长度 (默认: 500000)")
    parser.add_argument("--no-header-footer", action="store_true",
                        help="不去除页眉页脚")
    parser.add_argument("--no-url", action="store_true",
                        help="不去除URL")
    parser.add_argument("--no-phone", action="store_true",
                        help="不去除电话号码")
    parser.add_argument("--remove-non-chinese", action="store_true",
                        help="去除非中文字符")
    
    args = parser.parse_args()
    
    # 创建配置
    config = CleanConfig(
        min_length=args.min_length,
        max_length=args.max_length,
        remove_headers_footers=not args.no_header_footer,
        remove_urls=not args.no_url,
        remove_phone_numbers=not args.no_phone,
        remove_non_chinese=args.remove_non_chinese
    )
    
    input_path = Path(args.input)
    output_path = args.output
    
    if input_path.is_file():
        # 单文件处理
        cleaner = DataCleaner(config)
        
        if input_path.suffix == '.json':
            result = cleaner.clean_json(str(input_path), output_path)
        else:
            result = cleaner.clean_file(str(input_path), output_path)
        
        if result.get("success"):
            logger.info(f"清洗成功: {result.get('cleaned_length')} 字符")
        else:
            logger.error(f"清洗失败: {result.get('error')}")
            
    else:
        # 目录批量处理
        results = process_directory(str(input_path), output_path, config)


if __name__ == "__main__":
    main()
