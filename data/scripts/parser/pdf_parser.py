#!/usr/bin/env python3
"""
PDF解析模块
提取PDF中的文本和表格

使用方法：
    python pdf_parser.py --input ./pdfs --output ./processed
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PDFParseResult:
    """PDF解析结果"""
    file_path: str
    success: bool
    text: str = ""
    tables: List[List[List[str]]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    page_count: int = 0
    extracted_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "success": self.success,
            "text": self.text,
            "tables": self.tables,
            "metadata": self.metadata,
            "error": self.error,
            "page_count": self.page_count,
            "extracted_at": self.extracted_at
        }


class PDFParser:
    """PDF解析器"""
    
    def __init__(self, use_pdfplumber: bool = True, use_pymupdf: bool = False):
        """
        初始化解析器
        
        Args:
            use_pdfplumber: 是否优先使用pdfplumber
            use_pymupdf: 是否使用PyMuPDF作为备选
        """
        self.use_pdfplumber = use_pdfplumber
        self.use_pymupdf = use_pymupdf
        self.pdfplumber = None
        self.pymupdf = None
        
        self._init_parsers()
    
    def _init_parsers(self):
        """初始化解析库"""
        if self.use_pdfplumber:
            try:
                import pdfplumber
                self.pdfplumber = pdfplumber
                logger.info("使用 pdfplumber 解析器")
            except ImportError:
                logger.warning("pdfplumber 未安装，尝试使用 PyMuPDF")
                self.use_pdfplumber = False
        
        if self.use_pymupdf or not self.use_pdfplumber:
            try:
                import fitz  # PyMuPDF
                self.pymupdf = fitz
                logger.info("使用 PyMuPDF 解析器")
            except ImportError:
                logger.warning("PyMuPDF 未安装")
    
    def parse(self, pdf_path: str) -> PDFParseResult:
        """
        解析PDF文件
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            解析结果
        """
        result = PDFParseResult(
            file_path=pdf_path,
            extracted_at=datetime.now().isoformat()
        )
        
        if not Path(pdf_path).exists():
            result.error = "文件不存在"
            result.success = False
            return result
        
        try:
            # 尝试使用pdfplumber
            if self.pdfplumber:
                result = self._parse_with_pdfplumber(pdf_path)
            
            # 如果失败或未使用，尝试PyMuPDF
            if not result.success and self.pymupdf:
                result = self._parse_with_pymupdf(pdf_path)
            
            if result.success:
                logger.info(f"解析成功: {pdf_path} ({result.page_count} 页)")
            else:
                logger.error(f"解析失败: {pdf_path} - {result.error}")
                
        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"解析异常: {pdf_path} - {e}")
        
        return result
    
    def _parse_with_pdfplumber(self, pdf_path: str) -> PDFParseResult:
        """使用pdfplumber解析"""
        result = PDFParseResult(
            file_path=pdf_path,
            extracted_at=datetime.now().isoformat()
        )
        
        try:
            with self.pdfplumber.open(pdf_path) as pdf:
                result.page_count = len(pdf.pages)
                
                # 提取元数据
                result.metadata = {
                    "parser": "pdfplumber",
                    "page_count": result.page_count,
                    "file_size": Path(pdf_path).stat().st_size
                }
                
                # 提取每页文本和表格
                all_text = []
                all_tables = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)
                    
                    # 提取表格
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)
                
                result.text = "\n\n---PAGE BREAK---\n\n".join(all_text)
                result.tables = all_tables
                result.success = True
                
        except Exception as e:
            result.error = f"pdfplumber error: {e}"
            result.success = False
        
        return result
    
    def _parse_with_pymupdf(self, pdf_path: str) -> PDFParseResult:
        """使用PyMuPDF解析"""
        result = PDFParseResult(
            file_path=pdf_path,
            extracted_at=datetime.now().isoformat()
        )
        
        try:
            doc = self.pymupdf.open(pdf_path)
            result.page_count = len(doc)
            
            # 提取元数据
            meta = doc.metadata
            result.metadata = {
                "parser": "pymupdf",
                "page_count": result.page_count,
                "file_size": Path(pdf_path).stat().st_size,
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
            }
            
            # 提取每页文本
            all_text = []
            all_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                page_text = page.get_text("text")
                if page_text:
                    all_text.append(page_text)
                
                # 尝试提取表格
                tables = self._extract_tables_pymupdf(page)
                if tables:
                    all_tables.extend(tables)
            
            result.text = "\n\n---PAGE BREAK---\n\n".join(all_text)
            result.tables = all_tables
            result.success = True
            
            doc.close()
            
        except Exception as e:
            result.error = f"pymupdf error: {e}"
            result.success = False
        
        return result
    
    def _extract_tables_pymupdf(self, page) -> List[List[List[str]]]:
        """使用PyMuPDF提取表格"""
        tables = []
        
        try:
            # 使用表格检测
            tab = page.find_tables()
            if tab:
                for table in tab:
                    table_data = table.extract()
                    if table_data:
                        tables.append(table_data)
        except Exception:
            pass
        
        return tables


class MinerUParser:
    """
    MinerU高级PDF解析器（可选功能）
    需要安装 MinerU 环境
    """
    
    def __init__(self):
        self.available = False
        try:
            # 尝试导入magic_pdf
            from magic_pdf.data.dataset import PymuDocDataset
            self.available = True
            logger.info("MinerU 解析器可用")
        except ImportError:
            logger.warning("MinerU 未安装，将使用基础解析器")
    
    def parse(self, pdf_path: str, output_dir: str) -> Optional[Dict]:
        """
        使用MinerU解析PDF
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
        
        Returns:
            解析结果字典
        """
        if not self.available:
            return None
        
        try:
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.infer import Parser
            
            # 初始化解析器
            ds = PymuDocDataset(pdf_path)
            
            # 解析
            parser = Parser(ds)
            result = parser.parse()
            
            return result
            
        except Exception as e:
            logger.error(f"MinerU解析失败: {e}")
            return None


def process_single_pdf(args: Tuple[str, str, bool, bool]) -> PDFParseResult:
    """
    处理单个PDF文件（用于多进程）
    
    Args:
        args: (pdf_path, output_dir, use_pdfplumber, use_pymupdf)
    
    Returns:
        解析结果
    """
    pdf_path, output_dir, use_pdfplumber, use_pymupdf = args
    
    parser = PDFParser(
        use_pdfplumber=use_pdfplumber,
        use_pymupdf=use_pymupdf
    )
    
    result = parser.parse(pdf_path)
    
    # 如果指定了输出目录，保存结果
    if output_dir and result.success:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        pdf_name = Path(pdf_path).stem
        json_path = output_path / f"{pdf_name}.json"
        
        # 保存JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存纯文本
        if result.text:
            txt_path = output_path / f"{pdf_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.text)
    
    return result


def batch_parse(input_dir: str, output_dir: str = None, 
                use_pdfplumber: bool = True, use_pymupdf: bool = True,
                workers: int = None) -> List[PDFParseResult]:
    """
    批量解析PDF文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        use_pdfplumber: 使用pdfplumber
        use_pymupdf: 使用PyMuPDF
        workers: 并行进程数
    
    Returns:
        解析结果列表
    """
    input_path = Path(input_dir)
    
    # 查找所有PDF文件
    pdf_files = []
    for ext in ['*.pdf', '*.PDF']:
        pdf_files.extend(input_path.rglob(ext))
    
    if not pdf_files:
        logger.warning(f"在 {input_dir} 中未找到PDF文件")
        return []
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 设置并行进程数
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    success_count = 0
    fail_count = 0
    
    # 准备参数
    args_list = [
        (str(pdf_path), output_dir, use_pdfplumber, use_pymupdf)
        for pdf_path in pdf_files
    ]
    
    # 使用多进程处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_pdf, args): args[0] 
                  for args in args_list}
        
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"处理失败 {pdf_path}: {e}")
                fail_count += 1
    
    # 打印统计
    logger.info("=" * 50)
    logger.info(f"解析完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info("=" * 50)
    
    return results


def merge_results(results: List[PDFParseResult], output_path: str):
    """
    合并解析结果
    
    Args:
        results: 解析结果列表
        output_path: 输出文件路径
    """
    merged = {
        "total": len(results),
        "success": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "files": [r.to_dict() for r in results]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    logger.info(f"合并结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF解析模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python pdf_parser.py --input ./pdfs --output ./processed
    python pdf_parser.py --input ./pdfs --output ./processed --workers 4
    python pdf_parser.py --input ./pdfs/single.pdf --output ./processed
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入PDF文件或目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录 (默认: 与输入同目录)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="并行进程数 (默认: CPU核数-1)")
    parser.add_argument("--no-pdfplumber", action="store_true",
                        help="不使用pdfplumber")
    parser.add_argument("--no-pymupdf", action="store_true",
                        help="不使用PyMuPDF")
    parser.add_argument("--merge", action="store_true",
                        help="合并所有结果为单个JSON文件")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        if input_path.is_file():
            output_dir = str(input_path.parent / "parsed")
        else:
            output_dir = str(input_path / "parsed")
    
    # 解析
    if input_path.is_file():
        # 单文件解析
        parser = PDFParser(
            use_pdfplumber=not args.no_pdfplumber,
            use_pymupdf=not args.no_pymupdf
        )
        result = parser.parse(str(input_path))
        
        # 保存结果
        if result.success:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            json_path = output_path / f"{input_path.stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            if result.text:
                txt_path = output_path / f"{input_path.stem}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(result.text)
            
            logger.info(f"解析成功: {json_path}")
        else:
            logger.error(f"解析失败: {result.error}")
            
    else:
        # 批量解析
        results = batch_parse(
            str(input_path),
            output_dir,
            use_pdfplumber=not args.no_pdfplumber,
            use_pymupdf=not args.no_pymupdf,
            workers=args.workers
        )
        
        # 合并结果
        if args.merge and results:
            merge_path = Path(output_dir) / "all_results.json"
            merge_results(results, str(merge_path))


if __name__ == "__main__":
    main()
