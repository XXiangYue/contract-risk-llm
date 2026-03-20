#!/usr/bin/env python3
"""
中国政府采购网爬虫
爬取公开招标信息，支持PDF下载

使用方法：
    python chinabidding_crawler.py --keyword 合同 --pages 5 --output ./download
"""

import os
import re
import sys
import time
import json
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawler.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class ChinaBiddingCrawler:
    """中国政府采购网爬虫"""
    
    BASE_URL = "http://www.ccgp.gov.cn"
    SEARCH_URL = "http://www.ccgp.gov.cn/search/"
    
    # 请求头
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    
    def __init__(self, output_dir: str = "./download", delay: float = 2.0, 
                 max_workers: int = 3, timeout: int = 30):
        """
        初始化爬虫
        
        Args:
            output_dir: PDF下载保存目录
            delay: 请求间隔(秒)
            max_workers: 最大并发线程数
            timeout: 请求超时时间(秒)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_workers = max_workers
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        
        # 统计信息
        self.stats = {
            "pages_parsed": 0,
            "items_found": 0,
            "pdfs_downloaded": 0,
            "errors": 0
        }
        self.lock = threading.Lock()
        
        # 记录已下载的URL，避免重复
        self.downloaded_urls = set()
        self.load_downloaded_history()
    
    def load_downloaded_history(self):
        """加载历史下载记录"""
        history_file = self.output_dir / "download_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.downloaded_urls = set(data.get('urls', []))
                    logger.info(f"已加载 {len(self.downloaded_urls)} 条历史记录")
            except Exception as e:
                logger.warning(f"加载历史记录失败: {e}")
    
    def save_download_history(self):
        """保存下载记录"""
        history_file = self.output_dir / "download_history.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'urls': list(self.downloaded_urls),
                    'last_updated': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存历史记录失败: {e}")
    
    def _make_request(self, url: str, method: str = "GET", **kwargs) -> Optional[requests.Response]:
        """
        发送HTTP请求
        
        Args:
            url: 请求URL
            method: 请求方法
            **kwargs: 其他请求参数
        
        Returns:
            响应对象，失败返回None
        """
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=kwargs.get('timeout', self.timeout),
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            with self.lock:
                self.stats["errors"] += 1
            logger.error(f"请求失败 [{url}]: {e}")
            return None
    
    def search_page(self, keyword: str, page: int = 1, category: str = "cg") -> List[Dict]:
        """
        搜索招标信息
        
        Args:
            keyword: 搜索关键词
            page: 页码
            category: 类别 (cg=采购, zy=招标)
        
        Returns:
            招标信息列表
        """
        # 构建搜索URL
        params = {
            "keyword": keyword,
            "page": page,
            "category": category,
            "searchtype": "title"
        }
        
        url = f"{self.SEARCH_URL}?{quote('&'.join([f'{k}={v}' for k,v in params.items()]))}"
        
        logger.info(f"搜索页面: {keyword} - 第 {page} 页")
        
        response = self._make_request(url)
        if not response:
            return []
        
        return self._parse_search_results(response.text)
    
    def _parse_search_results(self, html: str) -> List[Dict]:
        """
        解析搜索结果页面
        
        Args:
            html: 页面HTML
        
        Returns:
            招标信息列表
        """
        soup = BeautifulSoup(html, 'lxml')
        results = []
        
        # 查找结果列表
        items = soup.select('div.list_info li, ul.village_list li, div.channel_box li')
        
        for item in items:
            try:
                # 提取标题和链接
                title_elem = item.select_one('a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href', '')
                
                if not link or not title:
                    continue
                
                # 补全URL
                if not link.startswith('http'):
                    link = urljoin(self.BASE_URL, link)
                
                # 提取日期
                date_elem = item.select_one('span.date, i.time')
                date_str = date_elem.get_text(strip=True) if date_elem else ""
                
                # 提取地区
                area_elem = item.select_one('span.area, i.loc')
                area = area_elem.get_text(strip=True) if area_elem else ""
                
                result = {
                    "title": title,
                    "url": link,
                    "date": date_str,
                    "area": area,
                    "crawled_at": datetime.now().isoformat()
                }
                
                results.append(result)
                with self.lock:
                    self.stats["items_found"] += 1
                    
            except Exception as e:
                logger.warning(f"解析列表项失败: {e}")
                with self.lock:
                    self.stats["errors"] += 1
        
        with self.lock:
            self.stats["pages_parsed"] += 1
        
        return results
    
    def get_detail(self, url: str) -> Optional[Dict]:
        """
        获取招标详情
        
        Args:
            url: 详情页URL
        
        Returns:
            详情信息字典
        """
        logger.info(f"获取详情: {url}")
        
        response = self._make_request(url)
        if not response:
            return None
        
        return self._parse_detail(response.text, url)
    
    def _parse_detail(self, html: str, url: str) -> Dict:
        """
        解析详情页面
        
        Args:
            html: 页面HTML
            url: 页面URL
        
        Returns:
            详情信息字典
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # 提取标题
        title = ""
        title_elem = soup.select_one('div.title, h1.title, h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        # 提取发布时间
        publish_time = ""
        time_elem = soup.select_one('span.time, div.time, i.time')
        if time_elem:
            publish_time = time_elem.get_text(strip=True)
        
        # 提取正文内容
        content = ""
        content_elem = soup.select_one('div.content, div.txt, div.con, article')
        if content_elem:
            content = content_elem.get_text(strip=True)
        
        # 提取PDF链接
        pdf_links = self._extract_pdf_links(soup, url)
        
        return {
            "url": url,
            "title": title,
            "publish_time": publish_time,
            "content": content[:5000] if content else "",  # 限制内容长度
            "pdf_links": pdf_links,
            "crawled_at": datetime.now().isoformat()
        }
    
    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        提取PDF下载链接
        
        Args:
            soup: BeautifulSoup对象
            base_url: 基础URL
        
        Returns:
            PDF链接列表
        """
        pdf_links = []
        
        # 查找所有链接
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            
            # 判断是否为PDF链接
            if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                
                pdf_links.append({
                    "url": href,
                    "name": text or href.split('/')[-1]
                })
        
        # 也检查附件区域
        attach_section = soup.select('div.attachment, div.attach, ul.attach_list')
        if attach_section:
            for a in attach_section[0].find_all('a', href=True):
                href = a.get('href', '')
                if href.lower().endswith('.pdf'):
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    
                    name = a.get_text(strip=True) or href.split('/')[-1]
                    if not any(p['url'] == href for p in pdf_links):
                        pdf_links.append({"url": href, "name": name})
        
        return pdf_links
    
    def download_pdf(self, pdf_url: str, save_dir: Optional[Path] = None) -> Optional[Path]:
        """
        下载PDF文件
        
        Args:
            pdf_url: PDF下载链接
            save_dir: 保存目录
        
        Returns:
            保存的文件路径，失败返回None
        """
        if save_dir is None:
            save_dir = self.output_dir / "pdfs"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已下载
        if pdf_url in self.downloaded_urls:
            logger.info(f"跳过已下载: {pdf_url}")
            return None
        
        # 生成文件名
        filename = pdf_url.split('/')[-1]
        if not filename.lower().endswith('.pdf'):
            filename = f"{hashlib.md5(pdf_url.encode()).hexdigest()[:8]}.pdf"
        
        # 清理文件名
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        save_path = save_dir / filename
        
        logger.info(f"下载PDF: {pdf_url}")
        
        response = self._make_request(pdf_url, stream=True)
        if not response:
            return None
        
        # 检查内容类型
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not response.content[:4].startswith(b'%PDF'):
            logger.warning(f"不是有效的PDF文件: {pdf_url}")
            return None
        
        # 保存文件
        try:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with self.lock:
                self.stats["pdfs_downloaded"] += 1
                self.downloaded_urls.add(pdf_url)
            
            logger.info(f"保存PDF: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"保存PDF失败: {e}")
            return None
    
    def crawl(self, keyword: str, pages: int = 5, download_pdfs: bool = True,
              max_pdfs: int = 100) -> List[Dict]:
        """
        执行爬取任务
        
        Args:
            keyword: 搜索关键词
            pages: 爬取页数
            download_pdfs: 是否下载PDF
            max_pdfs: 最大下载PDF数量
        
        Returns:
            爬取结果列表
        """
        all_results = []
        pdf_urls = []
        
        # 搜索各页
        for page in range(1, pages + 1):
            results = self.search_page(keyword, page)
            all_results.extend(results)
            
            # 延迟
            time.sleep(self.delay)
            
            # 获取详情和PDF链接
            for result in results:
                detail = self.get_detail(result['url'])
                if detail:
                    result.update(detail)
                    pdf_urls.extend(detail.get('pdf_links', []))
                
                time.sleep(self.delay)
        
        # 下载PDF
        if download_pdfs and pdf_urls:
            logger.info(f"开始下载 {len(pdf_urls)} 个PDF文件...")
            
            pdf_downloaded = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_pdf, pdf['url']): pdf 
                    for pdf in pdf_urls[:max_pdfs]
                }
                
                for future in as_completed(futures):
                    pdf_downloaded += 1
                    if pdf_downloaded >= max_pdfs:
                        break
                    time.sleep(self.delay)
        
        # 保存结果
        self._save_results(all_results)
        
        # 保存下载历史
        self.save_download_history()
        
        # 打印统计
        self._print_stats()
        
        return all_results
    
    def _save_results(self, results: List[Dict]):
        """保存爬取结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"crawl_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("爬取统计:")
        logger.info(f"  解析页面数: {self.stats['pages_parsed']}")
        logger.info(f"  发现条目数: {self.stats['items_found']}")
        logger.info(f"  下载PDF数: {self.stats['pdfs_downloaded']}")
        logger.info(f"  错误数: {self.stats['errors']}")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="中国政府采购网爬虫",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python chinabidding_crawler.py --keyword 合同 --pages 3
    python chinabidding_crawler.py --keyword 采购 --pages 5 --output ./download
    python chinabidding_crawler.py --keyword 招标 --pages 2 --no-pdf
        """
    )
    
    parser.add_argument("--keyword", "-k", type=str, default="合同",
                        help="搜索关键词 (默认: 合同)")
    parser.add_argument("--pages", "-p", type=int, default=3,
                        help="爬取页数 (默认: 3)")
    parser.add_argument("--output", "-o", type=str, default="./download",
                        help="输出目录 (默认: ./download)")
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                        help="请求间隔秒数 (默认: 2.0)")
    parser.add_argument("--workers", "-w", type=int, default=3,
                        help="最大并发线程数 (默认: 3)")
    parser.add_argument("--max-pdfs", type=int, default=100,
                        help="最大下载PDF数量 (默认: 100)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="不下载PDF文件")
    
    args = parser.parse_args()
    
    # 创建爬虫
    crawler = ChinaBiddingCrawler(
        output_dir=args.output,
        delay=args.delay,
        max_workers=args.workers
    )
    
    # 执行爬取
    crawler.crawl(
        keyword=args.keyword,
        pages=args.pages,
        download_pdfs=not args.no_pdf,
        max_pdfs=args.max_pdfs
    )


if __name__ == "__main__":
    main()
