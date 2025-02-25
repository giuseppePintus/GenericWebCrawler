#!/usr/bin/env python3

import os
import requests
import urllib.parse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup, Comment, Tag
import magic
import time
import logging
from pathlib import Path
import mimetypes
import hashlib
import socket
from typing import Set, Dict, Optional, List
import argparse
import json
import sys
import re
from datetime import datetime
from collections import deque

class DirectoryManager:
    def __init__(self, base_dir: str, site_name: str):
        self.base_dir = Path(base_dir)
        self.site_dir = self.base_dir / site_name  # Main site directory
        self.subdirs = ['html', 'pdf', 'images', 'other', 'logs', 'chunks']
    
    def setup(self) -> Path:
        """Create directory structure and return site directory"""
        for subdir in self.subdirs:
            dir_path = self.site_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            os.chmod(dir_path, 0o755)
        return self.site_dir

    def get_subdir(self, subdir: str) -> Path:
        """Get path to specific subdirectory"""
        return self.site_dir / subdir

class CrawlerConfig:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration requirements"""
        required_fields = ['sites', 'global_settings']
        if not all(field in self.config for field in required_fields):
            raise ValueError("Missing required config fields")
            
        # Validate each site has required fields
        for site_name, site_config in self.config['sites'].items():
            required_site_fields = ['domain', 'region']
            missing = [f for f in required_site_fields if f not in site_config]
            if missing:
                raise ValueError(f"Site '{site_name}' missing required fields: {missing}")
    
    def get_site_config(self, site_name: str) -> Optional[dict]:
        """Find site configuration by site name"""
        sites = self.config.get('sites', {})
        site_config = sites.get(site_name)
        if not site_config:
            raise ValueError(f"Site '{site_name}' not found in configuration")
        if 'domain' not in site_config:
            raise ValueError(f"Missing domain for site '{site_name}'")
        return site_config
    
    @property
    def global_settings(self) -> dict:
        return self.config.get('global_settings', {})

class PageCounter:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._counters = {}  # Cache for counters
        
    def get_next_number(self, subdir: str) -> int:
        """Get next available page number with caching"""
        dir_path = self.base_dir / subdir
        
        if subdir not in self._counters:
            # Initialize counter for this subdir
            if not dir_path.exists():
                self._counters[subdir] = 1
            else:
                existing_files = list(dir_path.glob("*_[0-9]*.html"))
                numbers = []
                for f in existing_files:
                    match = re.search(r'[a-zA-Z]+_(\d+)\.html', f.name)
                    if match:
                        numbers.append(int(match.group(1)))
                self._counters[subdir] = max(numbers, default=0) + 1
        else:
            # Increment existing counter
            self._counters[subdir] += 1
            
        return self._counters[subdir]


class ContentExtractor:
    def __init__(self):
        # Primary content containers
        self.main_content_selectors = [
            'main',
            'article',
            '#content',
            '#main-content',
            '.main-content',
            '[role="main"]'
        ]
        
        # Secondary content areas
        self.content_selectors = [
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content'
        ]
        
        # Text containers
        self.text_selectors = [
            'article p',
            '.content p',
            'main p',
            '[role="main"] p'
        ]
        
        self.ignore_selectors = [
            'nav', 'header', 'footer',
            '.social-share', '.cookie-notice',
            '.advertisement', '.sidebar',
            'script', 'style', 'iframe',
            '.menu', '.search-form',
            '.related-posts', '.comments',
            '[class*="ad-"]',
            '[class*="social"]',
            '[class*="share"]',
            '[class*="cookie"]',
            '[class*="popup"]',
            '[class*="newsletter"]'
        ]

    def _get_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find main content container"""
        # Try primary selectors
        for selector in self.main_content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 100:
                return content
                
        # Try secondary selectors
        for selector in self.content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 100:
                return content
                
        # Fallback to body
        return soup.find('body')

    def _extract_text_blocks(self, container: Tag) -> List[str]:
        """Extract non-overlapping text blocks"""
        # Remove unwanted elements
        for selector in self.ignore_selectors:
            for element in container.select(selector):
                element.decompose()

        # Get text blocks
        text_blocks = []
        seen_content = set()
        
        # Try specific text selectors first
        for selector in self.text_selectors:
            for element in container.select(selector):
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if text_hash not in seen_content:
                        text_blocks.append(text)
                        seen_content.add(text_hash)

        # Fallback to all paragraph elements if no text found
        if not text_blocks:
            for element in container.find_all(['p', 'div', 'section']):
                if element.find_parent(self.ignore_selectors):
                    continue
                    
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if text_hash not in seen_content:
                        text_blocks.append(text)
                        seen_content.add(text_hash)

        return text_blocks

    def create_chunks(self, content: str, min_size: int = 100, max_size: int = 1000) -> List[str]:
        """Create clean text chunks from content"""
        soup = BeautifulSoup(content, 'html.parser')
        main_content = self._get_main_content(soup)
        
        if not main_content:
            return []
            
        text_blocks = self._extract_text_blocks(main_content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for text in text_blocks:
            text = self._clean_text(text)
            if not text or len(text) < min_size:
                continue
                
            if current_size + len(text) > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
            current_chunk.append(text)
            current_size += len(text)
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return [chunk for chunk in chunks if len(chunk.strip()) >= min_size]

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[\t\n\r]+', ' ', text)  # Remove newlines/tabs
        text = re.sub(r'\s*\|\s*', ' ', text)  # Remove separators
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # Remove emails
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)  # Remove zero-width spaces
        return text.strip()

    def extract_body(self, html_content: str, url: str) -> str:
        """Extract and clean body content, add metadata"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for selector in self.ignore_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Additional cleaning
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        metadata = f"""<!--
    Source URL: {url}
    Crawl Date: {datetime.now().isoformat()}
-->"""
        
        # Clean body content
        body = soup.find('body') or soup.new_tag('body')
        for element in body.find_all(True):
            if not element.get_text(strip=True):
                element.decompose()

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="source-url" content="{url}">
    <meta name="crawl-date" content="{datetime.now().isoformat()}">
    <title>Crawled: {url}</title>
</head>
<body>
{metadata}
{body.prettify()}
</body>
</html>"""

class GeneralSpider:
    def __init__(self, site_name: str, site_config: dict, global_config: dict, output_dir: str):
        self.site_name = site_name
        self.site_config = site_config
        self.global_config = global_config
        self.region = site_config['region']  # Store region
        
        self.domain = site_config['domain']
        self.start_url = site_config.get('start_url') or self._normalize_url(self.domain)
        
        # Initialize URL queue and tracking
        self.url_queue = deque([self.start_url])
        self.visited_urls: Set[str] = set()
        self.processed_count = 0
        
        # Initialize components
        self.dir_manager = DirectoryManager(output_dir, site_name)
        self.base_dir = self.dir_manager.setup()
        self.content_extractor = ContentExtractor()
        self.page_counter = PageCounter(self.base_dir)
        
        # Setup custom headers
        default_headers = {
            'User-Agent': 'GeneralSpider/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        global_headers = global_config.get('headers', {})
        site_headers = site_config.get('headers', {})
        
        self.custom_headers = {
            **default_headers,
            **global_headers,
            **site_headers
        }
        
        # Configuration
        self.max_pages = site_config.get('max_pages', global_config.get('max_pages', 100))
        self.delay = site_config.get('delay', global_config.get('delay', 1.0))
        self.allowed_types = site_config.get('allowed_types', global_config.get('allowed_types', ['text/html']))
        
        # Setup logging and robots
        self._setup_logging()
        self._setup_robots_parser()

    def crawl(self):
        """Main crawling method"""
        self.logger.info(f"Crawling: {self.start_url}")
        
        while self.url_queue and self.processed_count < self.max_pages:
            url = self.url_queue.popleft()
            
            if url not in self.visited_urls:
                try:
                    self.process_url(url)
                    self.visited_urls.add(url)
                    self.processed_count += 1
                    time.sleep(self.delay)
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {str(e)}")

        self.logger.info(f"Crawling complete. Processed {self.processed_count} pages")

    def process_url(self, url: str):
        """Process a single URL"""
        try:
            response = requests.get(url, headers=self.custom_headers, timeout=30)
            content_type = response.headers.get('content-type', '').split(';')[0]
            
            if any(allowed in content_type for allowed in self.allowed_types):
                saved_path = self._save_file(response.content, url, content_type)
                
                if content_type.startswith('text/html'):
                    new_urls = self._extract_urls(response.text, url)
                    self._add_urls(new_urls)
                    
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            raise

    def _extract_urls(self, html: str, base_url: str) -> Set[str]:
        """Extract and normalize URLs from HTML content"""
        urls = set()
        soup = BeautifulSoup(html, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            try:
                url = urllib.parse.urljoin(base_url, link['href'])
                if self._is_valid_url(url):
                    urls.add(url)
            except Exception as e:
                self.logger.debug(f"Error extracting URL from {link}: {str(e)}")
                
        return urls

    def _add_urls(self, new_urls: Set[str]):
        """Add new URLs to queue if valid and not visited"""
        filtered_urls = {
            url for url in new_urls 
            if url not in self.visited_urls 
            and self._is_valid_url(url)
        }
        self.url_queue.extend(filtered_urls)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL with protocol"""
        if not url.startswith(('http://', 'https://')):
            protocols = [
                self.global_config.get('default_protocol', 'https'),
                self.global_config.get('fallback_protocol', 'http')
            ]
            
            for protocol in protocols:
                try:
                    test_url = f"{protocol}://{url}"
                    parsed = urllib.parse.urlparse(test_url)
                    socket.gethostbyname(parsed.netloc)
                    return test_url
                except socket.gaierror:
                    continue
                    
            raise ValueError(f"Could not resolve domain: {url}")
        return url

    def _setup_logging(self):
        self.logger = logging.getLogger(f"crawler.{self.domain}")
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(self.base_dir, 'logs', 'crawler.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _setup_robots_parser(self):
        """Setup robots parser with proper error handling"""
        self.robots_parser = RobotFileParser()
        robots_url = urllib.parse.urljoin(self.start_url, '/robots.txt')
        self.robots_parser.set_url(robots_url)
        
        try:
            self.robots_parser.read()
            self.logger.info(f"Successfully loaded robots.txt from {robots_url}")
        except Exception as e:
            self.logger.warning(f"Could not read robots.txt ({str(e)}). Continuing without restrictions.")
            self.robots_parser = None

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urllib.parse.urlparse(url)
            return (parsed.netloc == self.domain and 
                    parsed.scheme in ['http', 'https'] and
                    self.robots_parser.can_fetch("*", url))
        except Exception:
            return False

    def _get_file_extension(self, content_type: str, url: str) -> str:
        """Determine file extension based on content type and URL"""
        if content_type.startswith('image/'):
            return mimetypes.guess_extension(content_type) or '.img'
        elif content_type == 'application/pdf':
            return '.pdf'
        elif content_type.startswith('text/html'):
            return '.html'
        else:
            # Try to get extension from URL or default to .bin
            return os.path.splitext(url)[1] or '.bin'

    def _extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """Extract and normalize links from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        for link in soup.find_all(['a', 'link']):
            href = link.get('href')
            if href:
                absolute_url = urllib.parse.urljoin(base_url, href)
                if self._is_valid_url(absolute_url):
                    links.add(absolute_url)
        
        return links

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped based on robots.txt and visit history"""
        if url in self.visited_urls:
            return True
        
        if self.robots_parser:
            try:
                if not self.robots_parser.can_fetch("*", url):
                    self.logger.debug(f"Skipping {url} - blocked by robots.txt")
                    return True
            except Exception as e:
                self.logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
                return False
        
        return False

    def _save_file(self, content: bytes, url: str, content_type: str) -> str:
        """Save downloaded content and create chunks if HTML"""
        try:
            if content_type.startswith('text/html'):
                html_content = content.decode('utf-8', errors='ignore')
                clean_html = self.content_extractor.extract_body(html_content, url)
                
                page_num = self.page_counter.get_next_number('html')
                page_dir = f"{self.region}_page_{page_num:04d}"
                
                # Save original HTML
                html_dir = self.dir_manager.get_subdir('html')
                html_path = html_dir / f"{page_dir}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(clean_html)
                
                # Create and save chunks with region info
                chunks = self.content_extractor.create_chunks(clean_html)
                if chunks:
                    chunks_dir = self.dir_manager.get_subdir('chunks') / page_dir
                    chunks_dir.mkdir(parents=True, exist_ok=True)
                    
                    metadata = {
                        'url': url,
                        'timestamp': datetime.now().isoformat(),
                        'content_type': content_type,
                        'chunk_count': len(chunks),
                        'region': self.region
                    }
                    
                    with open(chunks_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    
                    for i, chunk in enumerate(chunks, 1):
                        chunk_path = chunks_dir / f"{self.region}_chunk_{i:04d}.json"
                        with open(chunk_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'text': chunk,
                                'region': self.region
                            }, f, ensure_ascii=False, indent=2)
                
                return str(html_path)
            
            # Handle other content types
            elif any(content_type.startswith(t) for t in self.allowed_types):
                subdir = 'pdf' if content_type == 'application/pdf' else \
                        'images' if content_type.startswith('image/') else 'other'
                
                file_num = self.page_counter.get_next_number(subdir)
                ext = content_type.split('/')[-1]
                filename = f"{self.region}_file_{file_num:04d}.{ext}"
                
                file_path = self.dir_manager.get_subdir(subdir) / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                    
                return str(file_path)
                
        except Exception as e:
            self.logger.error(f"Error saving file from {url}: {str(e)}")
            return ""
        
        return ""

def main():
    parser = argparse.ArgumentParser(description='General Web Crawler')
    parser.add_argument('--site', required=True, help='Site name from config')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--log', required=True, help='Log file path')
    args = parser.parse_args()

    try:
        # Setup root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(args.log),
                logging.StreamHandler()
            ]
        )

        # Load and validate config
        config = CrawlerConfig(args.config)
        
        # Get site-specific configuration
        site_config = config.get_site_config(args.site)
        
        # Create and run spider
        spider = GeneralSpider(
            site_name=args.site,  # Pass site name
            site_config=site_config,
            global_config=config.global_settings,
            output_dir=args.output
        )
        spider.crawl()

    except Exception as e:
        logging.error(f"Crawler failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())