import numpy as np
try:
    import cupy as cp  # For GPU support
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import os
from datetime import datetime
import scrapy
import hashlib

class GeneralSpider(scrapy.Spider):
    name = 'general_spider'  # This is the name you need to use
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            self.xp = cp  # Use cupy if GPU available
        else:
            self.xp = np
        self.logger.info(f"GPU acceleration: {self.gpu_available}")
        
        # Initialize sentence transformer (lighter than BERT)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.gpu_available:
            self.model.to('cuda')
            
    def get_page_id(self, url):
        """Generate unique page ID from URL"""
        if url.endswith('/'):
            url = url[:-1]
        page_name = url.split('/')[-1] or 'index'
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{page_name}_{url_hash}"
        
    def parse(self, response):
        # Create site directory structure
        site_dir = Path(f"output/{self.name}")
        pages_dir = site_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        # Create unique directory for this page
        page_id = self.get_page_id(response.url)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        page_dir = pages_dir / f"{page_id}_{timestamp}"
        
        # Create subdirectories
        raw_dir = page_dir / "raw"
        processed_dir = page_dir / "processed"
        chunks_dir = processed_dir / "chunks"
        
        for directory in [raw_dir, processed_dir, chunks_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Save raw HTML
        with open(raw_dir / "html.txt", "w", encoding='utf-8') as f:
            f.write(response.text)

        # Save metadata
        metadata = {
            "url": response.url,
            "timestamp": timestamp,
            "title": response.css("title::text").get(),
            "headers": dict(response.headers.items()),
            "status": response.status
        }
        with open(raw_dir / "metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Process text content
        if response.headers.get('Content-Type', b'').startswith(b'text/html'):
            text = " ".join(response.css("body ::text").getall())
            chunks = self._create_chunks(text)
            
            # Process and save chunks
            for i, chunk in enumerate(chunks):
                # Generate embeddings
                embedding = self.model.encode(chunk)
                if self.gpu_available:
                    embedding = cp.asnumpy(embedding)
                
                chunk_data = {
                    "text": chunk,
                    "embedding": embedding.tolist(),
                    "chunk_index": i,
                    "page_url": response.url
                }
                
                with open(chunks_dir / f"chunk_{i:04d}.json", "w", encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2)

            # Follow links within allowed domains
            for link in response.css('a::attr(href)').getall():
                yield response.follow(link, self.parse)

    def _create_chunks(self, text, chunk_size=1000, overlap=200):
        """Create overlapping chunks from text"""
        if not text:
            self.logger.warning("Empty text received for chunking")
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = end - overlap
            
        return chunks