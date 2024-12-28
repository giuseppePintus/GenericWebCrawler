import argparse
import os
import json
import glob
from datetime import datetime
import time
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema, 
    CollectionSchema,
    DataType,
    utility,
    MilvusException
)

class ChunkEmbedder:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 collection_name: str = 'document_chunks',
                 dim: int = 384,
                 max_retries: int = 5,
                 reset: bool = False):
        self.model = SentenceTransformer(model_name)
        self.config_path = os.path.join('/config', 'crawler_config.json')
        self.data_dir = os.path.join('/app/data')
        self.collection_name = collection_name
        self.dim = dim
        self.max_retries = max_retries
        
        print(f"Using config file at: {self.config_path}")
        print(f"Using data directory at: {self.data_dir}")
        
        self.site_configs = self._load_crawler_config()
        self._connect_milvus_with_retry()

        if reset:
            self.reset_collection()
            
        self._setup_collection()

    def reset_collection(self):
        """Reset (drop and recreate) the Milvus collection"""
        try:
            if utility.has_collection(self.collection_name):
                print(f"Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
                print(f"Collection {self.collection_name} dropped successfully")
            else:
                print(f"No existing collection found: {self.collection_name}")
        except Exception as e:
            raise Exception(f"Failed to reset collection: {str(e)}")

    def _load_crawler_config(self) -> Dict:
        """Load and extract only the sites configuration"""
        if not os.path.exists(self.config_path):
            raise Exception(f"Config file not found at: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Extract only the sites section
                sites = config.get('sites', {})
                print(f"Loaded configuration for {len(sites)} sites: {', '.join(sites.keys())}")
                return sites
        except Exception as e:
            raise Exception(f"Failed to load crawler config: {str(e)}")

    def _connect_milvus_with_retry(self):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                connections.connect(
                    alias="default", 
                    host=os.getenv('MILVUS_HOST', 'localhost'),
                    port=os.getenv('MILVUS_PORT', '19530')
                )
                print("Successfully connected to Milvus")
                return
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise Exception(f"Failed to connect to Milvus after {self.max_retries} attempts: {str(e)}")
                print(f"Connection attempt {retry_count} failed, retrying in 5 seconds...")
                time.sleep(5)

    def _setup_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                print(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings",
                enable_dynamic_field=False
            )
            self.collection = Collection(name=self.collection_name, schema=schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Collection {self.collection_name} created successfully")
            
        except MilvusException as e:
            raise Exception(f"Failed to setup collection: {str(e)}")

    def _debug_directory_structure(self, site_name: str):
        """Debug helper to print directory structure"""
        print(f"\nDebugging directory structure for {site_name}:")
        site_dir = os.path.join(self.data_dir, site_name)
        chunks_dir = os.path.join(site_dir, 'chunks')
        
        print(f"Site directory: {site_dir} (exists: {os.path.exists(site_dir)})")
        if os.path.exists(site_dir):
            print("Contents:", os.listdir(site_dir))
            
        print(f"Chunks directory: {chunks_dir} (exists: {os.path.exists(chunks_dir)})")
        if os.path.exists(chunks_dir):
            print("Contents:", os.listdir(chunks_dir))

    def load_chunks(self, domain: str) -> List[Dict]:
        domain_dir = os.path.join(self.data_dir, domain, 'chunks')
        print(f"Looking for chunks in directory: {domain_dir}")
        
        if not os.path.exists(domain_dir):
            print(f"Warning: Chunks directory not found at {domain_dir}")
            return []
            
        all_chunks = []
        
        # List all page directories
        page_dirs = sorted([d for d in os.listdir(domain_dir) 
                          if os.path.isdir(os.path.join(domain_dir, d))])
        
        print(f"Found {len(page_dirs)} page directories")
        
        for page_dir in page_dirs:
            page_path = os.path.join(domain_dir, page_dir)
            metadata_path = os.path.join(page_path, 'metadata.json')
            
            # Load metadata
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    url = metadata.get('url', '')
                    print(f"Processing page {page_dir} from URL: {url}")
            except Exception as e:
                print(f"Error loading metadata from {metadata_path}: {str(e)}")
                continue
            
            # Load chunks
            chunk_files = sorted(glob.glob(os.path.join(page_path, 'chunk_*.json')))
            print(f"Found {len(chunk_files)} chunks in {page_dir}")
            
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        # Extract text directly if chunk_data is dict, or use as-is if string
                        text = chunk_data.get('text', chunk_data) if isinstance(chunk_data, dict) else chunk_data
                        chunk = {
                            'text': str(text),  # Ensure text is string
                            'url': url,
                            'page_id': page_dir
                        }
                        all_chunks.append(chunk)
                except Exception as e:
                    print(f"Error loading chunk from {chunk_file}: {str(e)}")
                    continue
                    
        print(f"Loaded total of {len(all_chunks)} chunks for domain {domain}")
        return all_chunks

    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk['text'] for chunk in chunks]
        print(f"Creating embeddings for {len(texts)} texts")
        return self.model.encode(texts)

    def process_domain(self, site_name: str):
        chunks = self.load_chunks(site_name)
        if not chunks:
            print(f"No chunks found for site: {site_name}")
            return

        valid_chunks = [chunk for chunk in chunks if chunk and isinstance(chunk.get('text'), str)]
        if not valid_chunks:
            print(f"No valid chunks found for site: {site_name}")
            return

        print(f"Creating embeddings for {len(valid_chunks)} texts")
        embeddings = self.create_embeddings(valid_chunks)
        
        # Prepare insert data with explicit string conversion
        insert_data = [
            embeddings.tolist(),  # embedding field
            [site_name] * len(valid_chunks),  # domain field
            [str(chunk.get('url', '')) for chunk in valid_chunks],  # url field
            [str(chunk['text']) for chunk in valid_chunks]  # text field, ensure string
        ]

        try:
            print(f"Inserting {len(insert_data)} field arrays with {len(valid_chunks)} chunks each")
            self.collection.insert(insert_data)
            print(f"Successfully inserted {len(valid_chunks)} chunks for site: {site_name}")
        except MilvusException as e:
            print(f"Failed to insert chunks for site {site_name}: {str(e)}")
            raise

    def process_all_domains(self):
        """Process all sites"""
        sites = list(self.site_configs.keys())
        print(f"\nProcessing {len(sites)} sites...")
        
        for i, site_name in enumerate(sites, 1):
            print(f"\nProcessing site {i}/{len(sites)}: {site_name}")
            self.process_domain(site_name)
            
        self.collection.flush()
        print("\nAll sites processed successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process chunks and store embeddings in Milvus')
    parser.add_argument('--reset', action='store_true', help='Reset Milvus collection before processing')
    args = parser.parse_args()

    embedder = ChunkEmbedder(reset=args.reset)
    embedder.process_all_domains()