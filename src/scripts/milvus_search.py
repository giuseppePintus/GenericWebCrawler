import os
import time
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    utility,
    MilvusException
)

class MilvusSearcher:
    def __init__(self, 
                 collection_name: str = "document_chunks",
                 model_name: str = "all-MiniLM-L6-v2",
                 max_retries: int = 5,
                 retry_delay: int = 2):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get connection params from environment
        self.host = os.getenv('MILVUS_HOST', 'localhost')
        self.port = os.getenv('MILVUS_PORT', '19530')
        
        self._connect_milvus()
        self._load_collection()

    def _connect_milvus(self):
        """Connect to Milvus with retry mechanism"""
        retry_count = 0
        last_exception = None

        while retry_count < self.max_retries:
            try:
                print(f"Attempting to connect to Milvus at {self.host}:{self.port} (attempt {retry_count + 1})")
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=30  # Extended timeout
                )
                print("Successfully connected to Milvus")
                return
            except MilvusException as e:
                last_exception = e
                retry_count += 1
                if retry_count < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    print(f"Connection failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        raise MilvusException(f"Failed to connect to Milvus after {self.max_retries} attempts: {str(last_exception)}")

    def _load_collection(self):
        self.collection = Collection(self.collection_name)
        self.collection.load()
        print(f"Loaded collection: {self.collection_name}")

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        # Generate embedding for query
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Define output fields including text
        output_fields = ["domain", "url", "text"]
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        return results[0]  # Return first (and only) query results

    def list_by_domain(self, domain: str):
        expr = f'domain == "{domain}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["domain", "url"]
        )
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Search vectors in Milvus')
    parser.add_argument('--query', type=str, required=True, help='Search query text')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    args = parser.parse_args()

    try:
        searcher = MilvusSearcher()
        results = searcher.search_similar(args.query, args.top_k)
        
        print(f"\nTop {args.top_k} results for query: '{args.query}'")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Domain: {result.entity.get('domain')}")
            print(f"   URL: {result.entity.get('url')}")
            print(f"   Text: {result.entity.get('text')}")
            print()
            
    except Exception as e:
        print(f"Error during search: {str(e)}")
        exit(1)