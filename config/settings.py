from dataclasses import dataclass
import os
import torch

@dataclass
class Settings:
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    
    # LLM Settings
    LLM_MODEL = "TheBloke/Llama-2-7B-Chat-GGML"
    LLM_CACHE_DIR = os.path.join(CACHE_DIR, "llm")
    
    # Embedder Settings
    EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDER_CACHE_DIR = os.path.join(CACHE_DIR, "embedder")
    
    # Milvus Settings
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    COLLECTION_PREFIX = "documents_"
    
    # Model Parameters
    MAX_LENGTH = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.95
    
    # Hallucination Detection Settings
    HALLUCINATION_THRESHOLD = 0.5
    SIMILARITY_THRESHOLD = 0.3
    DEBUG_MODE = True

    # Hallucination Detection Parameters
    SEMANTIC_ENTROPY_THRESHOLD = 0.5
    EIGEN_SCORE_THRESHOLD = -1.5
    FEATURE_CLIP_PERCENTILE = 0.2
    NUM_GENERATIONS = 5

    # Device Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

settings = Settings()