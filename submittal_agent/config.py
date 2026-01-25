"""Configuration settings for the Construction Submittal Review Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# LLM Configuration
class LLMConfig:
    """LLM model configuration with fallback support."""
    
    # Primary LLM (Claude)
    PRIMARY_PROVIDER = "anthropic"
    PRIMARY_MODEL = "claude-sonnet-4-20250514"
    PRIMARY_MAX_TOKENS = 4096
    
    # Fallback LLM (GPT-4)
    FALLBACK_PROVIDER = "openai"
    FALLBACK_MODEL = "gpt-4-turbo"
    FALLBACK_MAX_TOKENS = 4096
    
    # Vision model (Claude)
    VISION_MODEL = "claude-sonnet-4-20250514"
    VISION_MAX_TOKENS = 2048
    
    # Temperature for deterministic outputs
    TEMPERATURE = 0.0
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds


# Embedding Configuration
class EmbeddingConfig:
    """Embedding model configuration."""
    
    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    BATCH_SIZE = 100


# Chunking Configuration
class ChunkingConfig:
    """Document chunking configuration."""
    
    CHUNK_SIZE = 512  # tokens
    CHUNK_OVERLAP = 50  # tokens
    
    # Construction spec section markers
    SECTION_MARKERS = [
        r"^\d+\.\d+\s+",      # "3.01 EXAMINATION"
        r"^PART\s+\d+",       # "PART 2 PRODUCTS"
        r"^[A-Z]\.\s+",       # "A. Materials"
        r"^Section\s+\d+",    # "Section 03 30 00"
    ]


# Retrieval Configuration
class RetrievalConfig:
    """Retrieval configuration."""
    
    TOP_K = 8
    RELEVANCE_THRESHOLD = 0.3
    
    # Collection name in ChromaDB
    COLLECTION_NAME = "construction_specs"


# Review Decision Types
DECISION_TYPES = [
    "approved",
    "approved_as_noted", 
    "revise_and_resubmit",
    "rejected"
]

# Compliance Check Categories
COMPLIANCE_CATEGORIES = [
    "specification",
    "code",
    "completeness",
    "quality",
    "coordination"
]

# Submittal Types
SUBMITTAL_TYPES = [
    "product_data",
    "shop_drawing",
    "material_sample",
    "mix_design",
    "certification",
    "other"
]
