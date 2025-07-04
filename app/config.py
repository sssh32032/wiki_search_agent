"""
Environment settings and configuration management
Manage API keys, system parameters and environment variables
"""

# Always load .env before any config or os.environ usage
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Application settings class"""
    # API key settings
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    cohere_api_key: str = Field(default_factory=lambda: os.environ.get("COHERE_API_KEY", ""))
    # Model settings
    openai_model: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    # Vector database settings
    vector_db_path: str = Field(default_factory=lambda: os.environ.get("VECTOR_DB_PATH", "./faiss_index"))
    chunk_size: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", 1000)))
    chunk_overlap: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", 200)))
    # Wikipedia settings
    wiki_language: str = Field(default_factory=lambda: os.environ.get("WIKI_LANGUAGE", "en"))
    wiki_max_pages: int = Field(default_factory=lambda: int(os.environ.get("WIKI_MAX_PAGES", 100)))
    # Retrieval settings
    top_k: int = Field(default_factory=lambda: int(os.environ.get("TOP_K", 5)))
    similarity_threshold: float = Field(default_factory=lambda: float(os.environ.get("SIMILARITY_THRESHOLD", 0.7)))
    # Reranking settings
    rerank_top_n: int = Field(default_factory=lambda: int(os.environ.get("RERANK_TOP_N", 3)))
    # Application settings
    app_name: str = Field(default_factory=lambda: os.environ.get("APP_NAME", "Wikipedia Assistant"))
    app_version: str = Field(default_factory=lambda: os.environ.get("APP_VERSION", "1.0.0"))
    debug: bool = Field(default_factory=lambda: os.environ.get("DEBUG", "False").lower() == "true")
    # Data directory
    data_dir: str = Field(default_factory=lambda: os.environ.get("DATA_DIR", "./data"))

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get settings instance"""
    return settings

def validate_api_keys() -> bool:
    """Validate if API keys are set"""
    required_keys = ["openai_api_key", "cohere_api_key"]
    for key in required_keys:
        if not getattr(settings, key):
            print(f"Warning: {key} not set")
            return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        settings.data_dir,
        settings.vector_db_path,
        "./logs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

# Create directories on initialization
if __name__ == "__main__":
    create_directories()
    if validate_api_keys():
        print("✅ API key validation successful")
    else:
        print("❌ API key validation failed, please check .env file") 