"""
Environment settings and configuration management
Manage API keys, system parameters and environment variables
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings class"""
    
    # API key settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    cohere_api_key: str = Field(default="", env="COHERE_API_KEY")
    
    # Model settings
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Vector database settings
    vector_db_path: str = Field(default="./faiss_index", env="VECTOR_DB_PATH")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Wikipedia settings
    wiki_language: str = Field(default="en", env="WIKI_LANGUAGE")
    wiki_max_pages: int = Field(default=100, env="WIKI_MAX_PAGES")
    
    # Retrieval settings
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Reranking settings
    rerank_top_n: int = Field(default=3, env="RERANK_TOP_N")
    
    # Application settings
    app_name: str = Field(default="Wikipedia Assistant", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Data directory
    data_dir: str = Field(default="./data", env="DATA_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


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