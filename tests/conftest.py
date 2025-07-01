"""
Pytest configuration and shared fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_wiki_data():
    """Mock Wikipedia data for testing"""
    return {
        "Artificial intelligence": {
            "title": "Artificial intelligence",
            "content": "Artificial intelligence (AI) is the capability of computational systems to perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "extract": "Artificial intelligence (AI) is intelligence demonstrated by machines..."
        }
    }


@pytest.fixture(scope="session")
def mock_embeddings_data():
    """Mock embeddings data for testing"""
    return [
        {
            "text": "Artificial intelligence (AI) is the capability of computational systems to perform tasks that typically require human intelligence.",
            "title": "Artificial intelligence",
            "chunk_index": 0,
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        },
        {
            "text": "These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            "title": "Artificial intelligence", 
            "chunk_index": 1,
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        }
    ]


@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    monkeypatch.setenv("WIKIPEDIA_LANGUAGE", "en")
    monkeypatch.setenv("VECTOR_DB_PATH", "tests/test_data/faiss_index")
    monkeypatch.setenv("DATA_DIR", "tests/test_data")
    monkeypatch.setenv("LOG_DIR", "tests/test_data/logs")


@pytest.fixture(scope="function")
def clean_test_environment():
    """Clean up test environment before and after tests"""
    # Setup
    test_data_dir = Path("tests/test_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir) 