"""
Pytest configuration and shared fixtures
"""

import pytest
import tempfile
import shutil
import logging
import os
from pathlib import Path
import sys
import subprocess
import time
import requests

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings

# Configure logging for tests - redirect to temporary file
@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup test logging to temporary file"""
    # Create temporary log file
    temp_log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
    temp_log_path = temp_log_file.name
    temp_log_file.close()
    
    # Configure logging to write to temporary file instead of normal log files
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()  # Also output to console for debugging
        ]
    )
    
    # Store the log file path for cleanup
    setup_test_logging.log_file = temp_log_path
    
    yield
    
    # Cleanup: remove temporary log file
    try:
        if os.path.exists(setup_test_logging.log_file):
            os.unlink(setup_test_logging.log_file)
    except Exception:
        pass  # Ignore cleanup errors

@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    monkeypatch.setenv("WIKI_LANGUAGE", "en")
    monkeypatch.setenv("VECTOR_DB_PATH", "tests/test_data/faiss_index")
    monkeypatch.setenv("DATA_DIR", "tests/test_data")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("WIKI_MAX_PAGES", "5")
    monkeypatch.setenv("RERANK_TOP_N", "2")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.7")


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


@pytest.fixture(scope="session")
def start_test_server():
    """Start the FastAPI server for integration tests (session scoped)"""
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "debug"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # Wait for the server to be ready
    for _ in range(60):
        try:
            res = requests.get("http://127.0.0.1:8000/health")
            if res.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        try:
            out, err = proc.communicate(timeout=5)
            print("[DEBUG] Server stdout:\n", out.decode())
            print("[DEBUG] Server stderr:\n", err.decode())
        except Exception as e:
            print(f"[DEBUG] Could not get server output: {e}")
        proc.terminate()
        raise RuntimeError("Server did not start in time")
    yield
    proc.terminate()
    proc.wait() 