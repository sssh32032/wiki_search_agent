"""
Integration tests for the system (updated for new architecture)
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
import shutil
import tempfile

from scripts.fetch_wiki import WikipediaFetcher

class TestSystemIntegration:
    """Test integration for WikipediaFetcher, config, and data persistence"""

    @pytest.fixture(autouse=True)
    def setup_integration(self, mock_env_vars, clean_test_environment):
        self.test_data_dir = Path("tests/test_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

    @pytest.mark.integration
    def test_llm_chain_end_to_end(self):
        """True integration test: WikipediaRAGChain with real Cohere API and temp DBs"""
        import os
        from app.core.llm_chain import WikipediaRAGChain
        # Use a temp directory for vector/memory DBs
        temp_db_dir = Path("faiss_index/test_integration_db")
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir)
        temp_db_dir.mkdir(parents=True, exist_ok=True)
        # Patch settings to use temp_db_dir
        from app import config
        old_vector_db_path = config.settings.vector_db_path
        old_memory_db_path = getattr(config.settings, 'memory_db_path', None)
        config.settings.vector_db_path = str(temp_db_dir / "wiki_db")
        config.settings.memory_db_path = str(temp_db_dir / "memory_db")
        try:
            rag = WikipediaRAGChain()
            answer = rag.generate("台灣最高的山是哪座山？")
            print("Integration test answer:", answer)
            assert any(x in answer for x in ["玉山", "Yushan", "Jade Mountain"])
        finally:
            # Restore settings
            config.settings.vector_db_path = old_vector_db_path
            if old_memory_db_path is not None:
                config.settings.memory_db_path = old_memory_db_path
            # Clean up temp DBs
            if temp_db_dir.exists():
                shutil.rmtree(temp_db_dir)

    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    def test_wikipedia_fetcher_error_handling(self, mock_wikipedia):
        """Test Wikipedia fetcher error handling"""
        mock_api = Mock()
        mock_api.page.side_effect = Exception("API Error")
        mock_wikipedia.return_value = mock_api
        fetcher = WikipediaFetcher()
        with pytest.raises(Exception):
            fetcher.get_page_content("NonExistentPage")

    def test_data_persistence(self, clean_test_environment):
        """Test data persistence and file operations"""
        test_titles = ["Test Page"]
        fetcher = WikipediaFetcher()
        with patch.object(fetcher, 'get_page_content', return_value={
            "title": "Test Page",
            "content": "Test content for persistence testing",
            "url": "https://test.com"
        }):
            results = fetcher.fetch_and_save_pages(test_titles)
            file_path = results.get('saved_file')
            assert file_path and Path(file_path).exists()
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            assert loaded_data['pages'][0]['title'] == "Test Page"
            Path(file_path).unlink()

    def test_configuration_loading(self):
        """Test configuration loading"""
        from app.config import settings
        assert hasattr(settings, 'wikipedia_language') or hasattr(settings, 'wiki_language')
        assert hasattr(settings, 'vector_db_path')
        assert hasattr(settings, 'data_dir')
        assert hasattr(settings, 'log_dir')

    # Placeholder for future llm_chain.py integration tests
    def test_llm_chain_integration_placeholder(self):
        """Placeholder for WikiRAGSystem integration test (to be implemented)"""
        pass 