"""
Integration tests for the complete system
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.retriever import Retriever
from scripts.fetch_wiki import WikipediaFetcher
from scripts.build_embeddings import EmbeddingProcessor


class TestSystemIntegration:
    """Test complete system integration"""
    
    @pytest.fixture(autouse=True)
    def setup_integration(self, mock_env_vars, clean_test_environment):
        """Setup integration test environment"""
        self.test_data_dir = Path("tests/test_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    @patch('scripts.build_embeddings.SentenceTransformer')
    @patch('app.core.retriever.SentenceTransformer')
    @patch('app.core.retriever.faiss.read_index')
    def test_complete_workflow(self, mock_faiss_read, mock_retriever_model, 
                              mock_embeddings_model, mock_wikipedia):
        """Test complete workflow from Wikipedia fetch to retrieval"""
        
        # Mock Wikipedia API
        mock_api = Mock()
        mock_page = Mock()
        mock_page.title = "Artificial intelligence"
        mock_page.text = "Artificial intelligence (AI) is the capability of computational systems to perform tasks that typically require human intelligence."
        mock_page.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        mock_api.page.return_value = mock_page
        mock_wikipedia.return_value = mock_api
        
        # Mock sentence transformer for embeddings
        mock_emb_model = Mock()
        mock_emb_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings_model.return_value = mock_emb_model
        
        # Mock sentence transformer for retriever
        mock_ret_model = Mock()
        mock_ret_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_retriever_model.return_value = mock_ret_model
        
        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (
            [[0.8, 0.6]],
            [[0, 1]]
        )
        mock_faiss_read.return_value = mock_index
        
        # Step 1: Fetch Wikipedia data
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '[]'
            
            fetcher = WikipediaFetcher()
            data = fetcher.fetch_page_content("Artificial intelligence")
            
            assert data['title'] == "Artificial intelligence"
            assert "Artificial intelligence" in data['content']
        
        # Step 2: Process embeddings
        with patch('faiss.IndexFlatIP') as mock_faiss_index:
            with patch('faiss.write_index') as mock_write_index:
                with patch('builtins.open', create=True) as mock_file_open:
                    mock_file_open.return_value.__enter__.return_value.read.return_value = '[]'
                    
                    processor = EmbeddingProcessor()
                    
                    # Create test data file
                    test_data = {
                        "Artificial intelligence": {
                            "title": "Artificial intelligence",
                            "content": "Artificial intelligence (AI) is the capability of computational systems to perform tasks that typically require human intelligence.",
                            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
                        }
                    }
                    
                    test_file = self.test_data_dir / "test_wiki.json"
                    with open(test_file, 'w', encoding='utf-8') as f:
                        json.dump(test_data, f, ensure_ascii=False, indent=2)
                    
                    try:
                        processor.build_index(test_file.name)
                        
                        # Verify FAISS operations were called
                        mock_faiss_index.assert_called_once()
                        mock_write_index.assert_called_once()
                    finally:
                        if test_file.exists():
                            test_file.unlink()
        
        # Step 3: Test retrieval
        with patch('builtins.open', create=True) as mock_open:
            mock_metadata = [
                {
                    'text': 'Artificial intelligence (AI) is the capability of computational systems to perform tasks that typically require human intelligence.',
                    'title': 'Artificial intelligence',
                    'chunk_index': 0,
                    'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence'
                }
            ]
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
            
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [Path('test_index.index')]
                
                retriever = Retriever()
                results = retriever.search_and_rerank("What is artificial intelligence?", k=1)
                
                assert len(results) == 1
                assert "artificial intelligence" in results[0]['text'].lower()
                assert 'hybrid_score' in results[0]
    
    def test_error_handling(self):
        """Test error handling in the system"""
        
        # Test retriever with missing index
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = []  # No index files
                
                mock_model_instance = Mock()
                mock_model.return_value = mock_model_instance
                
                with pytest.raises(FileNotFoundError):
                    Retriever()
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        from app.config import settings
        
        assert hasattr(settings, 'wikipedia_language')
        assert hasattr(settings, 'vector_db_path')
        assert hasattr(settings, 'data_dir')
        assert hasattr(settings, 'log_dir')
    
    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    def test_wikipedia_fetcher_error_handling(self, mock_wikipedia):
        """Test Wikipedia fetcher error handling"""
        # Mock Wikipedia API to raise exception
        mock_api = Mock()
        mock_api.page.side_effect = Exception("API Error")
        mock_wikipedia.return_value = mock_api
        
        fetcher = WikipediaFetcher()
        
        # Should handle errors gracefully
        with pytest.raises(Exception):
            fetcher.fetch_page_content("NonExistentPage")
    
    def test_data_persistence(self, clean_test_environment):
        """Test data persistence and file operations"""
        test_data = {
            "Test Page": {
                "title": "Test Page",
                "content": "Test content for persistence testing",
                "url": "https://test.com"
            }
        }
        
        # Test data saving
        fetcher = WikipediaFetcher()
        file_path = fetcher.save_data(test_data)
        
        assert Path(file_path).exists()
        
        # Test data loading
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
        
        # Cleanup
        Path(file_path).unlink() 