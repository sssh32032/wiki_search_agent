"""
Unit tests for scripts.build_embeddings module
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from scripts.build_embeddings import EmbeddingProcessor


class TestEmbeddingProcessor:
    """Test EmbeddingProcessor class"""
    
    @pytest.fixture(autouse=True)
    def setup_processor(self, mock_env_vars, clean_test_environment):
        """Setup processor for testing"""
        self.processor = EmbeddingProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert hasattr(self.processor, 'model')
        assert hasattr(self.processor, 'model_name')
    
    @patch('scripts.build_embeddings.SentenceTransformer')
    def test_create_text_chunks(self, mock_sentence_transformer):
        """Test text chunking"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        processor = EmbeddingProcessor()
        
        test_text = "This is a test sentence. This is another sentence. And a third one."
        chunks = processor.create_text_chunks(test_text, max_length=50)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    @patch('scripts.build_embeddings.SentenceTransformer')
    def test_process_data_file(self, mock_sentence_transformer):
        """Test data file processing"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        processor = EmbeddingProcessor()
        
        # Create test data file
        test_data = {
            "Test Page": {
                "title": "Test Page",
                "content": "This is test content for processing.",
                "url": "https://test.com"
            }
        }
        
        test_file = Path("tests/test_data/test_wiki.json")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        try:
            chunks = processor.process_data_file(test_file.name)
            assert len(chunks) > 0
            assert all('text' in chunk for chunk in chunks)
            assert all('title' in chunk for chunk in chunks)
        finally:
            if test_file.exists():
                test_file.unlink()
    
    @patch('scripts.build_embeddings.SentenceTransformer')
    @patch('scripts.build_embeddings.faiss.IndexFlatIP')
    def test_build_index(self, mock_faiss, mock_sentence_transformer):
        """Test index building"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_sentence_transformer.return_value = mock_model
        
        # Mock FAISS index
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        processor = EmbeddingProcessor()
        
        # Create test data file
        test_data = {
            "Test Page": {
                "title": "Test Page",
                "content": "This is test content for processing.",
                "url": "https://test.com"
            }
        }
        
        test_file = Path("tests/test_data/test_wiki.json")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        try:
            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                with patch('faiss.write_index') as mock_write_index:
                    processor.build_index(test_file.name)
                    
                    # Verify FAISS index was called
                    mock_faiss.assert_called_once()
                    mock_write_index.assert_called_once()
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_text_chunking_logic(self):
        """Test text chunking logic with different inputs"""
        processor = EmbeddingProcessor()
        
        # Test short text
        short_text = "Short text."
        chunks = processor.create_text_chunks(short_text, max_length=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Test long text
        long_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = processor.create_text_chunks(long_text, max_length=20)
        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
    
    @patch('scripts.build_embeddings.SentenceTransformer')
    def test_embedding_generation(self, mock_sentence_transformer):
        """Test embedding generation"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_sentence_transformer.return_value = mock_model
        
        processor = EmbeddingProcessor()
        
        # Test embedding generation
        texts = ["Text 1", "Text 2"]
        embeddings = processor.model.encode(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert len(embeddings[1]) == 3 