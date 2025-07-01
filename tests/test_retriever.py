"""
Unit tests for app.core.retriever module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import faiss

from app.core.retriever import Retriever


class TestRetriever:
    """Test Retriever class"""
    
    @pytest.fixture(autouse=True)
    def setup_retriever(self, mock_env_vars, clean_test_environment):
        """Setup retriever for testing"""
        self.retriever = None
    
    def test_retriever_initialization(self):
        """Test retriever initialization"""
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('app.core.retriever.faiss.read_index') as mock_faiss:
                with patch('builtins.open', create=True) as mock_open:
                    # Mock the model
                    mock_model_instance = Mock()
                    mock_model.return_value = mock_model_instance
                    
                    # Mock FAISS index
                    mock_index = Mock()
                    mock_faiss.return_value = mock_index
                    
                    # Mock metadata file
                    mock_open.return_value.__enter__.return_value.read.return_value = '[]'
                    
                    # Mock file existence
                    with patch('pathlib.Path.glob') as mock_glob:
                        mock_glob.return_value = [Path('test_index.index')]
                        
                        retriever = Retriever()
                        assert retriever.model_name == "all-MiniLM-L6-v2"
                        assert retriever.model is not None
                        assert retriever.index is not None
    
    def test_retrieve_method(self):
        """Test retrieve method"""
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('app.core.retriever.faiss.read_index') as mock_faiss:
                with patch('builtins.open', create=True) as mock_open:
                    # Setup mocks
                    mock_model_instance = Mock()
                    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                    mock_model.return_value = mock_model_instance
                    
                    mock_index = Mock()
                    mock_index.search.return_value = (
                        np.array([[0.8, 0.6, 0.4]]),
                        np.array([[0, 1, 2]])
                    )
                    mock_faiss.return_value = mock_index
                    
                    # Mock metadata
                    mock_metadata = [
                        {
                            'text': 'Test document 1',
                            'title': 'Test Title 1',
                            'chunk_index': 0,
                            'url': 'http://test1.com'
                        },
                        {
                            'text': 'Test document 2', 
                            'title': 'Test Title 2',
                            'chunk_index': 1,
                            'url': 'http://test2.com'
                        }
                    ]
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
                    
                    with patch('pathlib.Path.glob') as mock_glob:
                        mock_glob.return_value = [Path('test_index.index')]
                        
                        retriever = Retriever()
                        results = retriever.retrieve("test query", k=2)
                        
                        assert len(results) == 2
                        assert results[0]['text'] == 'Test document 1'
                        assert results[0]['score'] == 0.8
                        assert results[1]['text'] == 'Test document 2'
                        assert results[1]['score'] == 0.6
    
    def test_rerank_methods(self):
        """Test different reranking methods"""
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('app.core.retriever.faiss.read_index') as mock_faiss:
                with patch('builtins.open', create=True) as mock_open:
                    # Setup basic mocks
                    mock_model_instance = Mock()
                    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                    mock_model.return_value = mock_model_instance
                    mock_index = Mock()
                    mock_faiss.return_value = mock_index
                    mock_metadata = [{'text': 'Test', 'title': 'Test', 'chunk_index': 0}]
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
                    with patch('pathlib.Path.glob') as mock_glob:
                        mock_glob.return_value = [Path('test_index.index')]
                        retriever = Retriever()
                        docs = [
                            {'text': 'AI is artificial intelligence', 'score': 0.8},
                            {'text': 'Machine learning is a subset of AI', 'score': 0.6}
                        ]
                        # Test keyword reranking (不會 lazy load)
                        with patch.object(retriever, '_load_rerankers') as mock_load:
                            keyword_results = retriever.rerank("artificial intelligence", docs, method="keyword")
                            assert len(keyword_results) == 2
                            assert 'rerank_score' in keyword_results[0]
                            mock_load.assert_not_called()
                        # Test cross_encoder reranking (會 lazy load)
                        with patch.object(retriever, '_load_rerankers') as mock_load:
                            cross_results = retriever.rerank("artificial intelligence", docs, method="cross_encoder")
                            assert len(cross_results) == 2
                            mock_load.assert_called_once()
                        # Test hybrid reranking (會 lazy load)
                        with patch.object(retriever, '_load_rerankers') as mock_load:
                            hybrid_results = retriever.rerank("artificial intelligence", docs, method="hybrid")
                            assert len(hybrid_results) == 2
                            assert 'hybrid_score' in hybrid_results[0]
                            mock_load.assert_called_once()
                        # Test unknown method (預設不 rerank，直接回傳原始)
                        with patch.object(retriever, '_load_rerankers') as mock_load:
                            docs_copy = [d.copy() for d in docs]
                            result = retriever.rerank("artificial intelligence", docs_copy, method="unknown")
                            assert result == docs_copy
                            mock_load.assert_not_called()
    
    def test_batched_cross_encoder_scores(self):
        """Test batched cross-encoder scoring"""
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('app.core.retriever.faiss.read_index') as mock_faiss:
                with patch('builtins.open', create=True) as mock_open:
                    # Setup mocks
                    mock_model_instance = Mock()
                    mock_model.return_value = mock_model_instance
                    
                    mock_index = Mock()
                    mock_faiss.return_value = mock_index
                    
                    mock_metadata = [{'text': 'Test', 'title': 'Test', 'chunk_index': 0}]
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
                    
                    with patch('pathlib.Path.glob') as mock_glob:
                        mock_glob.return_value = [Path('test_index.index')]
                        
                        retriever = Retriever()
                        
                        # Mock cross-encoder model and tokenizer
                        retriever.cross_encoder_model = Mock()
                        retriever.tokenizer = Mock()
                        
                        # Mock tokenizer output
                        mock_features = {
                            'input_ids': Mock(),
                            'attention_mask': Mock()
                        }
                        retriever.tokenizer.return_value = mock_features
                        
                        # Mock model output
                        mock_outputs = Mock()
                        mock_outputs.logits = Mock()
                        retriever.cross_encoder_model.return_value = mock_outputs
                        
                        # Mock torch operations
                        with patch('torch.sigmoid') as mock_sigmoid:
                            with patch('torch.no_grad'):
                                mock_sigmoid.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([0.8, 0.6])
                                
                                docs = [
                                    {'text': 'AI is artificial intelligence'},
                                    {'text': 'Machine learning is a subset of AI'}
                                ]
                                
                                scores = retriever._batched_cross_encoder_scores("artificial intelligence", docs)
                                assert len(scores) == 2
                                assert scores[0] == 0.8
                                assert scores[1] == 0.6
    
    def test_search_and_rerank(self):
        """Test search and rerank method"""
        with patch('app.core.retriever.SentenceTransformer') as mock_model:
            with patch('app.core.retriever.faiss.read_index') as mock_faiss:
                with patch('builtins.open', create=True) as mock_open:
                    # Setup mocks
                    mock_model_instance = Mock()
                    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                    mock_model.return_value = mock_model_instance
                    
                    mock_index = Mock()
                    mock_index.search.return_value = (
                        np.array([[0.8, 0.6]]),
                        np.array([[0, 1]])
                    )
                    mock_faiss.return_value = mock_index
                    
                    mock_metadata = [
                        {'text': 'Test doc 1', 'title': 'Test 1', 'chunk_index': 0},
                        {'text': 'Test doc 2', 'title': 'Test 2', 'chunk_index': 1}
                    ]
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
                    
                    with patch('pathlib.Path.glob') as mock_glob:
                        mock_glob.return_value = [Path('test_index.index')]
                        
                        retriever = Retriever()
                        results = retriever.search_and_rerank("test query", k=2, rerank_method="hybrid")
                        
                        assert len(results) == 2
                        assert 'hybrid_score' in results[0]
                        assert 'semantic_score' in results[0]
                        assert 'cross_encoder_score' in results[0] 