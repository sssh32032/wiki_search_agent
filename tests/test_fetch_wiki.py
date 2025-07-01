"""
Unit tests for scripts.fetch_wiki module
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from scripts.fetch_wiki import WikipediaFetcher


class TestWikipediaFetcher:
    """Test WikipediaFetcher class"""
    
    @pytest.fixture(autouse=True)
    def setup_fetcher(self, mock_env_vars, clean_test_environment):
        """Setup fetcher for testing"""
        self.fetcher = WikipediaFetcher()
    
    def test_fetcher_initialization(self):
        """Test fetcher initialization"""
        assert self.fetcher.language == "en"
        assert hasattr(self.fetcher, 'api')
    
    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    def test_search_pages(self, mock_wikipedia):
        """Test search pages method"""
        # Mock Wikipedia API response
        mock_api = Mock()
        mock_page = Mock()
        mock_page.title = "Artificial intelligence"
        mock_page.summary = "AI is intelligence demonstrated by machines"
        mock_page.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        
        mock_api.page.return_value = mock_page
        mock_wikipedia.return_value = mock_api
        
        with patch('scripts.fetch_wiki.wikipediaapi.Wikipedia') as mock_wiki:
            mock_wiki.return_value = mock_api
            
            fetcher = WikipediaFetcher()
            results = fetcher.search_pages("artificial intelligence", limit=1)
            
            assert len(results) == 1
            assert results[0]['title'] == "Artificial intelligence"
            assert results[0]['content'] == "AI is intelligence demonstrated by machines"
    
    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    def test_fetch_page_content(self, mock_wikipedia):
        """Test fetch page content method"""
        # Mock Wikipedia API response
        mock_api = Mock()
        mock_page = Mock()
        mock_page.title = "Test Page"
        mock_page.text = "This is test content"
        mock_page.url = "https://en.wikipedia.org/wiki/Test_Page"
        
        mock_api.page.return_value = mock_page
        mock_wikipedia.return_value = mock_api
        
        with patch('scripts.fetch_wiki.wikipediaapi.Wikipedia') as mock_wiki:
            mock_wiki.return_value = mock_api
            
            fetcher = WikipediaFetcher()
            content = fetcher.fetch_page_content("Test Page")
            
            assert content['title'] == "Test Page"
            assert content['content'] == "This is test content"
            assert content['url'] == "https://en.wikipedia.org/wiki/Test_Page"
    
    def test_save_data(self, clean_test_environment):
        """Test save data method"""
        test_data = {
            "Test Page": {
                "title": "Test Page",
                "content": "Test content",
                "url": "https://test.com"
            }
        }
        
        file_path = self.fetcher.save_data(test_data)
        
        assert Path(file_path).exists()
        
        # Verify saved content
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data == test_data
    
    @patch('scripts.fetch_wiki.wikipediaapi.Wikipedia')
    def test_error_handling(self, mock_wikipedia):
        """Test error handling"""
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
        file_path = self.fetcher.save_data(test_data)
        
        assert Path(file_path).exists()
        
        # Test data loading
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
        
        # Cleanup
        Path(file_path).unlink() 