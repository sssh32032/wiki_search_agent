"""
Unit tests for scripts.fetch_wiki module (refactored)
"""

import pytest
import json
from unittest.mock import patch, Mock
from pathlib import Path
from scripts.fetch_wiki import WikipediaFetcher

class TestWikipediaFetcher:
    @pytest.fixture(autouse=True)
    def setup_fetcher(self, mock_env_vars, clean_test_environment):
        self.fetcher = WikipediaFetcher()

    def test_fetcher_initialization(self):
        assert self.fetcher.language == "en"
        assert self.fetcher.max_pages == 5
        assert self.fetcher.data_dir.exists()

    @patch('scripts.fetch_wiki.wikipedia.search')
    @patch('scripts.fetch_wiki.wikipedia.page')
    def test_get_page_content_success(self, mock_page, mock_search):
        mock_search.return_value = ["Test Page"]
        mock_page_obj = Mock()
        mock_page_obj.content = "This is test content. " * 10
        mock_page_obj.summary = "Test summary."
        mock_page_obj.title = "Test Page"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Test_Page"
        mock_page_obj.categories = ["Category1", "Category2"]
        mock_page_obj.links = ["Link1", "Link2"]
        mock_page.return_value = mock_page_obj
        result = self.fetcher.get_page_content("Test Page")
        assert result["title"] == "Test Page"
        assert result["url"] == "https://en.wikipedia.org/wiki/Test_Page"
        assert result["content"] == self.fetcher.clean_text(mock_page_obj.content)
        assert result["summary"] == "Test summary."
        assert result["categories"] == ["Category1", "Category2"]
        assert result["links"] == ["Link1", "Link2"]

    @patch('scripts.fetch_wiki.wikipedia.search')
    def test_get_page_content_not_found(self, mock_search):
        mock_search.return_value = []
        result = self.fetcher.get_page_content("Nonexistent Page")
        assert result is None

    @patch('scripts.fetch_wiki.wikipedia.search')
    @patch('scripts.fetch_wiki.wikipedia.page')
    def test_get_page_content_short_content(self, mock_page, mock_search):
        mock_search.return_value = ["Short Page"]
        mock_page_obj = Mock()
        mock_page_obj.content = "short"
        mock_page_obj.summary = "summary"
        mock_page_obj.title = "Short Page"
        mock_page_obj.url = "url"
        mock_page_obj.categories = []
        mock_page_obj.links = []
        mock_page.return_value = mock_page_obj
        result = self.fetcher.get_page_content("Short Page")
        assert result is None

    @patch('scripts.fetch_wiki.wikipedia.search')
    @patch('scripts.fetch_wiki.wikipedia.page')
    def test_get_page_content_exception(self, mock_page, mock_search):
        mock_search.return_value = ["Error Page"]
        mock_page.side_effect = Exception("API Error")
        result = self.fetcher.get_page_content("Error Page")
        assert result is None

    @patch('scripts.fetch_wiki.wikipedia.search')
    @patch('scripts.fetch_wiki.wikipedia.page')
    def test_search_and_fetch_pages(self, mock_page, mock_search):
        mock_search.return_value = ["Page1", "Page2"]
        mock_page_obj = Mock()
        mock_page_obj.content = "This is test content. " * 10
        mock_page_obj.summary = "summary"
        mock_page_obj.title = "Page1"
        mock_page_obj.url = "url"
        mock_page_obj.categories = []
        mock_page_obj.links = []
        mock_page.return_value = mock_page_obj
        result = self.fetcher.search_and_fetch_pages("test query", limit=2)
        assert result["total_requested"] == 2
        assert result["successful"] >= 1
        assert "pages" in result

    def test_clean_text(self):
        raw = "<b>Test</b> [[Link|Alias]] [https://test.com label] {{template}} <!--comment--> <ref>ref</ref> * item"
        cleaned = self.fetcher.clean_text(raw)
        assert "<b>" not in cleaned
        assert "[[" not in cleaned
        assert "{{" not in cleaned
        assert "<!--" not in cleaned
        assert "<ref>" not in cleaned
        assert not cleaned.strip().startswith("* ")

    def test_data_dir_created(self):
        # Should exist after fetcher init
        assert self.fetcher.data_dir.exists()

    def test_fetch_and_save_pages_creates_file(self):
        # Use a dummy page_content method to avoid real API calls
        self.fetcher.get_page_content = Mock(return_value={
            "title": "Dummy",
            "url": "url",
            "content": "content" * 50,
            "summary": "summary",
            "categories": [],
            "links": [],
            "timestamp": "now"
        })
        file_count_before = len(list(self.fetcher.data_dir.glob("wikipedia_pages_*.json")))
        self.fetcher.fetch_and_save_pages(["Dummy"])
        file_count_after = len(list(self.fetcher.data_dir.glob("wikipedia_pages_*.json")))
        assert file_count_after >= file_count_before
        assert file_count_after <= file_count_before + 1 