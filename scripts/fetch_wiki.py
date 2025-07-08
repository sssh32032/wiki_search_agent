#!/usr/bin/env python3
"""
Wikipedia data fetching module
Fetch data from Wikipedia API, clean and preprocess, save to data directory
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Import project settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings

try:
    import wikipedia
except ImportError:
    print("Error: Please install wikipedia package first")
    print("Run: pip install wikipedia")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/wiki_fetch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WikipediaFetcher:
    """Wikipedia data fetcher"""
    
    def __init__(self):
        """Initialize Wikipedia settings"""
        self.language = settings.wiki_language
        self.max_pages = settings.wiki_max_pages
        self.data_dir = Path(settings.data_dir)
        
        # Set Wikipedia language
        wikipedia.set_lang(self.language)
        logger.info(f"Wikipedia language set to: {self.language}")
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and markers
        text = re.sub(r'\[\[([^|\]]*?)\]\]', r'\1', text)  # Remove internal link markers
        text = re.sub(r'\[\[[^|\]]*?\|([^\]]*?)\]\]', r'\1', text)  # Handle internal links with aliases
        text = re.sub(r'\[\[([^|\]]*?)\]\]', r'\1', text)  # Handle remaining internal links
        
        # Remove external links
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove template markers
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        
        # Remove comments
        text = re.sub(r'<!--[^>]*-->', '', text)
        
        # Remove reference markers
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)
        text = re.sub(r'<ref[^>]*/>', '', text)
        
        # Remove table markers
        text = re.sub(r'\|.*?\|', '', text)
        
        # Remove title markers
        text = re.sub(r'^=+\s*([^=]+)\s*=+$', r'\1', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^[\*#]\s*', '', text, flags=re.MULTILINE)
        
        # Remove extra blank lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def get_page_content(self, page_title: str) -> Optional[Dict]:
        """Get single page content"""
        try:
            logger.info(f"Fetching page: {page_title}")
            
            # Search for page
            search_results = wikipedia.search(page_title, results=1)
            if not search_results:
                logger.warning(f"Page not found: {page_title}")
                return None
            
            # Get page
            page = wikipedia.page(search_results[0])
            
            # Clean content
            cleaned_content = self.clean_text(page.content)
            summary = page.summary
            
            # Check content length
            if len(cleaned_content) < 100:
                logger.warning(f"Page content too short: {page_title}")
                return None
            
            return {
                'title': page.title,
                'url': page.url,
                'content': cleaned_content,
                'summary': summary,
                'categories': page.categories,
                'links': page.links[:10],  # Only take first 10 links
                'timestamp': datetime.now().isoformat()
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Page has disambiguation: {page_title}, options: {e.options[:5]}")
            return None
        except wikipedia.exceptions.PageError:
            logger.warning(f"Page does not exist: {page_title}")
            return None
        except Exception as e:
            logger.error(f"Error fetching page: {page_title}, error: {str(e)}")
            return None
    
    def search_and_fetch_pages(self, query: str, limit: int = 5) -> Dict:
        """Search and fetch pages based on query"""
        try:
            logger.info(f"Searching: {query}")
            
            # Search for related pages
            search_results = wikipedia.search(query, results=limit)
            logger.info(f"Found {len(search_results)} related pages")
            
            return self.fetch_and_save_pages(search_results)
            
        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            return {'total_requested': 0, 'successful': 0, 'failed': 0, 'pages': []}
    
    def get_popular_pages(self, limit: Optional[int] = None) -> List[str]:
        """Get popular page titles"""
        if limit is None:
            limit = self.max_pages
        
        try:
            # Get popular pages
            popular_pages = wikipedia.popular()
            logger.info(f"Retrieved {len(popular_pages)} popular pages")
            return popular_pages[:limit]
        except Exception as e:
            logger.error(f"Error getting popular pages: {str(e)}")
            return []
    
    def fetch_and_save_pages(self, page_titles: Optional[List[str]] = None) -> Dict:
        """Fetch and save page data"""
        if page_titles is None:
            page_titles = self.get_popular_pages()
        
        logger.info(f"Starting to fetch {len(page_titles)} pages")
        
        results = {
            'total_requested': len(page_titles),
            'successful': 0,
            'failed': 0,
            'pages': []
        }
        
        for i, title in enumerate(page_titles, 1):
            logger.info(f"Progress: {i}/{len(page_titles)} - {title}")
            
            page_data = self.get_page_content(title)
            if page_data:
                results['pages'].append(page_data)
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f"wikipedia_pages_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to: {output_file}")
        logger.info(f"Successful: {results['successful']}, Failed: {results['failed']}")
        
        # Add the saved file path to results
        results['saved_file'] = str(output_file)
        
        return results
    
    def fetch_by_category(self, category: str, limit: int = 10) -> Dict:
        """Fetch pages by category"""
        try:
            logger.info(f"Fetching pages for category '{category}'")
            
            # Get pages under category
            category_pages = wikipedia.search(category, results=limit)
            
            return self.fetch_and_save_pages(category_pages)
            
        except Exception as e:
            logger.error(f"Error fetching category pages: {str(e)}")
            return {'total_requested': 0, 'successful': 0, 'failed': 0, 'pages': []}


def main():
    """Main function - testing phase"""
    logger.info("Starting Wikipedia data fetching test")
    
    fetcher = WikipediaFetcher()
    
    # Test: Search pages by query (only take 1 page)
    test_query = "artificial intelligence"
    logger.info(f"Test query: {test_query}")
    
    results = fetcher.search_and_fetch_pages(test_query, limit=1)
    
    logger.info("Wikipedia data fetching test completed")

    # Cleanup: delete the generated data file
    try:
        if results and 'pages' in results and results['pages']:
            # Find the latest generated file
            data_dir = Path(fetcher.data_dir)
            wiki_files = list(data_dir.glob("wikipedia_pages_*.json"))
            if wiki_files:
                latest_file = max(wiki_files, key=lambda x: x.stat().st_mtime)
                latest_file.unlink()
                logger.info(f"Deleted test data file: {latest_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    
    return results


if __name__ == "__main__":
    main() 