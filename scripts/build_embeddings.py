#!/usr/bin/env python3
"""
Vectorization processing module
Slice, vectorize and store Wikipedia text to FAISS vector database
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project settings
from app.config import settings

try:
    import faiss
except ImportError:
    print("Error: Please install faiss-cpu package first")
    print("Run: pip install faiss-cpu")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Please install sentence-transformers package first")
    print("Run: pip install sentence-transformers")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embeddings.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunker"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Define sentence ending markers
        sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';', '\n\n']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            if char in sentence_endings:
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
        
        # Handle last sentence (if no ending marker)
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks, ensuring each chunk contains complete sentences
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # First split into sentences
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If a single sentence exceeds maximum length, use it as a chunk directly
            if sentence_length > self.chunk_size:
                # If current chunk has content, save it first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
                
                # Use the long sentence as an independent chunk
                chunks.append(sentence)
                continue
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk, need to handle overlap
                overlap_sentences = self._get_overlap_sentences(chunks, sentence_length)
                current_chunk = overlap_sentences + sentence
                current_length = len(current_chunk)
            else:
                # Can add current sentence
                current_chunk += sentence
                current_length += sentence_length
        
        # Save last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_sentences(self, chunks: List[str], new_sentence_length: int) -> str:
        """
        Select sentences to overlap from the previous chunk based on overlap requirements
        
        Args:
            chunks: List of existing chunks
            new_sentence_length: Length of new sentence
            
        Returns:
            Text of sentences to overlap
        """
        if not chunks:
            return ""
        
        last_chunk = chunks[-1]
        sentences = self.split_into_sentences(last_chunk)
        
        overlap_text = ""
        overlap_length = 0
        
        # From the last sentence, select sentences to overlap going backwards
        for sentence in reversed(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed overlap limit, stop
            if overlap_length + sentence_length > self.chunk_overlap:
                break
            
            overlap_text = sentence + overlap_text
            overlap_length += sentence_length
        
        return overlap_text


class EmbeddingProcessor:
    """Vectorization processor"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vectorization processor
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunker = TextChunker()
        
        # Ensure FAISS index directory exists
        self.index_dir = Path(settings.vector_db_path)
        self.index_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized vectorization processor with model: {model_name}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create vector embeddings for text list
        
        Args:
            texts: List of texts
            
        Returns:
            Vector embedding matrix
        """
        logger.info(f"Creating vector embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=False)  # Disable progress bar
        return embeddings
    
    def build_index(self, data_file: str) -> Dict:
        """
        Build vector index from Wikipedia data file
        
        Args:
            data_file: Path to Wikipedia JSON data file
            
        Returns:
            Processing result statistics
        """
        data_path = Path(settings.data_dir) / data_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Starting to process data file: {data_path}")
        
        # Read data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_chunks = []
        chunk_metadata = []
        
        # Process each page
        for page in data['pages']:
            title = page['title']
            content = page['content']
            summary = page['summary']
            url = page['url']
            
            logger.info(f"Processing page: {title}")
            
            # Split content into chunks
            content_chunks = self.chunker.split_text(content)
            
            # Add metadata for each chunk
            for i, chunk in enumerate(content_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'title': title,
                    'url': url,
                    'chunk_index': i,
                    'total_chunks': len(content_chunks),
                    'source': 'content',
                    'text': chunk  # Add original text content
                })
            
            # Also use summary as a chunk
            if summary and len(summary) > 50:
                all_chunks.append(summary)
                chunk_metadata.append({
                    'title': title,
                    'url': url,
                    'chunk_index': -1,  # -1 indicates summary
                    'total_chunks': len(content_chunks),
                    'source': 'summary',
                    'text': summary  # Add original text content
                })
        
        logger.info(f"Created {len(all_chunks)} text chunks in total")
        
        # Create vector embeddings
        embeddings = self.create_embeddings(all_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use inner product similarity
        
        # Normalize vectors (important for inner product index)
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        timestamp = data_file.replace('.json', '').split('_')[-1]
        index_file = self.index_dir / f"faiss_index_{timestamp}.index"
        metadata_file = self.index_dir / f"metadata_{timestamp}.json"
        
        faiss.write_index(index, str(index_file))
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vector index saved to: {index_file}")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return {
            'total_pages': len(data['pages']),
            'total_chunks': len(all_chunks),
            'embedding_dimension': dimension,
            'index_file': str(index_file),
            'metadata_file': str(metadata_file)
        }
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar texts
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of similar texts with text content and similarity scores
        """
        # Load latest index and metadata
        index_files = list(self.index_dir.glob("faiss_index_*.index"))
        if not index_files:
            raise FileNotFoundError("FAISS index file not found")
        
        # Use latest index
        latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
        metadata_file = latest_index.with_name(f"metadata_{latest_index.stem.split('_')[-1]}.json")
        
        # Load index
        index = faiss.read_index(str(latest_index))
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Create vector for query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar vectors
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'metadata': metadata[idx],
                    'text': metadata[idx]['text']  # Return text content directly
                }
                results.append(result)
        
        return results


def main():
    """Main function - test vectorization processing"""
    logger.info("Starting vectorization processing test")
    
    # Find latest Wikipedia data file
    data_dir = Path(settings.data_dir)
    wiki_files = list(data_dir.glob("wikipedia_pages_*.json"))
    created_files = []
    created_index = None
    created_metadata = None
    created_data = None
    
    if not wiki_files:
        logger.warning("Wikipedia data file not found, creating one for testing...")
        try:
            from scripts.fetch_wiki import WikipediaFetcher
            fetcher = WikipediaFetcher()
            fetcher.search_and_fetch_pages("artificial intelligence", limit=1)
            wiki_files = list(data_dir.glob("wikipedia_pages_*.json"))
            if not wiki_files:
                logger.error("Failed to create Wikipedia data file for testing")
                return
            created_data = max(wiki_files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            logger.error(f"Error creating Wikipedia data file: {str(e)}")
            return
    
    # Use latest file
    latest_file = max(wiki_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using data file: {latest_file.name}")
    
    # Create vectorization processor
    processor = EmbeddingProcessor()
    
    # Check if corresponding index already exists
    timestamp = latest_file.name.replace('.json', '').split('_')[-1]
    index_file = Path(settings.vector_db_path) / f"faiss_index_{timestamp}.index"
    metadata_file = Path(settings.vector_db_path) / f"metadata_{timestamp}.json"
    
    if index_file.exists():
        logger.info(f"Index already exists: {index_file}")
        logger.info("Skipping index creation, testing search function directly")
    else:
        logger.info("Index does not exist, starting to create new index")
        # Create index
        try:
            result = processor.build_index(latest_file.name)
            logger.info(f"Index creation completed: {result}")
            created_index = index_file
            created_metadata = metadata_file
        except Exception as e:
            logger.error(f"Error occurred while creating index: {str(e)}")
            return
    
    # Test search function
    try:
        test_query = "What is the definition of artificial intelligence?"
        logger.info("Starting to test search function")
        logger.info(f"Query: {test_query}")
        
        search_results = processor.search_similar(test_query, k=3)
        logger.info("Search results:")
        for result in search_results:
            logger.info(f"Rank {result['rank']}: Score {result['score']:.4f}")
            logger.info(f"Title: {result['metadata']['title']}")
            logger.info(f"Text: {result['text'][:100]}...")
        
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}")
    
    # Cleanup test data (only files created in this run)
    logger.info("Cleaning up test data...")
    try:
        if created_data and created_data.exists():
            created_data.unlink()
            logger.info(f"Deleted test data file: {created_data}")
        if created_index and created_index.exists():
            created_index.unlink()
            logger.info(f"Deleted test index file: {created_index}")
        if created_metadata and created_metadata.exists():
            created_metadata.unlink()
            logger.info(f"Deleted test metadata file: {created_metadata}")
        logger.info("Test data cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    
    logger.info("Vectorization processing test completed")


if __name__ == "__main__":
    main() 