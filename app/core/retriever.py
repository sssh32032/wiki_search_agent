"""
Retriever module
Responsible for retrieving relevant documents from vector database and implementing reranking functionality
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever class"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever
        
        Args:
            model_name: Sentence embedding model name
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = None
        self.cross_encoder_model = None
        self.tokenizer = None
        self._load_model()
        self._load_index()
    
    def _load_model(self):
        """Load sentence embedding model"""
        try:
            logger.info(f"Loading sentence embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_index(self):
        """Load FAISS index and metadata"""
        try:
            # Find latest index file
            index_dir = Path(settings.vector_db_path)
            index_files = list(index_dir.glob("faiss_index_*.index"))
            
            if not index_files:
                raise FileNotFoundError("FAISS index file not found")
            
            # Use latest index file
            latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_index.stem.split("_")[-1]
            
            # Load index
            logger.info(f"Loading FAISS index: {latest_index}")
            self.index = faiss.read_index(str(latest_index))
            
            # Load metadata
            metadata_file = index_dir / f"metadata_{timestamp}.json"
            logger.info(f"Loading metadata: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Index loaded successfully, contains {len(self.metadata)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def _load_rerankers(self):
        """Load quantized cross-encoder model using Hugging Face transformers and bitsandbytes (lazy load)"""
        if self.cross_encoder_model is not None and self.tokenizer is not None:
            return
        try:
            logger.info("Loading quantized cross-encoder model...")
            self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L2-v2')
            self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(
                'cross-encoder/ms-marco-TinyBERT-L2-v2',
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.cross_encoder_model.eval()
            logger.info("Quantized cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load quantized cross-encoder model: {str(e)}")
            self.cross_encoder_model = None
            self.tokenizer = None
    
    def retrieve(self, query: str, k: int = 10, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query text
            k: Number of documents to return
            threshold: Similarity threshold
            
        Returns:
            List of relevant documents
        """
        try:
            # Convert query to vector
            query_embedding = self.model.encode([query])
            
            # Search for similar documents
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= threshold and idx != -1:
                    doc_metadata = self.metadata[idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'text': doc_metadata['text'],
                        'metadata': {
                            'title': doc_metadata['title'],
                            'chunk_index': doc_metadata['chunk_index'],
                            'url': doc_metadata.get('url', ''),
                            'source': doc_metadata.get('source', '')
                        }
                    })
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Rerank documents
        
        Args:
            query: Query text
            documents: List of documents
            method: Reranking method ("hybrid", "keyword", "cross_encoder")
            
        Returns:
            Reranked document list
        """
        try:
            if method == "hybrid":
                self._load_rerankers()
                return self._hybrid_rerank(query, documents)
            elif method == "keyword":
                return self._keyword_rerank(query, documents)
            elif method == "cross_encoder":
                self._load_rerankers()
                return self._cross_encoder_rerank(query, documents)
            else:
                logger.warning(f"Unknown reranking method: {method}, skipping reranking and returning original documents")
                return documents
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents
    
    def _keyword_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keyword-based reranking"""
        try:
            # Extract query keywords
            query_words = set(query.lower().split())
            
            reranked_docs = []
            for doc in documents:
                doc_copy = doc.copy()
                text_words = set(doc['text'].lower().split())
                
                # Calculate keyword match score
                keyword_matches = len(query_words & text_words)
                keyword_coverage = keyword_matches / len(query_words) if query_words else 0
                
                # Calculate keyword density
                text_length = len(doc['text'])
                keyword_density = keyword_matches / text_length if text_length > 0 else 0
                
                # Combined keyword score
                keyword_score = 0.6 * keyword_coverage + 0.4 * keyword_density
                
                doc_copy['rerank_score'] = keyword_score
                doc_copy['keyword_matches'] = keyword_matches
                doc_copy['keyword_coverage'] = keyword_coverage
                reranked_docs.append(doc_copy)
            
            # Sort by keyword score
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Update rankings
            for i, doc in enumerate(reranked_docs):
                doc['rank'] = i + 1
            
            logger.info(f"Keyword reranking completed, processed {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Keyword reranking failed: {str(e)}")
            return documents
    
    def _batched_cross_encoder_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Batch compute cross-encoder scores for (query, doc['text']) pairs."""
        if not self.cross_encoder_model or not self.tokenizer:
            return [0.0] * len(documents)
        scores = []
        batch_size = 4
        try:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_pairs = [(query, doc['text']) for doc in batch_docs]
                features = self.tokenizer(
                    [pair[0] for pair in batch_pairs],
                    [pair[1] for pair in batch_pairs],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                device = next(self.cross_encoder_model.parameters()).device
                features = {k: v.to(device) for k, v in features.items()}
                with torch.no_grad():
                    outputs = self.cross_encoder_model(**features)
                    batch_scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
                    if len(batch_docs) == 1:
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {str(e)}")
            return [0.0] * len(documents)

    def _cross_encoder_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank using quantized Hugging Face cross-encoder model"""
        if not self.cross_encoder_model or not self.tokenizer:
            logger.warning("Quantized cross-encoder model not loaded.")
            return documents
        try:
            scores = self._batched_cross_encoder_scores(query, documents)
            
            # Normalize scores for consistency
            def norm(x):
                x = np.array(x)
                if x.max() == x.min():
                    return np.zeros_like(x)
                return (x - x.min()) / (x.max() - x.min())
            
            norm_scores = norm(scores)
            
            for doc, score, norm_score in zip(documents, scores, norm_scores):
                doc['cross_encoder_score'] = float(norm_score)
                doc['original_cross_encoder_score'] = float(score)  # Keep original for reference
                
            reranked_docs = sorted(documents, key=lambda x: x['cross_encoder_score'], reverse=True)
            for i, doc in enumerate(reranked_docs):
                doc['rank'] = i + 1
            logger.info(f"Quantized cross-encoder reranking completed, processed {len(reranked_docs)} documents")
            return reranked_docs
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}")
            return documents

    def _hybrid_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hybrid reranking: combine cosine similarity and quantized cross-encoder scores"""
        semantic_scores = [doc['score'] for doc in documents]
        if self.cross_encoder_model and self.tokenizer:
            cross_scores = self._batched_cross_encoder_scores(query, documents)
        else:
            cross_scores = [0.0] * len(documents)
        def norm(x):
            x = np.array(x)
            if x.max() == x.min():
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())
        norm_sem = norm(semantic_scores)
        norm_cross = norm(cross_scores)
        for i, doc in enumerate(documents):
            doc['hybrid_score'] = float(0.5 * norm_sem[i] + 0.5 * norm_cross[i])
            doc['semantic_score'] = float(norm_sem[i])
            doc['cross_encoder_score'] = float(norm_cross[i])
            # Keep original scores for reference
            doc['original_semantic_score'] = float(semantic_scores[i])
            doc['original_cross_encoder_score'] = float(cross_scores[i])
        reranked_docs = sorted(documents, key=lambda x: x['hybrid_score'], reverse=True)
        for i, doc in enumerate(reranked_docs):
            doc['rank'] = i + 1
        logger.info(f"Hybrid reranking (cosine + quantized cross-encoder) completed, processed {len(reranked_docs)} documents")
        return reranked_docs
    
    def search_and_rerank(self, query: str, k: int = 10, 
                         rerank_method: str = "hybrid", 
                         cutoff_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search and rerank (one-stop method)
        
        Args:
            query: Query text
            k: Initial retrieval count
            rerank_method: Reranking method
            cutoff_threshold: Minimum score threshold for relevance (0.0-1.0)
            
        Returns:
            Retrieved and reranked document list
        """
        try:
            # Initial retrieval
            initial_results = self.retrieve(query, k=k)
            
            if not initial_results:
                logger.warning("No relevant documents retrieved")
                return []
            
            # Reranking
            reranked_results = self.rerank(query, initial_results, method=rerank_method)
            
            # Apply cutoff threshold
            if cutoff_threshold > 0.0:
                # Determine which score to use based on rerank method
                if rerank_method == "hybrid":
                    score_key = "hybrid_score"
                elif rerank_method == "cross_encoder":
                    score_key = "cross_encoder_score"
                elif rerank_method == "keyword":
                    score_key = "keyword_coverage"
                else:
                    score_key = "score"  # Default to semantic similarity score
                
                # Filter results above threshold
                filtered_results = [
                    result for result in reranked_results 
                    if result.get(score_key, result.get('score', 0.0)) >= cutoff_threshold
                ]
                
                logger.info(f"Applied cutoff threshold {cutoff_threshold} using {score_key}")
                logger.info(f"Before cutoff: {len(reranked_results)} documents")
                logger.info(f"After cutoff: {len(filtered_results)} documents")
                
                if not filtered_results:
                    logger.warning(f"No documents meet the cutoff threshold {cutoff_threshold}")
                    return []
                
                reranked_results = filtered_results
            
            logger.info(f"Search and rerank completed, returning {len(reranked_results)} documents")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Search and rerank failed: {str(e)}")
            return []


def main():
    """Test retriever functionality"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/retriever.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    created_data = None
    created_index = None
    created_metadata = None
    try:
        # Check for FAISS index first
        data_dir = Path(settings.data_dir)
        index_dir = Path(settings.vector_db_path)
        index_files = list(index_dir.glob("faiss_index_*.index"))
        latest_file = None
        timestamp = None
        if not index_files:
            logger.warning("FAISS index file not found, will attempt to create one for testing...")
            # Check for Wikipedia data file
            wiki_files = list(data_dir.glob("wikipedia_pages_*.json"))
            if not wiki_files:
                logger.warning("Wikipedia data file not found, creating one for testing...")
                from scripts.fetch_wiki import WikipediaFetcher
                fetcher = WikipediaFetcher()
                fetcher.search_and_fetch_pages("artificial intelligence", limit=1)
                wiki_files = list(data_dir.glob("wikipedia_pages_*.json"))
                if not wiki_files:
                    logger.error("Failed to create Wikipedia data file for testing")
                    return
                created_data = max(wiki_files, key=lambda x: x.stat().st_mtime)
            latest_file = max(wiki_files, key=lambda x: x.stat().st_mtime)
            from scripts.build_embeddings import EmbeddingProcessor
            processor = EmbeddingProcessor()
            processor.build_index(latest_file.name)
            # Add index and metadata to cleanup
            timestamp = latest_file.name.replace('.json', '').split('_')[-1]
            created_index = index_dir / f"faiss_index_{timestamp}.index"
            created_metadata = index_dir / f"metadata_{timestamp}.json"
        else:
            # Use the latest index for possible cleanup (if needed)
            latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_index.stem.split('_')[-1]
            latest_file = None  # Not created in this run
        # ---
        logger.info("Starting retriever functionality test")
        # Initialize retriever
        retriever = Retriever()
        # Test queries
        test_queries = [
            "What is the definition of artificial intelligence?"
        ]
        for query in test_queries:
            logger.info(f"Test query: {query}")
            # One-stop search and rerank
            logger.info("Executing one-stop search and rerank...")
            final_results = retriever.search_and_rerank(query, k=5, rerank_method="default", cutoff_threshold=0.3)
            logger.info(f"Final result count: {len(final_results)}")
            if final_results:
                logger.info("Top 3 results:")
                for i, result in enumerate(final_results[:3]):
                    logger.info(f"  {i+1}. Hybrid score: {result.get('hybrid_score', result['score']):.4f}")
                    logger.info(f"     Semantic score: {result.get('semantic_score', result['score']):.4f}")
                    logger.info(f"     Keyword coverage: {result.get('keyword_coverage', 0):.4f}")
                    logger.info(f"     Cross-encoder score: {result.get('cross_encoder_score', 0):.4f}")
                    logger.info(f"     Title: {result['metadata']['title']}")
                    logger.info(f"     Text: {result['text'][:80]}...")
            logger.info("-" * 50)
        logger.info("Retriever functionality test completed")
    except Exception as e:
        logger.error(f"Error occurred during testing: {str(e)}")
        raise
    finally:
        # Cleanup only files created in this run
        for file_path in [created_data, created_index, created_metadata]:
            try:
                if file_path and isinstance(file_path, Path) and file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted test file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting test file {file_path}: {str(e)}")


if __name__ == "__main__":
    main() 