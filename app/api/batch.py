"""
Batch processing module for handling multiple queries
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from app.core.llm_chain import WikipediaRAGChain

class BatchProcessor:
    """Handles batch processing of multiple queries"""
    
    def __init__(self):
        self.rag = WikipediaRAGChain()
    
    async def process_single_query(self, query: str) -> Dict[str, Any]:
        """Process a single query asynchronously"""
        # Run the RAG pipeline
        start_time = datetime.now()

        #not sure if this is thread-safe, leave it without await asyncio.to_thread
        answer = self.rag.generate(query)

        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "input": query,
            "output": answer,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    async def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of queries asynchronously"""
        tasks = []
        for query in queries:
            task = self.process_single_query(query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": queries[i],
                    "success": False,
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # Type check to ensure result is a dictionary
                if isinstance(result, dict):
                    processed_results.append({
                        "query": queries[i],
                        "success": True,
                        "input": result.get("input", ""),
                        "output": result.get("output", ""),
                        "processing_time": result.get("processing_time", 0),
                        "timestamp": result.get("timestamp", datetime.now().isoformat())
                    })
                else:
                    processed_results.append({
                        "query": queries[i],
                        "success": False,
                        "error": f"Unexpected result type: {type(result)}",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return processed_results

# Global batch processor instance
batch_processor = BatchProcessor() 