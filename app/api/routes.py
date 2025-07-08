from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
from app.config import get_settings
from app.api.batch import batch_processor

router = APIRouter()

# API statistics tracking
api_stats = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "start_time": datetime.now(),
    "last_query_time": None
}

class QueryRequest(BaseModel):
    query: str = "What is the capital of Taiwan?"
    class Config:
        schema_extra = {
            "example": {"query": "Who is the current president of Taiwan?"}
        }

class BatchQueryRequest(BaseModel):
    queries: List[str] = ["What is the capital of Taiwan?", "Who are the presidents of Taiwan?"]
    class Config:
        schema_extra = {
            "example": {"queries": ["What is the capital of Taiwan?", "Who are the presidents of Taiwan?"]}
        }

class QueryResponse(BaseModel):
    input: str
    output: str
    processing_time: float
    timestamp: datetime
    class Config:
        schema_extra = {
            "example": {
                "input": "Who is the current president of Taiwan?",
                "output": "The current president of Taiwan is Lai Ching-te (as of 2024).",
                "processing_time": 0.42,
                "timestamp": "2024-07-08T12:34:56.789Z"
            }
        }

class BatchQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_queries: int
    successful_queries: int
    failed_queries: int
    processing_time: float
    timestamp: datetime
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "query": "What is the capital of Taiwan?",
                        "success": True,
                        "input": "What is the capital of Taiwan?",
                        "output": "The capital of Taiwan is Taipei.",
                        "processing_time": 0.35,
                        "timestamp": "2024-07-08T12:34:56.789Z"
                    },
                    {
                        "query": "Who are the presidents of Taiwan?",
                        "success": True,
                        "input": "Who are the presidents of Taiwan?",
                        "output": "The presidents of Taiwan include Chiang Kai-shek, Lee Teng-hui, ...",
                        "processing_time": 0.51,
                        "timestamp": "2024-07-08T12:34:57.123Z"
                    }
                ],
                "total_queries": 2,
                "successful_queries": 2,
                "failed_queries": 0,
                "processing_time": 0.86,
                "timestamp": "2024-07-08T12:34:57.123Z"
            }
        }

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Single RAG Query",
    description="Submit a single question to the RAG system and receive an answer generated from Wikipedia-based retrieval and LLM.",
    responses={
        200: {"description": "Successful response with answer."},
        400: {"description": "Invalid request."},
        500: {"description": "Internal server error."}
    },
)
async def query_rag(request: QueryRequest):
    """
    Main RAG query endpoint.
    
    Args:
        request (QueryRequest): The query request containing a single question.
    Returns:
        QueryResponse: The answer and metadata.
    """
    api_stats["total_queries"] += 1
    api_stats["last_query_time"] = datetime.now()
    
    try:
        # Use the batch processor's single query function
        result = await batch_processor.process_single_query(request.query)
        api_stats["successful_queries"] += 1
        
        # Convert the result to the expected response format
        response_data = {
            "input": result["input"],
            "output": result["output"],
            "processing_time": result["processing_time"],
            "timestamp": datetime.fromisoformat(result["timestamp"])
        }
            
        return response_data
        
    except Exception as e:
        api_stats["failed_queries"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/batch-query",
    response_model=BatchQueryResponse,
    summary="Batch RAG Query",
    description="Submit multiple questions in a single request. Each question will be processed in parallel and results returned as a list.",
    responses={
        200: {"description": "Successful response with batch results."},
        400: {"description": "Invalid request."},
        500: {"description": "Internal server error."}
    },
)
async def batch_query_rag(request: BatchQueryRequest):
    """
    Batch query endpoint for processing multiple questions.
    
    Args:
        request (BatchQueryRequest): The batch query request containing a list of questions.
    Returns:
        BatchQueryResponse: The list of answers and metadata for each question.
    """
    start_time = time.time()
    
    try:
        # Process batch queries
        results = await batch_processor.process_batch(request.queries)
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        successful_queries = sum(1 for r in results if r.get("success", False))
        failed_queries = len(results) - successful_queries
        
        return {
            "results": results,
            "total_queries": len(request.queries),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get(
    "/status",
    summary="System Status and Statistics",
    description="Get current system status, API statistics, and configuration.",
    responses={
        200: {"description": "System status and statistics."},
        500: {"description": "Internal server error."}
    },
)
def system_status():
    """
    System status and statistics endpoint.
    Returns system info, API usage statistics, and configuration.
    """
    settings = get_settings()
    uptime = datetime.now() - api_stats["start_time"]
    
    return {
        "system_info": {
            "app_name": settings.app_name,
            "version": settings.app_version,
            "uptime_seconds": uptime.total_seconds(),
            "start_time": api_stats["start_time"]
        },
        "api_statistics": {
            "total_queries": api_stats["total_queries"],
            "successful_queries": api_stats["successful_queries"],
            "failed_queries": api_stats["failed_queries"],
            "success_rate": api_stats["successful_queries"] / max(api_stats["total_queries"], 1),
            "last_query_time": api_stats["last_query_time"]
        },
        "configuration": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "wiki_max_pages": settings.wiki_max_pages,
            "wiki_language": settings.wiki_language
        }
    }

@router.get(
    "/config",
    summary="Get Current Configuration",
    description="Get current system configuration (excluding sensitive data).",
    responses={
        200: {"description": "Current configuration."},
        500: {"description": "Internal server error."}
    },
)
def get_configuration():
    """
    Get current configuration (without sensitive data).
    Returns retrieval, Wikipedia, and storage settings.
    """
    settings = get_settings()
    return {
        "retrieval_settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "rerank_top_n": settings.rerank_top_n
        },
        "wikipedia_settings": {
            "language": settings.wiki_language,
            "max_pages": settings.wiki_max_pages
        },
        "storage_settings": {
            "vector_db_path": settings.vector_db_path,
            "data_dir": settings.data_dir
        }
    }

@router.post(
    "/reset-stats",
    summary="Reset API Statistics",
    description="Reset all API usage statistics to zero.",
    responses={
        200: {"description": "Statistics reset successfully."},
        500: {"description": "Internal server error."}
    },
)
def reset_statistics():
    """
    Reset API statistics endpoint.
    Resets all API usage counters and timestamps.
    """
    global api_stats
    api_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "start_time": datetime.now(),
        "last_query_time": None
    }
    return {"message": "Statistics reset successfully", "timestamp": datetime.now()}
        