import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
import datetime
from app.api import routes

@pytest.mark.asyncio
async def test_query_rag_success():
    mock_result = {
        "input": "test query",
        "output": "test answer",
        "processing_time": 0.1,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with patch("app.api.routes.batch_processor.process_single_query", new=AsyncMock(return_value=mock_result)):
        request = routes.QueryRequest(query="test query")
        response = await routes.query_rag(request)
        assert response["input"] == "test query"
        assert response["output"] == "test answer"
        assert "processing_time" in response
        assert "timestamp" in response

@pytest.mark.asyncio
async def test_query_rag_failure():
    with patch("app.api.routes.batch_processor.process_single_query", new=AsyncMock(side_effect=Exception("fail"))):
        request = routes.QueryRequest(query="fail query")
        with pytest.raises(HTTPException) as exc:
            await routes.query_rag(request)
        assert exc.value.status_code == 500
        assert "fail" in str(exc.value.detail)

@pytest.mark.asyncio
async def test_batch_query_rag_success():
    mock_results = [
        {"query": "q1", "success": True, "input": "q1", "output": "a1", "processing_time": 0.1, "timestamp": datetime.datetime.now().isoformat()},
        {"query": "q2", "success": True, "input": "q2", "output": "a2", "processing_time": 0.2, "timestamp": datetime.datetime.now().isoformat()}
    ]
    with patch("app.api.routes.batch_processor.process_batch", new=AsyncMock(return_value=mock_results)):
        request = routes.BatchQueryRequest(queries=["q1", "q2"])
        response = await routes.batch_query_rag(request)
        assert response["total_queries"] == 2
        assert response["successful_queries"] == 2
        assert response["failed_queries"] == 0
        assert len(response["results"]) == 2

@pytest.mark.asyncio
async def test_batch_query_rag_failure():
    with patch("app.api.routes.batch_processor.process_batch", new=AsyncMock(side_effect=Exception("batch fail"))):
        request = routes.BatchQueryRequest(queries=["q1", "q2"])
        with pytest.raises(HTTPException) as exc:
            await routes.batch_query_rag(request)
        assert exc.value.status_code == 500
        assert "batch fail" in str(exc.value.detail)

def test_reset_statistics():
    # Call reset_statistics and check the return value
    result = routes.reset_statistics()
    assert result["message"] == "Statistics reset successfully"
    assert "timestamp" in result