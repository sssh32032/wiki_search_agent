import pytest
import requests

class APIClient:
    """API client for testing endpoints"""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health_check(self, timeout: int = 10) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            print("[DEBUG] test_health_check Exception:", e)
            return False

    def test_root_endpoint(self, timeout: int = 10) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/", timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            print("[DEBUG] test_root_endpoint Exception:", e)
            return False

    def test_status_endpoint(self, timeout: int = 10) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/status", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                assert "system_info" in data
                assert "api_statistics" in data
                return True
            print("[DEBUG] test_status_endpoint: Response status code:", response.status_code)
            print("[DEBUG] test_status_endpoint: Response text:", response.text)
            return False
        except Exception as e:
            print("[DEBUG] test_status_endpoint Exception:", e)
            return False

    def test_config_endpoint(self, timeout: int = 10) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/config", timeout=timeout)
            if response.status_code == 200:
                config = response.json()
                assert isinstance(config, dict)
                return True
            print("[DEBUG] test_config_endpoint: Response status code:", response.status_code)
            print("[DEBUG] test_config_endpoint: Response text:", response.text)
            return False
        except Exception as e:
            print("[DEBUG] test_config_endpoint Exception:", e)
            return False

    def test_single_query(self, query: str = "What is the tallest mountain in Taiwan?", timeout: int = 10) -> bool:
        try:
            payload = {"query": query}
            response = self.session.post(f"{self.base_url}/api/query", json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                assert "input" in data
                assert "output" in data
                assert "processing_time" in data
                return True
            print("[DEBUG] test_single_query: Response status code:", response.status_code)
            print("[DEBUG] test_single_query: Response text:", response.text)
            return False
        except Exception as e:
            print("[DEBUG] test_single_query Exception:", e)
            return False

    def test_batch_query(self, queries=None, timeout: int = 10) -> bool:
        try:
            if queries is None:
                queries = [
                    "What is the capital of Taiwan?",
                    "Who are the past presidents of Taiwan?",
                    "What is the population of Taiwan?"
                ]
            payload = {"queries": queries}
            response = self.session.post(f"{self.base_url}/api/batch-query", json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                assert "total_queries" in data
                assert "successful_queries" in data
                assert "failed_queries" in data
                assert "processing_time" in data
                assert "results" in data
                return True
            print("[DEBUG] test_batch_query: Response status code:", response.status_code)
            print("[DEBUG] test_batch_query: Response text:", response.text)
            return False
        except Exception as e:
            print("[DEBUG] test_batch_query Exception:", e)
            return False

    def test_reset_stats(self, timeout: int = 10) -> bool:
        try:
            response = self.session.post(f"{self.base_url}/api/reset-stats", timeout=timeout)
            if response.status_code == 200:
                return True
            print("[DEBUG] test_reset_stats: Response status code:", response.status_code)
            print("[DEBUG] test_reset_stats: Response text:", response.text)
            return False
        except Exception as e:
            print("[DEBUG] test_reset_stats Exception:", e)
            return False

@pytest.fixture
def api_client():
    return APIClient("http://localhost:8000")

class TestAPIIntegration:
    """Integration tests for API endpoints (requires running server)"""
    @pytest.mark.integration
    def test_health_check_integration(self, api_client, start_test_server):
        assert api_client.test_health_check(timeout=10)

    @pytest.mark.integration
    def test_root_endpoint_integration(self, api_client, start_test_server):
        assert api_client.test_root_endpoint(timeout=10)

    @pytest.mark.integration
    def test_status_endpoint_integration(self, api_client, start_test_server):
        assert api_client.test_status_endpoint(timeout=10)

    @pytest.mark.integration
    def test_config_endpoint_integration(self, api_client, start_test_server):
        assert api_client.test_config_endpoint(timeout=10)

    """
    @pytest.mark.integration
    def test_single_query_integration(self, api_client, start_test_server):
        assert api_client.test_single_query("台灣最高的山是哪座山?", timeout=30)

    @pytest.mark.integration
    def test_batch_query_integration(self, api_client, start_test_server):
        queries = [
            "台灣最高的山是哪座山?",
            "What is the capital of Taiwan?",
        ]
        assert api_client.test_batch_query(queries=queries, timeout=60)
    """

    @pytest.mark.integration
    def test_reset_stats_integration(self, api_client, start_test_server):
        assert api_client.test_reset_stats(timeout=10) 