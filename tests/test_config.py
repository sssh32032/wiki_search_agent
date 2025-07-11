"""
Unit tests for configuration module
Tests each function and setting individually
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
from app.config import Settings, get_settings, validate_api_keys, create_directories

class TestSettingsClass:
    """Test Settings class creation and attributes"""
    
    def test_settings_creation(self):
        """Test that settings can be created"""
        settings = Settings()
        assert isinstance(settings, Settings)
        assert hasattr(settings, 'cohere_api_key')
    
    def test_settings_required_fields(self):
        """Test that all required settings fields exist"""
        settings = Settings()
        
        # API key settings
        assert hasattr(settings, 'cohere_api_key')
        
        # Vector database settings
        assert hasattr(settings, 'vector_db_path')
        assert hasattr(settings, 'chunk_size')
        assert hasattr(settings, 'chunk_overlap')
        
        # Wikipedia settings
        assert hasattr(settings, 'wiki_language')
        assert hasattr(settings, 'wiki_max_pages')
        
        # Retrieval settings
        assert hasattr(settings, 'top_k')
        assert hasattr(settings, 'similarity_threshold')
        assert hasattr(settings, 'rerank_top_n')
        
        # Application settings
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'app_version')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'data_dir')
    
    def test_settings_default_values(self):
        """Test that settings have appropriate default values"""
        # Clear any existing environment variables that might affect defaults
        env_vars_to_clear = ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "WIKI_MAX_PAGES", 
                            "WIKI_LANGUAGE", "RERANK_TOP_N", "SIMILARITY_THRESHOLD", 
                            "APP_NAME", "APP_VERSION", "DEBUG", "DATA_DIR", "VECTOR_DB_PATH"]
        
        original_env = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                original_env[var] = os.environ[var]
                del os.environ[var]
        
        try:
            settings = Settings()
            
            # Test default values (these should match env.example)
            assert settings.chunk_size == 1000
            assert settings.chunk_overlap == 200
            assert settings.top_k == 5
            assert settings.wiki_max_pages == 100
            assert settings.wiki_language == "en"
            assert settings.rerank_top_n == 3
            assert settings.similarity_threshold == 0.7
            assert settings.app_name == "Wikipedia Assistant"
            assert settings.app_version == "1.0.0"
            assert settings.debug is False
            assert settings.data_dir == "./data"
            assert settings.vector_db_path == "./faiss_index"
        finally:
            # Restore original environment
            for var, value in original_env.items():
                os.environ[var] = value
    
    def test_settings_field_types(self):
        """Test that settings fields have correct types"""
        settings = Settings()
        
        assert isinstance(settings.cohere_api_key, str)
        assert isinstance(settings.vector_db_path, str)
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.chunk_overlap, int)
        assert isinstance(settings.wiki_language, str)
        assert isinstance(settings.wiki_max_pages, int)
        assert isinstance(settings.top_k, int)
        assert isinstance(settings.similarity_threshold, float)
        assert isinstance(settings.rerank_top_n, int)
        assert isinstance(settings.app_name, str)
        assert isinstance(settings.app_version, str)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.data_dir, str)

class TestEnvironmentVariableOverride:
    """Test environment variable override functionality"""
    
    def test_environment_variable_override(self):
        """Test that environment variables can override defaults"""
        # Set environment variables
        os.environ["CHUNK_SIZE"] = "500"
        os.environ["TOP_K"] = "10"
        os.environ["WIKI_LANGUAGE"] = "zh"
        os.environ["SIMILARITY_THRESHOLD"] = "0.8"
        os.environ["DEBUG"] = "true"
        
        # Create new settings instance
        settings = Settings()
        
        # Check that environment variables are used
        assert settings.chunk_size == 500
        assert settings.top_k == 10
        assert settings.wiki_language == "zh"
        assert settings.similarity_threshold == 0.8
        assert settings.debug is True
        
        # Clean up
        del os.environ["CHUNK_SIZE"]
        del os.environ["TOP_K"]
        del os.environ["WIKI_LANGUAGE"]
        del os.environ["SIMILARITY_THRESHOLD"]
        del os.environ["DEBUG"]
    
    def test_environment_variable_invalid_values(self):
        """Test handling of invalid environment variable values"""
        # Note: The current system doesn't handle invalid values gracefully
        # This test documents the current behavior
        with pytest.raises(ValueError):
            # Set invalid environment variables
            os.environ["CHUNK_SIZE"] = "invalid"
            os.environ["TOP_K"] = "not_a_number"
            os.environ["SIMILARITY_THRESHOLD"] = "invalid_float"
            
            # This should raise ValueError
            settings = Settings()
        
        # Clean up
        if "CHUNK_SIZE" in os.environ:
            del os.environ["CHUNK_SIZE"]
        if "TOP_K" in os.environ:
            del os.environ["TOP_K"]
        if "SIMILARITY_THRESHOLD" in os.environ:
            del os.environ["SIMILARITY_THRESHOLD"]
    
    def test_environment_variable_empty_values(self):
        """Test handling of empty environment variable values"""
        # Set empty environment variables
        os.environ["COHERE_API_KEY"] = ""
        os.environ["APP_NAME"] = ""
        
        # Create new settings instance
        settings = Settings()
        
        # Check that empty values are handled correctly
        assert settings.cohere_api_key == ""
        assert settings.app_name == ""  # Empty environment variable overrides default
        
        # Clean up
        if "COHERE_API_KEY" in os.environ:
            del os.environ["COHERE_API_KEY"]
        if "APP_NAME" in os.environ:
            del os.environ["APP_NAME"]

class TestGetSettingsFunction:
    """Test get_settings function"""
    
    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance (singleton pattern)"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

class TestValidateApiKeysFunction:
    """Test validate_api_keys function"""
    
    @patch('app.config.settings')
    def test_validate_api_keys_success(self, mock_settings):
        """Test successful API key validation"""
        # Mock settings with valid API key
        mock_settings.cohere_api_key = "valid_api_key"
        
        result = validate_api_keys()
        assert result is True
    
    @patch('app.config.settings')
    def test_validate_api_keys_missing_cohere_key(self, mock_settings):
        """Test API key validation with missing Cohere API key"""
        # Mock settings with missing API key
        mock_settings.cohere_api_key = ""
        
        result = validate_api_keys()
        assert result is False
    
    @patch('app.config.settings')
    def test_validate_api_keys_none_cohere_key(self, mock_settings):
        """Test API key validation with None Cohere API key"""
        # Mock settings with None API key
        mock_settings.cohere_api_key = None
        
        result = validate_api_keys()
        assert result is False

class TestCreateDirectoriesFunction:
    """Test create_directories function"""
    
    def test_create_directories_success(self):
        """Test successful directory creation"""
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings to use temporary directory
            with patch('app.config.settings') as mock_settings:
                mock_settings.data_dir = os.path.join(temp_dir, "data")
                mock_settings.vector_db_path = os.path.join(temp_dir, "faiss_index")
                
                # Create directories
                create_directories()
                
                # Check that directories exist
                assert os.path.exists(mock_settings.data_dir)
                assert os.path.exists(mock_settings.vector_db_path)
                assert os.path.exists("./logs")
    
    def test_create_directories_already_exist(self):
        """Test directory creation when directories already exist"""
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings to use temporary directory
            with patch('app.config.settings') as mock_settings:
                mock_settings.data_dir = os.path.join(temp_dir, "data")
                mock_settings.vector_db_path = os.path.join(temp_dir, "faiss_index")
                
                # Create directories first time
                create_directories()
                
                # Create directories again (should not fail)
                create_directories()
                
                # Check that directories still exist
                assert os.path.exists(mock_settings.data_dir)
                assert os.path.exists(mock_settings.vector_db_path)
                assert os.path.exists("./logs")
    
    def test_create_directories_permissions(self):
        """Test directory creation with proper permissions"""
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings to use temporary directory
            with patch('app.config.settings') as mock_settings:
                mock_settings.data_dir = os.path.join(temp_dir, "data")
                mock_settings.vector_db_path = os.path.join(temp_dir, "faiss_index")
                
                # Create directories
                create_directories()
                
                # Check that directories are writable
                assert os.access(mock_settings.data_dir, os.W_OK)
                assert os.access(mock_settings.vector_db_path, os.W_OK)
                assert os.access("./logs", os.W_OK)

class TestSettingsIntegration:
    """Integration tests for settings"""
    
    def test_settings_with_real_environment(self):
        """Test settings with real environment variables"""
        # Save original environment
        original_env = {}
        for key in ["COHERE_API_KEY", "CHUNK_SIZE", "TOP_K", "WIKI_LANGUAGE"]:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Set test environment variables
            os.environ["COHERE_API_KEY"] = "test_key_123"
            os.environ["CHUNK_SIZE"] = "750"
            os.environ["TOP_K"] = "8"
            os.environ["WIKI_LANGUAGE"] = "zh"
            
            # Create settings
            settings = Settings()
            
            # Verify settings
            assert settings.cohere_api_key == "test_key_123"
            assert settings.chunk_size == 750
            assert settings.top_k == 8
            assert settings.wiki_language == "zh"
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                os.environ[key] = value
            # Clean up test environment variables
            for key in ["COHERE_API_KEY", "CHUNK_SIZE", "TOP_K", "WIKI_LANGUAGE"]:
                if key in os.environ and key not in original_env:
                    del os.environ[key]
