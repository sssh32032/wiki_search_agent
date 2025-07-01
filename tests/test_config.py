"""
Unit tests for app.config module
"""

import pytest
import os
from unittest.mock import patch
from app.config import settings


class TestSettings:
    """Test settings configuration"""
    
    def test_default_settings(self):
        """Test default settings values"""
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'cohere_api_key')
        assert hasattr(settings, 'wikipedia_language')
        assert hasattr(settings, 'vector_db_path')
        assert hasattr(settings, 'data_dir')
        assert hasattr(settings, 'log_dir')
    
    def test_wikipedia_language_default(self):
        """Test default Wikipedia language"""
        assert settings.wikipedia_language == "en"
    
    def test_paths_are_pathlib_objects(self):
        """Test that path settings are Path objects"""
        from pathlib import Path
        assert isinstance(settings.vector_db_path, Path)
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.log_dir, Path)
    
    @patch.dict(os.environ, {
        'WIKIPEDIA_LANGUAGE': 'zh',
        'VECTOR_DB_PATH': '/custom/path',
        'DATA_DIR': '/custom/data',
        'LOG_DIR': '/custom/logs'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults"""
        # Reimport settings to get updated values
        import importlib
        import app.config
        importlib.reload(app.config)
        
        assert app.config.settings.wikipedia_language == "zh"
        assert str(app.config.settings.vector_db_path) == "/custom/path"
        assert str(app.config.settings.data_dir) == "/custom/data"
        assert str(app.config.settings.log_dir) == "/custom/logs"
    
    def test_api_keys_handling(self):
        """Test API key handling"""
        # Should handle missing API keys gracefully
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'cohere_api_key')
    
    def test_settings_immutability(self):
        """Test that settings are immutable"""
        with pytest.raises(Exception):
            settings.wikipedia_language = "fr" 