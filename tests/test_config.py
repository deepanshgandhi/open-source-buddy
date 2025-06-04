import os
import pytest
from unittest.mock import patch

from app.config import Settings, get_settings


class TestSettings:
    @patch.dict(os.environ, {
        "GH_TOKEN": "test_gh_token",
        "OPENAI_API_KEY": "test_openai_key"
    })
    def test_settings_from_env(self):
        """Test that Settings loads correctly from environment variables."""
        settings = Settings()
        
        assert settings.GH_TOKEN == "test_gh_token"
        assert settings.OPENAI_API_KEY == "test_openai_key"
        assert settings.EMBED_MODEL == "all-MiniLM-L6-v2"
        assert settings.OPENAI_MODEL == "gpt-4o-mini"

    @patch.dict(os.environ, {
        "GH_TOKEN": "cached_gh_token",
        "OPENAI_API_KEY": "cached_openai_key"
    })
    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        # Clear any existing cache
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should return the same instance due to lru_cache
        assert settings1 is settings2
        assert settings1.GH_TOKEN == "cached_gh_token"
        assert settings1.OPENAI_API_KEY == "cached_openai_key"

    def test_settings_missing_required_env(self):
        """Test that Settings raises error when required env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # pydantic.ValidationError
                Settings()

    @patch.dict(os.environ, {
        "GH_TOKEN": "override_gh_token",
        "OPENAI_API_KEY": "override_openai_key",
        "EMBED_MODEL": "custom-model",
        "OPENAI_MODEL": "gpt-4"
    })
    def test_settings_override_defaults(self):
        """Test that environment variables can override default values."""
        settings = Settings()
        
        assert settings.GH_TOKEN == "override_gh_token"
        assert settings.OPENAI_API_KEY == "override_openai_key"
        assert settings.EMBED_MODEL == "custom-model"
        assert settings.OPENAI_MODEL == "gpt-4" 