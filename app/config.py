from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GH_TOKEN: str
    OPENAI_API_KEY: str
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_MODEL: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings() 