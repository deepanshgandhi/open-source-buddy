from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GH_TOKEN: str
    OPENAI_API_KEY: str
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_MODEL: str = "gpt-4o-mini"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings() 