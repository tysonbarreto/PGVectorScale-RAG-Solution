import logging
from logging import StreamHandler, FileHandler
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional
import sys, os


from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

def AILogger():
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                    logging.FileHandler(os.path.join("logs", f"{datetime.now().strftime('%m_%d_%Y')}.log"))])

    return logging.getLogger("AILoggger")


class LLMSettings(BaseModel):
    """Settings LLM Configuration"""
    temperature:float = 0.0
    max_tokens: Optional[int] = None
    max_retries:int = 3

class OpenAISettings(BaseModel):
    """OpenAI Specific Settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")


class DatabaseSettings(BaseModel):
    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))

class VectoreStoreSettings(BaseModel):
    table_name: str = "embeddings"
    embedding_dimensions: int = Field(default=1536)
    time_parition_interval: timedelta =timedelta(days=7)

class Settings(BaseModel):
    openai:OpenAISettings=Field(default_factory=OpenAISettings)
    database:DatabaseSettings=Field(default_factory=DatabaseSettings)
    vector_store:VectoreStoreSettings=Field(default_factory=VectoreStoreSettings)


@lru_cache()
def get_settings()->Settings:
    settings=Settings()
    AILogger()
    return settings

if __name__ == "__main__":
    __all__=["setup_logging","LLMSettings","OpenAISettings","DatabaseSettings","VectoreStoreSettings","Settings","get_settings"]