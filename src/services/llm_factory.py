from typing import Any, Dict, List, Type
import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel
from src.config.settings import get_settings
import os, sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class LLMFactory:
    def __init__(self, provider:str="openai"): #specify the llm provider name
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client(provider)
    
    def _initialize_client(self)->Any:
        client_initializers = {
            "openai":lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
            "anthropic": lambda s: instructor.from_anthropic(Anthropic(api_key=s.api_key)),
            "llama": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key, base_url=s.base_url), mode=instructor.Mode.JSON)
        }
        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def create_completion(self,response_model:Type[BaseModel],messages:List[Dict[str,str]],**kwargs)->Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create(**completion_params)
    
if __name__ == "__main__":
    __all__=["LLMFactory"]