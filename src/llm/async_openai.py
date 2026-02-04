
import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import tiktoken

from .base import BaseLLM, LLMResponse

class AsyncOpenAIClient(BaseLLM):
    """Async client for interacting with OpenAI API."""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Name of OpenAI model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
            
        self.model_name = model_name or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    async def generate_async(
        self,
        prompt: str,
        strategy: str = "unknown",
        max_tokens: int = 20,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from OpenAI async.
        """
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content or ""
            
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                response_tokens = response.usage.completion_tokens
            else:
                prompt_tokens = len(self.encoder.encode(prompt))
                response_tokens = len(self.encoder.encode(response_text))
                
            return LLMResponse(
                text=response_text,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                latency_ms=latency,
                success=True,
                model=self.model_name,
                strategy=strategy
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                text="",
                prompt_tokens=0,
                response_tokens=0,
                latency_ms=latency,
                success=False,
                error=str(e),
                model=self.model_name,
                strategy=strategy
            )

    def generate(self, *args, **kwargs):
        """Sync wrapper if needed (usually not used in async flow)."""
        # Not implemented for this async-first client or use asyncio.run
        raise NotImplementedError("Use generate_async with AsyncOpenAIClient")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
