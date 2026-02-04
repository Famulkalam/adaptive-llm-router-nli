"""
OpenAI API Client

Production client for OpenAI's API.
Requires OPENAI_API_KEY environment variable.
"""

import os
import time
from typing import Optional, Dict, Any

from .base import BaseLLM, LLMResponse, LLMError


class OpenAIClient(BaseLLM):
    """
    Production OpenAI API client.
    
    Uses the openai library for API calls.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model_name: Model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Prioritize constructor arg, then env var, then default
        model = model_name or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        super().__init__(model)
        
        if not self.api_key:
            raise LLMError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.",
                error_type="auth"
            )
        
        self._setup_client()
    
    def _setup_client(self):
        """Configure the OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMError(
                "openai package not installed. Run: pip install openai",
                error_type="import"
            )
        except Exception as e:
            raise LLMError(f"Failed to configure OpenAI client: {e}", error_type="config")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (approximation if not installed).
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to approximation (English avg: 4 chars/token)
            return len(text) // 4 + 1
        except Exception:
            return len(text) // 4 + 1
    
    def generate(
        self,
        prompt: str,
        strategy: str = "unknown",
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from OpenAI.
        
        Args:
            prompt: Input prompt
            strategy: Prompting strategy (for tracking)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            response_text = response.choices[0].message.content.strip() if response.choices else ""
            
            # Get token counts from response usage or fallback
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                response_tokens = response.usage.completion_tokens
            else:
                prompt_tokens = self.count_tokens(prompt)
                response_tokens = self.count_tokens(response_text)
            
            # Update stats
            self.call_count += 1
            self.total_tokens += prompt_tokens + response_tokens
            
            return LLMResponse(
                text=response_text,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                model=self.model_name,
                strategy=strategy,
                latency_ms=latency_ms,
                success=True,
                raw_response=response
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.errors.append(str(e))
            
            # Determine if retryable
            retryable = any(x in str(e).lower() for x in ["rate limit", "timeout", "unavailable", "500", "502", "503"])
            
            return LLMResponse(
                text="",
                prompt_tokens=self.count_tokens(prompt),
                response_tokens=0,
                model=self.model_name,
                strategy=strategy,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                raw_response=None
            )
