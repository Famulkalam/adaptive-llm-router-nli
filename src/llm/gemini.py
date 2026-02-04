"""
Real Gemini API Client

Production client for Google's Gemini API.
Requires GEMINI_API_KEY environment variable.
"""

import os
import time
from typing import Optional, Dict, Any

from .base import BaseLLM, LLMResponse, LLMError


class GeminiClient(BaseLLM):
    """
    Production Gemini API client.
    
    Uses the google-generativeai library for API calls.
    """
    
    DEFAULT_MODEL = "gemini-1.5-flash"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (or use GEMINI_API_KEY env var)
            model_name: Model to use (default: gemini-1.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        model = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        super().__init__(model)
        
        if not self.api_key:
            raise LLMError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter.",
                error_type="auth"
            )
        
        self._setup_client()
    
    def _setup_client(self):
        """Configure the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.genai = genai
        except ImportError:
            raise LLMError(
                "google-generativeai package not installed. Run: pip install google-generativeai",
                error_type="import"
            )
        except Exception as e:
            raise LLMError(f"Failed to configure Gemini client: {e}", error_type="config")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's tokenizer.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        try:
            result = self.client.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback to approximation
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
        Generate a response from Gemini.
        
        Args:
            prompt: Input prompt
            strategy: Prompting strategy (for tracking)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        try:
            # Configure generation
            generation_config = self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Make API call
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            response_text = response.text.strip() if response.text else ""
            
            # Get token counts
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
            retryable = any(x in str(e).lower() for x in ["rate limit", "timeout", "unavailable"])
            
            return LLMResponse(
                text="",
                prompt_tokens=self.count_tokens(prompt),
                response_tokens=0,
                model=self.model_name,
                strategy=strategy,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


def create_client(use_mock: bool = False, provider: str = "auto", **kwargs) -> BaseLLM:
    """
    Factory function to create appropriate LLM client.
    
    Args:
        use_mock: Whether to use mock client
        provider: 'gemini', 'openai', or 'auto' (detect from env)
        **kwargs: Additional arguments for client
        
    Returns:
        LLM client instance
    """
    if use_mock:
        from .mock_gemini import MockGeminiClient
        return MockGeminiClient(**kwargs)
    
    # Auto-detection
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "gemini" # Default fallback
            
    # Remove random_seed from kwargs for real clients as they don't support it
    if "random_seed" in kwargs:
        kwargs.pop("random_seed")
            
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(**kwargs)
    else:
        return GeminiClient(**kwargs)


if __name__ == "__main__":
    # Test the client
    try:
        client = GeminiClient()
        
        response = client.generate(
            prompt="What is 2+2? Answer with just the number.",
            strategy="test",
            max_tokens=10,
            temperature=0.0
        )
        
        print(f"Response: {response.text}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        
    except LLMError as e:
        print(f"Error: {e}")
        print("Using mock client for testing...")
        
        from .mock_gemini import MockGeminiClient
        client = MockGeminiClient()
        
        response = client.generate(
            prompt="Test prompt",
            strategy="zero-shot",
            true_label="entailment"
        )
        print(f"Mock response: {response.text}")
