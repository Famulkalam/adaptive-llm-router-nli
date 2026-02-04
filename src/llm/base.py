"""
Base LLM Interface

Abstract base class for LLM clients with response dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    prompt_tokens: int
    response_tokens: int
    model: str
    strategy: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.response_tokens
    
    def get_label(self) -> Optional[str]:
        """
        Extract the NLI label from the response text.
        
        Returns:
            One of 'entailment', 'contradiction', 'neutral', or None if invalid
        """
        text = self.text.lower().strip()
        
        # Check for exact matches first
        for label in ["entailment", "contradiction", "neutral"]:
            if text == label:
                return label
        
        # Check if label appears in the response
        for label in ["entailment", "contradiction", "neutral"]:
            if label in text:
                return label
        
        # Handle common variations
        if "entail" in text:
            return "entailment"
        if "contradict" in text:
            return "contradiction"
        
        return None


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.call_count = 0
        self.total_tokens = 0
        self.errors = []
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        strategy: str = "unknown",
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            strategy: Prompting strategy used (for tracking)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional model-specific arguments
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "model": self.model_name,
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "errors": len(self.errors)
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.call_count = 0
        self.total_tokens = 0
        self.errors = []


class LLMError(Exception):
    """Custom exception for LLM errors."""
    
    def __init__(self, message: str, error_type: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
