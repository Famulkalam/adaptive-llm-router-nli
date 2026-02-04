"""
Prompt Manager

Handles prompt selection, formatting, and token tracking for NLI classification.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .templates import PromptTemplates, NLIExample


@dataclass
class PromptStats:
    """Statistics for prompt usage."""
    strategy: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_calls: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens
    
    def update(self, prompt_tokens: int, response_tokens: int):
        """Update stats with new call."""
        self.prompt_tokens += prompt_tokens
        self.response_tokens += response_tokens
        self.total_calls += 1


class PromptManager:
    """Manager for creating and tracking prompts."""
    
    STRATEGIES = ["zero-shot", "one-shot", "few-shot", "cot"]
    
    # Approximate token costs per prompt type (for estimation)
    ESTIMATED_PROMPT_TOKENS = {
        "zero-shot": 150,
        "one-shot": 220,
        "few-shot": 450,
        "cot": 350
    }
    
    def __init__(self, custom_examples: Optional[List[NLIExample]] = None):
        """
        Initialize the prompt manager.
        
        Args:
            custom_examples: Custom few-shot examples (uses defaults if None)
        """
        self.custom_examples = custom_examples
        self.stats: Dict[str, PromptStats] = {
            strategy: PromptStats(strategy=strategy)
            for strategy in self.STRATEGIES
        }
        self.call_history: List[Dict[str, Any]] = []
    
    def create_prompt(
        self,
        strategy: str,
        premise: str,
        hypothesis: str,
        **kwargs
    ) -> str:
        """
        Create a prompt for the given strategy.
        
        Args:
            strategy: Prompting strategy
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            **kwargs: Additional arguments
            
        Returns:
            Formatted prompt string
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {self.STRATEGIES}")
        
        # Use custom examples for few-shot if available
        if strategy == "few-shot" and self.custom_examples:
            kwargs["examples"] = self.custom_examples
        
        return PromptTemplates.get_prompt(strategy, premise, hypothesis, **kwargs)
    
    def record_usage(
        self,
        strategy: str,
        prompt_tokens: int,
        response_tokens: int,
        premise: str,
        hypothesis: str,
        response: str
    ):
        """
        Record token usage for a prompt call.
        
        Args:
            strategy: Prompting strategy used
            prompt_tokens: Number of tokens in the prompt
            response_tokens: Number of tokens in the response
            premise: Input premise
            hypothesis: Input hypothesis
            response: Model response
        """
        self.stats[strategy].update(prompt_tokens, response_tokens)
        
        self.call_history.append({
            "strategy": strategy,
            "premise": premise[:100] + "..." if len(premise) > 100 else premise,
            "hypothesis": hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens
        })
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of token usage across all strategies.
        
        Returns:
            Dictionary with usage statistics
        """
        summary = {
            "by_strategy": {},
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0
        }
        
        for strategy, stats in self.stats.items():
            summary["by_strategy"][strategy] = {
                "prompt_tokens": stats.prompt_tokens,
                "response_tokens": stats.response_tokens,
                "total_tokens": stats.total_tokens,
                "calls": stats.total_calls,
                "avg_prompt_tokens": stats.prompt_tokens / max(stats.total_calls, 1),
                "avg_response_tokens": stats.response_tokens / max(stats.total_calls, 1)
            }
            summary["total_prompt_tokens"] += stats.prompt_tokens
            summary["total_response_tokens"] += stats.response_tokens
            summary["total_tokens"] += stats.total_tokens
            summary["total_calls"] += stats.total_calls
        
        return summary
    
    def estimate_tokens(self, strategy: str, n_samples: int) -> int:
        """
        Estimate total tokens needed for a strategy.
        
        Args:
            strategy: Prompting strategy
            n_samples: Number of samples to process
            
        Returns:
            Estimated token count
        """
        base_tokens = self.ESTIMATED_PROMPT_TOKENS.get(strategy, 200)
        response_tokens = 50  # Approximate response length
        return n_samples * (base_tokens + response_tokens)
    
    def estimate_cost(
        self,
        strategy: str,
        n_samples: int,
        price_per_1k_input: float = 0.000125,  # Gemini 1.5 Flash pricing
        price_per_1k_output: float = 0.000375
    ) -> float:
        """
        Estimate cost for a strategy.
        
        Args:
            strategy: Prompting strategy
            n_samples: Number of samples
            price_per_1k_input: Cost per 1000 input tokens
            price_per_1k_output: Cost per 1000 output tokens
            
        Returns:
            Estimated cost in currency units
        """
        prompt_tokens = self.ESTIMATED_PROMPT_TOKENS.get(strategy, 200) * n_samples
        response_tokens = 50 * n_samples  # Approximate
        
        cost = (prompt_tokens / 1000 * price_per_1k_input + 
                response_tokens / 1000 * price_per_1k_output)
        return cost
    
    def get_strategy_comparison(self, n_samples: int = 500) -> Dict[str, Dict[str, float]]:
        """
        Get comparison of estimated costs across strategies.
        
        Args:
            n_samples: Number of samples to estimate for
            
        Returns:
            Dictionary with per-strategy estimates
        """
        comparison = {}
        
        for strategy in self.STRATEGIES:
            comparison[strategy] = {
                "estimated_tokens": self.estimate_tokens(strategy, n_samples),
                "estimated_cost_usd": self.estimate_cost(strategy, n_samples),
                "tokens_per_sample": self.ESTIMATED_PROMPT_TOKENS[strategy] + 50
            }
        
        return comparison
    
    def reset_stats(self):
        """Reset all usage statistics."""
        for strategy in self.STRATEGIES:
            self.stats[strategy] = PromptStats(strategy=strategy)
        self.call_history = []
    
    def set_custom_examples(self, examples: List[NLIExample]):
        """
        Set custom few-shot examples.
        
        Args:
            examples: List of NLI examples
        """
        self.custom_examples = examples


if __name__ == "__main__":
    # Demo the prompt manager
    manager = PromptManager()
    
    premise = "The economy grew by 3% last quarter."
    hypothesis = "Economic growth was positive."
    
    print("Prompt Manager Demo")
    print("=" * 50)
    
    for strategy in PromptManager.STRATEGIES:
        print(f"\n{strategy.upper()} prompt length:")
        prompt = manager.create_prompt(strategy, premise, hypothesis)
        print(f"  Characters: {len(prompt)}")
        print(f"  Estimated tokens: {manager.ESTIMATED_PROMPT_TOKENS[strategy]}")
    
    print("\n" + "=" * 50)
    print("Cost Comparison (500 samples)")
    print("=" * 50)
    
    comparison = manager.get_strategy_comparison(500)
    for strategy, data in comparison.items():
        print(f"\n{strategy}:")
        print(f"  Est. tokens: {data['estimated_tokens']:,}")
        print(f"  Est. cost: ${data['estimated_cost_usd']:.4f}")
