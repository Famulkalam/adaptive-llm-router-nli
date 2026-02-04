"""
Cost Analysis Module

Analyzes token usage and costs for different prompting strategies.
Calculates ROI of adaptive gatekeeper vs. baseline approaches.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PricingConfig:
    """Pricing configuration for LLM APIs."""
    name: str
    input_price_per_1k: float  # USD per 1000 input tokens
    output_price_per_1k: float  # USD per 1000 output tokens
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


# Predefined pricing configs (as of 2024)
PRICING_CONFIGS = {
    "gemini-1.5-flash": PricingConfig(
        name="Gemini 1.5 Flash",
        input_price_per_1k=0.000075,  # $0.075 per 1M = $0.000075 per 1K
        output_price_per_1k=0.0003    # $0.30 per 1M = $0.0003 per 1K
    ),
    "gemini-1.5-pro": PricingConfig(
        name="Gemini 1.5 Pro",
        input_price_per_1k=0.00125,   # $1.25 per 1M
        output_price_per_1k=0.005     # $5.00 per 1M
    ),
    "gpt-4o": PricingConfig(
        name="GPT-4o",
        input_price_per_1k=0.005,     # $5 per 1M
        output_price_per_1k=0.015     # $15 per 1M
    ),
    "gpt-4o-mini": PricingConfig(
        name="GPT-4o Mini",
        input_price_per_1k=0.00015,   # $0.15 per 1M
        output_price_per_1k=0.0006    # $0.60 per 1M
    )
}


class CostAnalyzer:
    """
    Analyzer for LLM inference costs.
    
    Calculates:
    - Per-strategy costs
    - Adaptive routing savings
    - Cost-accuracy tradeoffs
    - Budget projections
    """
    
    # Default token estimates per strategy
    DEFAULT_TOKEN_ESTIMATES = {
        "zero-shot": {"input": 150, "output": 15},
        "one-shot": {"input": 220, "output": 15},
        "few-shot": {"input": 450, "output": 15},
        "cot": {"input": 350, "output": 150}
    }
    
    def __init__(
        self,
        pricing: Optional[PricingConfig] = None,
        token_estimates: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """
        Initialize the cost analyzer.
        
        Args:
            pricing: Pricing configuration to use
            token_estimates: Custom token estimates per strategy
        """
        self.pricing = pricing or PRICING_CONFIGS["gemini-1.5-flash"]
        self.token_estimates = token_estimates or self.DEFAULT_TOKEN_ESTIMATES
    
    def calculate_strategy_cost(
        self,
        strategy: str,
        n_samples: int,
        custom_tokens: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Calculate cost for a strategy.
        
        Args:
            strategy: Prompting strategy
            n_samples: Number of samples
            custom_tokens: Custom token counts (overrides estimates)
            
        Returns:
            Dictionary with cost breakdown
        """
        tokens = custom_tokens or self.token_estimates.get(strategy, {"input": 200, "output": 50})
        
        total_input = tokens["input"] * n_samples
        total_output = tokens["output"] * n_samples
        total_cost = self.pricing.calculate_cost(total_input, total_output)
        
        return {
            "strategy": strategy,
            "n_samples": n_samples,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "cost_usd": total_cost,
            "cost_per_sample": total_cost / n_samples if n_samples > 0 else 0
        }
    
    def compare_strategies(
        self,
        n_samples: int = 500
    ) -> pd.DataFrame:
        """
        Compare costs across all strategies.
        
        Args:
            n_samples: Number of samples to estimate for
            
        Returns:
            DataFrame with cost comparison
        """
        results = []
        
        for strategy in self.token_estimates.keys():
            cost_data = self.calculate_strategy_cost(strategy, n_samples)
            results.append(cost_data)
        
        df = pd.DataFrame(results)
        df["relative_cost"] = df["cost_usd"] / df["cost_usd"].min()
        
        return df.sort_values("cost_usd")
    
    def calculate_adaptive_savings(
        self,
        strategy_distribution: Dict[str, int],
        baseline_strategy: str = "cot"
    ) -> Dict[str, Any]:
        """
        Calculate savings from adaptive routing.
        
        Args:
            strategy_distribution: Count of samples per strategy
            baseline_strategy: Strategy to compare against
            
        Returns:
            Dictionary with savings analysis
        """
        total_samples = sum(strategy_distribution.values())
        
        # Calculate adaptive cost
        adaptive_cost = 0
        adaptive_tokens = 0
        
        for strategy, count in strategy_distribution.items():
            result = self.calculate_strategy_cost(strategy, count)
            adaptive_cost += result["cost_usd"]
            adaptive_tokens += result["total_tokens"]
        
        # Calculate baseline cost
        baseline_result = self.calculate_strategy_cost(baseline_strategy, total_samples)
        baseline_cost = baseline_result["cost_usd"]
        baseline_tokens = baseline_result["total_tokens"]
        
        savings = baseline_cost - adaptive_cost
        savings_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        return {
            "total_samples": total_samples,
            "strategy_distribution": strategy_distribution,
            "adaptive_cost_usd": adaptive_cost,
            "adaptive_tokens": adaptive_tokens,
            "baseline_cost_usd": baseline_cost,
            "baseline_tokens": baseline_tokens,
            "cost_savings_usd": savings,
            "cost_savings_pct": savings_pct,
            "token_savings": baseline_tokens - adaptive_tokens,
            "token_savings_pct": (baseline_tokens - adaptive_tokens) / baseline_tokens * 100 if baseline_tokens > 0 else 0
        }
    
    def calculate_cost_accuracy_frontier(
        self,
        strategy_results: Dict[str, Dict[str, float]],
        n_samples: int = 500
    ) -> pd.DataFrame:
        """
        Calculate cost-accuracy frontier for strategy comparison.
        
        Args:
            strategy_results: Dict with accuracy for each strategy
                             Format: {"strategy": {"accuracy": 0.85, ...}}
            n_samples: Number of samples
            
        Returns:
            DataFrame for frontier plotting
        """
        frontier_data = []
        
        for strategy, metrics in strategy_results.items():
            cost_data = self.calculate_strategy_cost(strategy, n_samples)
            
            frontier_data.append({
                "strategy": strategy,
                "accuracy": metrics.get("accuracy", 0),
                "macro_f1": metrics.get("macro_f1", 0),
                "cost_usd": cost_data["cost_usd"],
                "total_tokens": cost_data["total_tokens"],
                "cost_per_sample": cost_data["cost_per_sample"]
            })
        
        df = pd.DataFrame(frontier_data)
        
        # Calculate efficiency (accuracy per dollar)
        df["efficiency"] = df["accuracy"] / df["cost_usd"]
        
        # Identify Pareto-optimal strategies
        df["pareto_optimal"] = False
        for idx, row in df.iterrows():
            is_dominated = False
            for _, other in df.iterrows():
                if (other["accuracy"] > row["accuracy"] and 
                    other["cost_usd"] <= row["cost_usd"]):
                    is_dominated = True
                    break
                if (other["cost_usd"] < row["cost_usd"] and 
                    other["accuracy"] >= row["accuracy"]):
                    is_dominated = True
                    break
            df.loc[idx, "pareto_optimal"] = not is_dominated
        
        return df.sort_values("cost_usd")
    
    def project_budget_usage(
        self,
        budget_usd: float,
        strategy_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Project how many samples can be processed with a budget.
        
        Args:
            budget_usd: Available budget in USD
            strategy_distribution: Optional distribution of strategies
                                   (as percentages, should sum to 1)
            
        Returns:
            Dictionary with budget projections
        """
        if strategy_distribution is None:
            # Assume equal distribution
            n_strategies = len(self.token_estimates)
            strategy_distribution = {s: 1/n_strategies for s in self.token_estimates}
        
        # Calculate weighted average cost per sample
        weighted_cost = 0
        for strategy, weight in strategy_distribution.items():
            cost_data = self.calculate_strategy_cost(strategy, 1)
            weighted_cost += cost_data["cost_per_sample"] * weight
        
        max_samples = int(budget_usd / weighted_cost) if weighted_cost > 0 else 0
        
        return {
            "budget_usd": budget_usd,
            "avg_cost_per_sample": weighted_cost,
            "max_samples": max_samples,
            "strategy_distribution": strategy_distribution,
            "per_strategy_samples": {
                s: int(max_samples * w) 
                for s, w in strategy_distribution.items()
            }
        }
    
    def format_cost_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Format cost analysis results as a report.
        
        Args:
            results: Results from calculate_adaptive_savings or similar
            
        Returns:
            Formatted string report
        """
        lines = [
            "=" * 50,
            "Cost Analysis Report",
            "=" * 50,
            f"Model: {self.pricing.name}",
            f"Input: ${self.pricing.input_price_per_1k:.6f}/1K tokens",
            f"Output: ${self.pricing.output_price_per_1k:.6f}/1K tokens",
            "",
        ]
        
        if "adaptive_cost_usd" in results:
            lines.extend([
                "Adaptive Routing Analysis:",
                f"  Total samples: {results['total_samples']:,}",
                "",
                "  Strategy Distribution:",
            ])
            
            for strategy, count in results.get("strategy_distribution", {}).items():
                pct = count / results["total_samples"] * 100
                lines.append(f"    {strategy}: {count:,} ({pct:.1f}%)")
            
            lines.extend([
                "",
                f"  Adaptive cost: ${results['adaptive_cost_usd']:.4f}",
                f"  Baseline cost: ${results['baseline_cost_usd']:.4f}",
                f"  Savings: ${results['cost_savings_usd']:.4f} ({results['cost_savings_pct']:.1f}%)",
                f"  Token savings: {results['token_savings']:,} ({results['token_savings_pct']:.1f}%)",
            ])
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo cost analysis
    analyzer = CostAnalyzer()
    
    print("Cost Analyzer Demo")
    print("=" * 50)
    
    # Compare strategies for 500 samples
    print("\nStrategy Cost Comparison (500 samples):")
    comparison = analyzer.compare_strategies(500)
    print(comparison.to_string(index=False))
    
    # Calculate adaptive savings
    print("\n" + "=" * 50)
    print("Adaptive Routing Savings:")
    
    # Simulate adaptive distribution
    distribution = {
        "zero-shot": 200,   # 40% easy queries
        "few-shot": 200,    # 40% medium queries
        "cot": 100          # 20% hard queries
    }
    
    savings = analyzer.calculate_adaptive_savings(distribution, baseline_strategy="cot")
    print(analyzer.format_cost_report(savings))
    
    # Budget projection
    print("\nBudget Projection (£20 ≈ $25):")
    projection = analyzer.project_budget_usage(25.0)
    print(f"  Max samples with equal distribution: {projection['max_samples']:,}")
