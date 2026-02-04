"""
Adaptive Router for Query Routing

Routes queries to appropriate prompting strategies based on
predicted difficulty and confidence thresholds.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .classifier import DifficultyClassifier


class RoutingStrategy(Enum):
    """Available routing strategies."""
    ZERO_SHOT = "zero-shot"
    ONE_SHOT = "one-shot"
    FEW_SHOT = "few-shot"
    COT = "cot"


@dataclass
class RoutingDecision:
    """A routing decision for a query."""
    strategy: str
    difficulty_score: float
    confidence: float
    reasoning: str
    estimated_tokens: int


@dataclass
class RoutingStats:
    """Statistics for routing decisions."""
    total_queries: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    total_tokens_saved: int = 0
    total_tokens_spent: int = 0
    
    def add_decision(self, decision: RoutingDecision, baseline_tokens: int):
        """Record a routing decision."""
        self.total_queries += 1
        strategy = decision.strategy
        self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + 1
        self.total_tokens_spent += decision.estimated_tokens
        self.total_tokens_saved += max(0, baseline_tokens - decision.estimated_tokens)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of routing statistics."""
        return {
            "total_queries": self.total_queries,
            "strategy_distribution": {
                k: v / max(self.total_queries, 1) 
                for k, v in self.strategy_counts.items()
            },
            "tokens_spent": self.total_tokens_spent,
            "tokens_saved": self.total_tokens_saved,
            "savings_ratio": self.total_tokens_saved / max(self.total_tokens_spent + self.total_tokens_saved, 1)
        }


class AdaptiveRouter:
    """
    Adaptive router for NLI queries.
    
    Routes queries to different prompting strategies based on:
    - Predicted difficulty (from classifier)
    - Confidence threshold
    - Cost considerations
    """
    
    # Estimated tokens per strategy
    TOKEN_ESTIMATES = {
        "zero-shot": 165,    # 150 prompt + 15 response
        "one-shot": 235,     # 220 prompt + 15 response
        "few-shot": 465,     # 450 prompt + 15 response
        "cot": 500           # 350 prompt + 150 response
    }
    
    def __init__(
        self,
        classifier: Optional[DifficultyClassifier] = None,
        confidence_threshold: float = 0.7,
        hard_threshold: float = 0.6,
        use_tiered_routing: bool = True
    ):
        """
        Initialize the adaptive router.
        
        Args:
            classifier: Trained difficulty classifier
            confidence_threshold: Minimum confidence to use cheap strategy
            hard_threshold: Difficulty score threshold for CoT
            use_tiered_routing: Whether to use 3-tier (zero/few/cot) routing
        """
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.hard_threshold = hard_threshold
        self.use_tiered_routing = use_tiered_routing
        self.stats = RoutingStats()
    
    def route_single(
        self,
        features: pd.DataFrame,
        track_stats: bool = True
    ) -> RoutingDecision:
        """
        Route a single query.
        
        Args:
            features: DataFrame with single row of features
            track_stats: Whether to track statistics
            
        Returns:
            RoutingDecision
        """
        if self.classifier is None or not self.classifier.is_fitted:
            # Default to few-shot if no classifier
            return RoutingDecision(
                strategy="few-shot",
                difficulty_score=0.5,
                confidence=0.5,
                reasoning="No classifier available, using default strategy",
                estimated_tokens=self.TOKEN_ESTIMATES["few-shot"]
            )
        
        # Get difficulty prediction
        difficulty_score = self.classifier.get_difficulty_score(features)[0]
        probas = self.classifier.predict_proba(features)[0]
        confidence = max(probas)  # Confidence in the prediction
        
        # Determine strategy based on difficulty and confidence
        if self.use_tiered_routing:
            decision = self._tiered_routing(difficulty_score, confidence)
        else:
            decision = self._binary_routing(difficulty_score, confidence)
        
        if track_stats:
            self.stats.add_decision(decision, self.TOKEN_ESTIMATES["cot"])
        
        return decision
    
    def _binary_routing(
        self,
        difficulty_score: float,
        confidence: float
    ) -> RoutingDecision:
        """
        Simple binary routing: zero-shot or CoT.
        
        Args:
            difficulty_score: Predicted difficulty (0-1)
            confidence: Prediction confidence (0-1)
            
        Returns:
            RoutingDecision
        """
        if difficulty_score < self.hard_threshold and confidence > self.confidence_threshold:
            return RoutingDecision(
                strategy="zero-shot",
                difficulty_score=difficulty_score,
                confidence=confidence,
                reasoning=f"Low difficulty ({difficulty_score:.2f}) with high confidence ({confidence:.2f})",
                estimated_tokens=self.TOKEN_ESTIMATES["zero-shot"]
            )
        else:
            return RoutingDecision(
                strategy="cot",
                difficulty_score=difficulty_score,
                confidence=confidence,
                reasoning=f"High difficulty ({difficulty_score:.2f}) or low confidence ({confidence:.2f})",
                estimated_tokens=self.TOKEN_ESTIMATES["cot"]
            )
    
    def _tiered_routing(
        self,
        difficulty_score: float,
        confidence: float
    ) -> RoutingDecision:
        """
        Three-tier routing: zero-shot, few-shot, or CoT.
        
        Args:
            difficulty_score: Predicted difficulty (0-1)
            confidence: Prediction confidence (0-1)
            
        Returns:
            RoutingDecision
        """
        # Easy queries: zero-shot
        if difficulty_score < 0.3 and confidence > self.confidence_threshold:
            return RoutingDecision(
                strategy="zero-shot",
                difficulty_score=difficulty_score,
                confidence=confidence,
                reasoning=f"Easy query (score={difficulty_score:.2f}, conf={confidence:.2f})",
                estimated_tokens=self.TOKEN_ESTIMATES["zero-shot"]
            )
        
        # Hard queries: CoT
        elif difficulty_score > self.hard_threshold:
            return RoutingDecision(
                strategy="cot",
                difficulty_score=difficulty_score,
                confidence=confidence,
                reasoning=f"Hard query (score={difficulty_score:.2f}), using reasoning",
                estimated_tokens=self.TOKEN_ESTIMATES["cot"]
            )
        
        # Medium queries: few-shot
        else:
            return RoutingDecision(
                strategy="few-shot",
                difficulty_score=difficulty_score,
                confidence=confidence,
                reasoning=f"Medium difficulty (score={difficulty_score:.2f}), using examples",
                estimated_tokens=self.TOKEN_ESTIMATES["few-shot"]
            )
    
    def route_batch(
        self,
        df: pd.DataFrame,
        track_stats: bool = True
    ) -> List[RoutingDecision]:
        """
        Route a batch of queries.
        
        Args:
            df: DataFrame with features for multiple queries
            track_stats: Whether to track statistics
            
        Returns:
            List of RoutingDecisions
        """
        decisions = []
        
        for idx in range(len(df)):
            row_df = df.iloc[[idx]]
            decision = self.route_single(row_df, track_stats)
            decisions.append(decision)
        
        return decisions
    
    def add_routing_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add routing decisions to a DataFrame.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with added routing columns
        """
        decisions = self.route_batch(df, track_stats=False)
        
        df = df.copy()
        df["routed_strategy"] = [d.strategy for d in decisions]
        df["difficulty_score"] = [d.difficulty_score for d in decisions]
        df["routing_confidence"] = [d.confidence for d in decisions]
        df["routing_reasoning"] = [d.reasoning for d in decisions]
        df["estimated_tokens"] = [d.estimated_tokens for d in decisions]
        
        return df
    
    def estimate_savings(
        self,
        df: pd.DataFrame,
        baseline_strategy: str = "cot"
    ) -> Dict[str, Any]:
        """
        Estimate token savings from adaptive routing.
        
        Args:
            df: DataFrame with features
            baseline_strategy: Strategy to compare against
            
        Returns:
            Dictionary with savings estimates
        """
        decisions = self.route_batch(df, track_stats=False)
        
        baseline_tokens = self.TOKEN_ESTIMATES[baseline_strategy] * len(df)
        adaptive_tokens = sum(d.estimated_tokens for d in decisions)
        
        strategy_breakdown = {}
        for d in decisions:
            strategy_breakdown[d.strategy] = strategy_breakdown.get(d.strategy, 0) + 1
        
        return {
            "n_queries": len(df),
            "baseline_tokens": baseline_tokens,
            "adaptive_tokens": adaptive_tokens,
            "tokens_saved": baseline_tokens - adaptive_tokens,
            "savings_percentage": (baseline_tokens - adaptive_tokens) / baseline_tokens * 100,
            "strategy_breakdown": strategy_breakdown,
            "strategy_percentages": {k: v/len(df)*100 for k, v in strategy_breakdown.items()}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset routing statistics."""
        self.stats = RoutingStats()


if __name__ == "__main__":
    # Demo with synthetic data
    import numpy as np
    from classifier import DifficultyClassifier
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        "premise_length": np.random.randint(5, 50, n_samples),
        "hypothesis_length": np.random.randint(3, 30, n_samples),
        "combined_length": np.random.randint(10, 80, n_samples),
        "length_ratio": np.random.uniform(0.3, 2.0, n_samples),
        "premise_negations": np.random.randint(0, 3, n_samples),
        "hypothesis_negations": np.random.randint(0, 2, n_samples),
        "total_negations": np.random.randint(0, 5, n_samples),
        "jaccard_similarity": np.random.uniform(0, 0.7, n_samples),
        "hypothesis_overlap": np.random.uniform(0, 0.8, n_samples),
        "premise_coverage": np.random.uniform(0, 0.6, n_samples),
        "shared_words": np.random.randint(0, 10, n_samples),
        "genre": np.random.choice(["fiction", "government", "telephone"], n_samples)
    })
    
    # Train classifier
    difficulty = (df["combined_length"] > 40) | (df["total_negations"] > 2)
    zero_shot_correct = ~difficulty | (np.random.random(n_samples) > 0.3)
    cot_correct = ~difficulty | (np.random.random(n_samples) > 0.1)
    
    classifier = DifficultyClassifier()
    classifier.fit(df, zero_shot_correct, cot_correct)
    
    # Create router
    router = AdaptiveRouter(classifier=classifier)
    
    # Route queries
    df_routed = router.add_routing_to_dataframe(df)
    
    print("Adaptive Router Demo")
    print("=" * 50)
    
    # Show sample decisions
    print("\nSample Routing Decisions:")
    for idx in range(min(5, len(df_routed))):
        row = df_routed.iloc[idx]
        print(f"  Query {idx+1}: {row['routed_strategy']} (score={row['difficulty_score']:.2f})")
    
    # Savings estimate
    savings = router.estimate_savings(df)
    print(f"\nSavings Estimate (vs all-CoT):")
    print(f"  Baseline tokens: {savings['baseline_tokens']:,}")
    print(f"  Adaptive tokens: {savings['adaptive_tokens']:,}")
    print(f"  Tokens saved: {savings['tokens_saved']:,} ({savings['savings_percentage']:.1f}%)")
    print(f"\nStrategy breakdown:")
    for strategy, pct in savings['strategy_percentages'].items():
        print(f"  {strategy}: {pct:.1f}%")
