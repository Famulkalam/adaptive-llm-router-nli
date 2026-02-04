
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging

class AdaptiveGatekeeper:
    """
    Refined Adaptive Gatekeeper for NLI.
    
    Implements strict heuristic routing to optimize for accuracy vs cost-efficiency,
    addressing the 'CoT Paradox' by prioritizing Few-Shot for complex cases.
    """
    
    def __init__(
        self, 
        length_threshold: int = 220, 
        negation_threshold: int = 3, 
        overlap_threshold: float = 0.4
    ):
        self.length_threshold = length_threshold
        self.negation_threshold = negation_threshold
        self.overlap_threshold = overlap_threshold
        
        # Configure logging for MSc level technical execution
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def route_sample(self, features: Dict[str, Any], genre: str = "unknown") -> str:
        """
        Determines the optimal strategy for a single sample based on strict heuristics.
        
        Heuristic:
        - Trigger Path B (6-Shot) ONLY IF:
            - char_length > 220
            - negation_count >= 3
            - lexical_overlap < 0.4
        - Otherwise, use Path A (Zero-Shot).
        - Exception: Hybrid Path (6-Shot + CoT) if genre is Gov or OUP.
        """
        
        # Extract features
        char_len = features.get("char_length", 0)
        negations = features.get("total_negations", 0)
        overlap = features.get("lexical_overlap", 1.0)
        
        # Genre-based reasoning layer (Hybrid Path)
        if genre.lower() in ["government", "oup"]:
            self.logger.debug(f"Genre '{genre}' triggered Hybrid Layer.")
            return "6-shot-cot"
            
        # Standard Heuristic Path
        is_complex = (
            char_len > self.length_threshold and
            negations >= self.negation_threshold and
            overlap < self.overlap_threshold
        )
        
        if is_complex:
            self.logger.debug(f"Complex sample detected (Len:{char_len}, Neg:{negations}, Overlap:{overlap}). Routing to 6-Shot.")
            return "6-shot"
        else:
            return "zero-shot"

    def add_routing_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies routing logic to an entire dataframe.
        """
        self.logger.info(f"Applying adaptive routing to {len(df)} samples...")
        
        routes = []
        for idx, row in df.iterrows():
            # Support both column names
            genre = row.get("genre", "unknown")
            route = self.route_sample(row.to_dict(), genre)
            routes.append(route)
            
        df_result = df.copy()
        df_result["routed_strategy"] = routes
        
        # Log distribution
        dist = df_result["routed_strategy"].value_counts()
        self.logger.info(f"Routing distribution:\n{dist}")
        
        return df_result
