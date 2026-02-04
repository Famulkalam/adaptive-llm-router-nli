"""
Linguistic Feature Extractor

Extracts features from premise-hypothesis pairs for the Adaptive Gatekeeper:
- Sentence length
- Negation count  
- Lexical overlap (Jaccard similarity)
"""

import re
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class FeatureExtractor:
    """Extract linguistic features from NLI sentence pairs."""
    
    # Strict negation tokens for refinement task
    NEGATION_WORDS = {"not", "never", "no", "neither", "nor"}
    
    # Simple regex for strict tokens
    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bneither\b", r"\bnor\b"
    ]
    
    def __init__(self, use_nltk: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            use_nltk: Whether to use NLTK for tokenization (more accurate but slower)
        """
        self.use_nltk = use_nltk
        
        if use_nltk:
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                print(f"Warning: NLTK setup failed, using simple tokenization: {e}")
                self.use_nltk = False
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.use_nltk:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        
        # Fallback: simple tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def calculate_sentence_length(self, premise: str, hypothesis: str) -> Dict[str, int]:
        """
        Calculate word counts for premise and hypothesis.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            Dictionary with length metrics
        """
        premise_tokens = self.tokenize(premise)
        hypothesis_tokens = self.tokenize(hypothesis)
        
        return {
            "premise_length": len(premise_tokens),
            "hypothesis_length": len(hypothesis_tokens),
            "combined_length": len(premise_tokens) + len(hypothesis_tokens),
            "length_ratio": len(hypothesis_tokens) / max(len(premise_tokens), 1),
            "char_length": len(premise) + len(hypothesis)
        }
    
    def count_negations(self, text: str) -> int:
        """
        Count negation words and patterns in text.
        
        Args:
            text: Input text
            
        Returns:
            Count of negations
        """
        text_lower = text.lower()
        tokens = set(self.tokenize(text_lower))
        
        # Count explicit negation words
        count = len(tokens.intersection(self.NEGATION_WORDS))
        
        # Count pattern matches (avoiding double counting)
        for pattern in self.NEGATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            # Only count matches not already counted
            for match in matches:
                if match.split()[0] not in self.NEGATION_WORDS:
                    count += 1
        
        return count
    
    def calculate_lexical_overlap(
        self, 
        premise: str, 
        hypothesis: str,
        remove_stopwords: bool = True
    ) -> Dict[str, float]:
        """
        Calculate lexical overlap using Jaccard similarity.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence 
            remove_stopwords: Whether to remove stopwords before comparison
            
        Returns:
            Dictionary with overlap metrics
        """
        premise_tokens = set(self.tokenize(premise))
        hypothesis_tokens = set(self.tokenize(hypothesis))
        
        if remove_stopwords and self.stop_words:
            premise_tokens = premise_tokens - self.stop_words
            hypothesis_tokens = hypothesis_tokens - self.stop_words
        
        # Jaccard similarity (lexical overlap)
        intersection = len(premise_tokens.intersection(hypothesis_tokens))
        union = len(premise_tokens.union(hypothesis_tokens))
        jaccard = intersection / max(union, 1)
        
        # Overlap ratio (proportion of hypothesis words in premise)
        hypothesis_overlap = intersection / max(len(hypothesis_tokens), 1)
        
        # Premise coverage (proportion of premise words in hypothesis)
        premise_coverage = intersection / max(len(premise_tokens), 1)
        
        return {
            "jaccard_similarity": jaccard,
            "lexical_overlap": jaccard, # Alias for consistency with prompt
            "hypothesis_overlap": hypothesis_overlap,
            "premise_coverage": premise_coverage,
            "shared_words": intersection
        }
    
    def extract_features(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Extract all features for a single premise-hypothesis pair.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        # Sentence length features
        length_features = self.calculate_sentence_length(premise, hypothesis)
        features.update(length_features)
        
        # Negation features
        features["premise_negations"] = self.count_negations(premise)
        features["hypothesis_negations"] = self.count_negations(hypothesis)
        features["total_negations"] = features["premise_negations"] + features["hypothesis_negations"]
        
        # Lexical overlap features
        overlap_features = self.calculate_lexical_overlap(premise, hypothesis)
        features.update(overlap_features)
        
        return features
    
    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all samples in a DataFrame.
        
        Args:
            df: DataFrame with 'premise' and 'hypothesis' columns
            
        Returns:
            DataFrame with original data plus feature columns
        """
        print(f"Extracting features for {len(df)} samples...")
        
        features_list = []
        for idx, row in df.iterrows():
            features = self.extract_features(row["premise"], row["hypothesis"])
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        print("Feature extraction complete!")
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            "premise_length",
            "hypothesis_length", 
            "combined_length",
            "length_ratio",
            "premise_negations",
            "hypothesis_negations",
            "total_negations",
            "jaccard_similarity",
            "hypothesis_overlap",
            "premise_coverage",
            "shared_words"
        ]
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get feature matrix for ML models.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            NumPy array of features
        """
        feature_names = self.get_feature_names()
        return df[feature_names].values


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Test cases
    test_cases = [
        {
            "premise": "The cat sat on the mat.",
            "hypothesis": "An animal is sitting on something."
        },
        {
            "premise": "The company did not report any profits this quarter.",
            "hypothesis": "The company was profitable."
        },
        {
            "premise": "Scientists have discovered a new species of deep-sea fish.",
            "hypothesis": "The weather was sunny yesterday."
        }
    ]
    
    print("Feature Extraction Test")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Premise: {case['premise']}")
        print(f"  Hypothesis: {case['hypothesis']}")
        
        features = extractor.extract_features(case["premise"], case["hypothesis"])
        print("  Features:")
        for name, value in features.items():
            if isinstance(value, float):
                print(f"    {name}: {value:.3f}")
            else:
                print(f"    {name}: {value}")
