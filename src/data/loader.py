"""
MultiNLI Dataset Loader

Downloads and loads the MultiNLI dataset from HuggingFace.
Supports both matched and mismatched validation sets.
"""

import os
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class MultiNLILoader:
    """Loader for the MultiNLI dataset from HuggingFace."""
    
    # Genre mapping from MultiNLI
    GENRES = [
        "fiction",
        "government", 
        "slate",
        "telephone",
        "travel",
        "facetoface",
        "letters",
        "nineeleven",
        "oup",
        "verbatim"
    ]
    
    LABEL_MAP = {
        0: "entailment",
        1: "neutral", 
        2: "contradiction"
    }
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory to cache downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(
        self,
        split: Literal["train", "validation_matched", "validation_mismatched"] = "validation_matched",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load the MultiNLI dataset.
        
        Args:
            split: Dataset split to load
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with premise, hypothesis, label, and genre columns
        """
        cache_path = self.data_dir / f"multinli_{split}.parquet"
        
        # Check cache
        if use_cache and cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Load from HuggingFace
        print(f"Downloading MultiNLI {split} split from HuggingFace...")
        dataset = load_dataset("multi_nli", split=split)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "premise": dataset["premise"],
            "hypothesis": dataset["hypothesis"],
            "label": [self.LABEL_MAP.get(l, "unknown") for l in dataset["label"]],
            "genre": dataset["genre"],
            "pairID": dataset["pairID"] if "pairID" in dataset.features else range(len(dataset))
        })
        
        # Cache the data
        df.to_parquet(cache_path)
        print(f"Cached data to {cache_path}")
        
        return df
    
    def load_validation_combined(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load both matched and mismatched validation sets combined.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Combined DataFrame from both validation splits
        """
        matched = self.load_dataset("validation_matched", use_cache)
        matched["source"] = "matched"
        
        mismatched = self.load_dataset("validation_mismatched", use_cache)
        mismatched["source"] = "mismatched"
        
        combined = pd.concat([matched, mismatched], ignore_index=True)
        return combined
    
    def get_genre_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of samples across genres and labels.
        
        Args:
            df: DataFrame with genre and label columns
            
        Returns:
            Cross-tabulation of genres and labels
        """
        return pd.crosstab(df["genre"], df["label"], margins=True)
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate the loaded dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_samples": len(df),
            "unique_genres": df["genre"].nunique(),
            "genres": df["genre"].unique().tolist(),
            "label_distribution": df["label"].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unknown_labels": (df["label"] == "unknown").sum()
        }
        return results


if __name__ == "__main__":
    # Test the loader
    loader = MultiNLILoader()
    df = loader.load_dataset("validation_matched")
    print(f"\nLoaded {len(df)} samples")
    print(f"\nGenre distribution:")
    print(loader.get_genre_distribution(df))
    print(f"\nValidation results:")
    print(loader.validate_data(df))
