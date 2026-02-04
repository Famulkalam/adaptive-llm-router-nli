"""
Data Preprocessor

Handles cleaning, filtering, and stratified sampling of the MultiNLI dataset.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocessor for cleaning and sampling MultiNLI data."""
    
    VALID_LABELS = ["entailment", "neutral", "contradiction"]
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing ambiguous labels and invalid entries.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)
        
        # Remove rows with unknown/ambiguous labels
        df_clean = df[df["label"].isin(self.VALID_LABELS)].copy()
        
        # Remove rows with empty premise or hypothesis
        df_clean = df_clean[
            (df_clean["premise"].str.strip().str.len() > 0) &
            (df_clean["hypothesis"].str.strip().str.len() > 0)
        ]
        
        # Remove duplicates based on premise-hypothesis pair
        df_clean = df_clean.drop_duplicates(subset=["premise", "hypothesis"])
        
        removed = original_len - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} invalid/ambiguous samples ({removed/original_len*100:.1f}%)")
        
        return df_clean.reset_index(drop=True)
    
    def stratified_sample(
        self,
        df: pd.DataFrame,
        samples_per_genre: int = 50,
        random_state: int = 42,
        balance_labels: bool = True
    ) -> pd.DataFrame:
        """
        Create a stratified sample with equal representation across genres.
        
        Args:
            df: Cleaned DataFrame
            samples_per_genre: Number of samples per genre
            random_state: Random seed for reproducibility
            balance_labels: Whether to balance labels within each genre
            
        Returns:
            Stratified sample DataFrame
        """
        np.random.seed(random_state)
        samples = []
        
        for genre in df["genre"].unique():
            genre_df = df[df["genre"] == genre]
            
            if balance_labels:
                # Try to get equal samples per label
                samples_per_label = samples_per_genre // 3
                remainder = samples_per_genre % 3
                
                for i, label in enumerate(self.VALID_LABELS):
                    label_df = genre_df[genre_df["label"] == label]
                    n_samples = samples_per_label + (1 if i < remainder else 0)
                    
                    if len(label_df) >= n_samples:
                        sampled = label_df.sample(n=n_samples, random_state=random_state)
                    else:
                        # If not enough samples, take all available
                        sampled = label_df
                        print(f"Warning: Only {len(label_df)} samples for {genre}/{label}")
                    
                    samples.append(sampled)
            else:
                # Simple random sampling
                if len(genre_df) >= samples_per_genre:
                    sampled = genre_df.sample(n=samples_per_genre, random_state=random_state)
                else:
                    sampled = genre_df
                    print(f"Warning: Only {len(genre_df)} samples for {genre}")
                
                samples.append(sampled)
        
        result = pd.concat(samples, ignore_index=True)
        
        # Shuffle the final dataset
        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        return result
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split stratified by genre and label.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Create stratification key combining genre and label
        df["strat_key"] = df["genre"] + "_" + df["label"]
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["strat_key"],
            random_state=random_state
        )
        
        # Remove temporary column
        train_df = train_df.drop(columns=["strat_key"]).reset_index(drop=True)
        test_df = test_df.drop(columns=["strat_key"]).reset_index(drop=True)
        
        return train_df, test_df
    
    def process_and_save(
        self,
        df: pd.DataFrame,
        samples_per_genre: int = 50,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Full preprocessing pipeline: clean, sample, split, and save.
        
        Args:
            df: Raw DataFrame
            samples_per_genre: Samples per genre for stratified sample
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with 'full', 'train', and 'test' DataFrames
        """
        # Step 1: Clean
        print("Step 1: Cleaning data...")
        df_clean = self.clean_data(df)
        
        # Step 2: Stratified sample
        print(f"Step 2: Stratified sampling ({samples_per_genre} per genre)...")
        df_sampled = self.stratified_sample(
            df_clean, 
            samples_per_genre=samples_per_genre,
            random_state=random_state
        )
        
        # Step 3: Train/test split
        print(f"Step 3: Creating train/test split ({1-test_size:.0%}/{test_size:.0%})...")
        train_df, test_df = self.create_train_test_split(
            df_sampled,
            test_size=test_size,
            random_state=random_state
        )
        
        # Step 4: Save
        print("Step 4: Saving processed data...")
        df_sampled.to_parquet(self.output_dir / "full_sample.parquet")
        train_df.to_parquet(self.output_dir / "train.parquet")
        test_df.to_parquet(self.output_dir / "test.parquet")
        
        # Also save as CSV for easy viewing
        df_sampled.to_csv(self.output_dir / "full_sample.csv", index=False)
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)
        
        print(f"\nProcessing complete!")
        print(f"  Full sample: {len(df_sampled)} samples")
        print(f"  Train set: {len(train_df)} samples")
        print(f"  Test set: {len(test_df)} samples")
        print(f"  Files saved to: {self.output_dir}")
        
        return {
            "full": df_sampled,
            "train": train_df,
            "test": test_df
        }
    
    def get_sample_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed statistics about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(df),
            "genres": {
                genre: {
                    "total": len(genre_df),
                    "labels": genre_df["label"].value_counts().to_dict()
                }
                for genre, genre_df in df.groupby("genre")
            },
            "label_distribution": df["label"].value_counts().to_dict(),
            "avg_premise_length": df["premise"].str.split().str.len().mean(),
            "avg_hypothesis_length": df["hypothesis"].str.split().str.len().mean()
        }
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MultiNLI data")
    parser.add_argument("--validate", action="store_true", help="Validate existing data")
    args = parser.parse_args()
    
    if args.validate:
        # Validate existing processed data
        preprocessor = DataPreprocessor()
        for split in ["full_sample", "train", "test"]:
            path = preprocessor.output_dir / f"{split}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n{split}:")
                print(f"  Samples: {len(df)}")
                print(f"  Genres: {df['genre'].nunique()}")
                print(f"  Labels: {df['label'].value_counts().to_dict()}")
    else:
        # Full preprocessing
        from loader import MultiNLILoader
        
        loader = MultiNLILoader()
        df_raw = loader.load_dataset("validation_matched")
        
        preprocessor = DataPreprocessor()
        datasets = preprocessor.process_and_save(df_raw, samples_per_genre=50)
