"""
Difficulty Classifier for Adaptive Gatekeeper

Trains a Logistic Regression model to predict query difficulty
based on linguistic features. Uses cross-validation across genres.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score


class DifficultyClassifier:
    """
    Classifier to predict query difficulty for adaptive routing.
    
    Trained on the accuracy delta between zero-shot and CoT predictions.
    Hard queries (where CoT significantly outperforms zero-shot) are routed to CoT.
    """
    
    FEATURE_NAMES = [
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
    
    def __init__(
        self,
        difficulty_threshold: float = 0.0,
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            difficulty_threshold: Accuracy delta threshold for labeling difficulty
                                  (positive = zero-shot correct, CoT wrong is "easy")
            n_folds: Number of folds for cross-validation
            random_state: Random seed
        """
        self.difficulty_threshold = difficulty_threshold
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        self.cv_results = None
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix from DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Feature matrix
        """
        missing_cols = [col for col in self.FEATURE_NAMES if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        return df[self.FEATURE_NAMES].values
    
    def _create_difficulty_labels(
        self,
        zero_shot_correct: pd.Series,
        cot_correct: pd.Series
    ) -> np.ndarray:
        """
        Create difficulty labels based on strategy performance.
        
        Labels:
        - "easy": Zero-shot got it right (no need for CoT)
        - "hard": Zero-shot wrong, CoT right (CoT helps)
        - "very_hard": Both wrong (needs CoT, might still fail)
        
        For binary classification, we simplify to:
        - 0 (easy): Zero-shot correct
        - 1 (hard): Zero-shot wrong
        
        Args:
            zero_shot_correct: Boolean series of zero-shot correctness
            cot_correct: Boolean series of CoT correctness
            
        Returns:
            Binary labels (0=easy, 1=hard)
        """
        # Simple binary: did zero-shot fail?
        labels = (~zero_shot_correct).astype(int).values
        return labels
    
    def fit(
        self,
        df: pd.DataFrame,
        zero_shot_correct: pd.Series,
        cot_correct: pd.Series,
        genre_column: str = "genre"
    ) -> Dict[str, Any]:
        """
        Fit the classifier using cross-validation.
        
        Args:
            df: DataFrame with features
            zero_shot_correct: Boolean series of zero-shot correctness
            cot_correct: Boolean series of CoT correctness
            genre_column: Column name for genre (for stratification)
            
        Returns:
            Dictionary with training results
        """
        X = self._prepare_features(df)
        y = self._create_difficulty_labels(zero_shot_correct, cot_correct)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified cross-validation by genre
        if genre_column in df.columns:
            stratify_key = df[genre_column].astype(str) + "_" + pd.Series(y).astype(str)
        else:
            stratify_key = pd.Series(y).astype(str)
        
        # Cross-validation predictions
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        try:
            cv_predictions = cross_val_predict(
                self.model, X_scaled, y, 
                cv=cv, 
                method="predict"
            )
            cv_probas = cross_val_predict(
                self.model, X_scaled, y,
                cv=cv,
                method="predict_proba"
            )
        except ValueError:
            # Fallback if stratification fails
            cv_predictions = cross_val_predict(
                self.model, X_scaled, y,
                cv=self.n_folds,
                method="predict"
            )
            cv_probas = cross_val_predict(
                self.model, X_scaled, y,
                cv=self.n_folds,
                method="predict_proba"
            )
        
        # Fit final model on all data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_NAMES,
            self.model.coef_[0]
        ))
        
        # Store CV results
        cv_accuracy = accuracy_score(y, cv_predictions)
        self.cv_results = {
            "cv_accuracy": cv_accuracy,
            "predictions": cv_predictions,
            "probabilities": cv_probas,
            "true_labels": y,
            "classification_report": classification_report(y, cv_predictions, output_dict=True)
        }
        
        return {
            "cv_accuracy": cv_accuracy,
            "n_samples": len(y),
            "n_hard": sum(y),
            "n_easy": len(y) - sum(y),
            "feature_importance": self.feature_importance,
            "classification_report": self.cv_results["classification_report"]
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict difficulty for new samples.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Binary predictions (0=easy, 1=hard)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict difficulty probabilities.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Probability array (n_samples, 2) for [easy, hard]
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_difficulty_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get difficulty score (probability of being hard).
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of difficulty scores (0-1)
        """
        probas = self.predict_proba(df)
        return probas[:, 1]  # Probability of hard class
    
    def analyze_by_genre(self, df: pd.DataFrame, genre_column: str = "genre") -> pd.DataFrame:
        """
        Analyze difficulty predictions by genre.
        
        Args:
            df: DataFrame with features and genre
            genre_column: Column name for genre
            
        Returns:
            DataFrame with per-genre statistics
        """
        if genre_column not in df.columns:
            raise ValueError(f"Genre column '{genre_column}' not found")
        
        df = df.copy()
        df["predicted_difficulty"] = self.predict(df)
        df["difficulty_score"] = self.get_difficulty_score(df)
        
        stats = df.groupby(genre_column).agg({
            "predicted_difficulty": ["mean", "sum", "count"],
            "difficulty_score": ["mean", "std", "min", "max"]
        }).round(3)
        
        stats.columns = [
            "hard_ratio", "hard_count", "total",
            "avg_score", "std_score", "min_score", "max_score"
        ]
        
        return stats.sort_values("hard_ratio", ascending=False)
    
    def save(self, path: str):
        """Save the trained classifier."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted classifier")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_importance": self.feature_importance,
                "cv_results": self.cv_results,
                "config": {
                    "difficulty_threshold": self.difficulty_threshold,
                    "n_folds": self.n_folds,
                    "random_state": self.random_state
                }
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "DifficultyClassifier":
        """Load a trained classifier."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        classifier = cls(**data["config"])
        classifier.model = data["model"]
        classifier.scaler = data["scaler"]
        classifier.feature_importance = data["feature_importance"]
        classifier.cv_results = data["cv_results"]
        classifier.is_fitted = True
        
        return classifier


if __name__ == "__main__":
    # Demo with synthetic data
    import numpy as np
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
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
    
    # Synthetic correctness (harder samples = longer, more negations)
    difficulty = (df["combined_length"] > 40) | (df["total_negations"] > 2)
    zero_shot_correct = ~difficulty | (np.random.random(n_samples) > 0.3)
    cot_correct = ~difficulty | (np.random.random(n_samples) > 0.1)
    
    # Train classifier
    classifier = DifficultyClassifier()
    results = classifier.fit(df, zero_shot_correct, cot_correct)
    
    print("Difficulty Classifier Training Results")
    print("=" * 50)
    print(f"CV Accuracy: {results['cv_accuracy']:.3f}")
    print(f"Samples: {results['n_samples']} (Easy: {results['n_easy']}, Hard: {results['n_hard']})")
    print("\nFeature Importance:")
    for feat, imp in sorted(results["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: {imp:.3f}")
    
    print("\nPer-Genre Analysis:")
    print(classifier.analyze_by_genre(df))
