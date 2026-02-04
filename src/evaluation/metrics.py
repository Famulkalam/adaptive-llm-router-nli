"""
Evaluation Metrics Calculator

Computes classification metrics for NLI predictions including:
- Accuracy, Macro-F1, per-class scores
- Confusion matrices
- Per-genre breakdown
- Statistical tests
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """
    Calculator for NLI classification metrics.
    
    Supports:
    - Standard classification metrics
    - Per-genre analysis
    - Strategy comparison
    - Confusion matrix analysis
    """
    
    LABELS = ["entailment", "neutral", "contradiction"]
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.results_cache = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names (optional)
            
        Returns:
            Dictionary with all metrics
        """
        if labels is None:
            labels = self.LABELS
        
        # Filter out None predictions
        mask = pd.Series(y_pred).notna()
        y_true_clean = np.array(y_true)[mask]
        y_pred_clean = np.array(y_pred)[mask]
        
        # Handle case where predictions might be missing some labels
        unique_true = set(y_true_clean)
        unique_pred = set(y_pred_clean)
        all_labels = list(unique_true | unique_pred)
        
        metrics = {
            "accuracy": accuracy_score(y_true_clean, y_pred_clean),
            "macro_f1": f1_score(y_true_clean, y_pred_clean, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true_clean, y_pred_clean, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true_clean, y_pred_clean, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true_clean, y_pred_clean, average="macro", zero_division=0),
            "n_samples": len(y_true_clean),
            "n_invalid": sum(~mask),
            "per_class": {}
        }
        
        # Per-class metrics
        for label in labels:
            if label in all_labels:
                # Binary metrics for this class
                y_true_binary = (y_true_clean == label).astype(int)
                y_pred_binary = (y_pred_clean == label).astype(int)
                
                metrics["per_class"][label] = {
                    "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                    "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                    "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
                    "support": sum(y_true_clean == label)
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)
        metrics["confusion_matrix"] = cm
        metrics["confusion_matrix_labels"] = labels
        
        return metrics
    
    def calculate_metrics_from_df(
        self,
        df: pd.DataFrame,
        true_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> Dict[str, Any]:
        """
        Calculate metrics from DataFrame columns.
        
        Args:
            df: DataFrame with predictions
            true_col: Column name for true labels
            pred_col: Column name for predicted labels
            
        Returns:
            Dictionary with metrics
        """
        return self.calculate_metrics(
            df[true_col].values,
            df[pred_col].values
        )
    
    def calculate_per_genre_metrics(
        self,
        df: pd.DataFrame,
        genre_col: str = "genre",
        true_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> pd.DataFrame:
        """
        Calculate metrics broken down by genre.
        
        Args:
            df: DataFrame with predictions
            genre_col: Column name for genre
            true_col: Column name for true labels
            pred_col: Column name for predicted labels
            
        Returns:
            DataFrame with per-genre metrics
        """
        results = []
        
        for genre in df[genre_col].unique():
            genre_df = df[df[genre_col] == genre]
            metrics = self.calculate_metrics(
                genre_df[true_col].values,
                genre_df[pred_col].values
            )
            
            results.append({
                "genre": genre,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "n_samples": metrics["n_samples"],
                "entailment_f1": metrics["per_class"].get("entailment", {}).get("f1", 0),
                "neutral_f1": metrics["per_class"].get("neutral", {}).get("f1", 0),
                "contradiction_f1": metrics["per_class"].get("contradiction", {}).get("f1", 0)
            })
        
        return pd.DataFrame(results).sort_values("accuracy", ascending=False)
    
    def compare_strategies(
        self,
        results: Dict[str, pd.DataFrame],
        true_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> pd.DataFrame:
        """
        Compare metrics across different prompting strategies.
        
        Args:
            results: Dictionary mapping strategy name to result DataFrame
            true_col: Column name for true labels
            pred_col: Column name for predicted labels
            
        Returns:
            DataFrame comparing strategies
        """
        comparison = []
        
        for strategy, df in results.items():
            metrics = self.calculate_metrics_from_df(df, true_col, pred_col)
            
            comparison.append({
                "strategy": strategy,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "n_samples": metrics["n_samples"],
                "entailment_f1": metrics["per_class"].get("entailment", {}).get("f1", 0),
                "neutral_f1": metrics["per_class"].get("neutral", {}).get("f1", 0),
                "contradiction_f1": metrics["per_class"].get("contradiction", {}).get("f1", 0)
            })
        
        return pd.DataFrame(comparison).sort_values("accuracy", ascending=False)
    
    def analyze_confusion_patterns(
        self,
        df: pd.DataFrame,
        true_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> Dict[str, Any]:
        """
        Analyze common confusion patterns.
        
        Args:
            df: DataFrame with predictions
            true_col: Column name for true labels
            pred_col: Column name for predicted labels
            
        Returns:
            Dictionary with confusion analysis
        """
        # Filter valid predictions
        valid_df = df[df[pred_col].notna()].copy()
        
        # Create confusion pairs
        valid_df["confusion_pair"] = valid_df[true_col] + " → " + valid_df[pred_col].astype(str)
        valid_df["is_correct"] = valid_df[true_col] == valid_df[pred_col]
        
        # Count errors
        errors = valid_df[~valid_df["is_correct"]]
        error_counts = errors["confusion_pair"].value_counts()
        
        # Specific confusion analysis
        neutral_entailment = len(errors[
            (errors[true_col] == "neutral") & (errors[pred_col] == "entailment")
        ])
        entailment_neutral = len(errors[
            (errors[true_col] == "entailment") & (errors[pred_col] == "neutral")
        ])
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(valid_df),
            "top_confusions": error_counts.head(6).to_dict(),
            "neutral_to_entailment": neutral_entailment,
            "entailment_to_neutral": entailment_neutral,
            "neutral_entailment_confusion_rate": (neutral_entailment + entailment_neutral) / max(len(errors), 1)
        }
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform McNemar's test to compare two classifiers.
        
        Args:
            y_true: True labels
            y_pred_a: Predictions from model A
            y_pred_b: Predictions from model B
            
        Returns:
            Dictionary with test statistics
        """
        # Build contingency table
        correct_a = y_pred_a == y_true
        correct_b = y_pred_b == y_true
        
        # Counts
        b = sum(correct_a & ~correct_b)  # A correct, B wrong
        c = sum(~correct_a & correct_b)  # A wrong, B correct
        
        # McNemar's test statistic
        if b + c == 0:
            statistic = 0
            p_value = 1.0
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            # Approximate p-value using chi-square with 1 df
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(statistic, 1)
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "a_only_correct": b,
            "b_only_correct": c,
            "significant": p_value < 0.05
        }
    
    def format_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a readable report.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted string report
        """
        lines = [
            "=" * 50,
            "Classification Metrics Report",
            "=" * 50,
            f"Samples: {metrics['n_samples']}",
            f"Invalid predictions: {metrics.get('n_invalid', 0)}",
            "",
            "Overall Metrics:",
            f"  Accuracy:     {metrics['accuracy']:.4f}",
            f"  Macro F1:     {metrics['macro_f1']:.4f}",
            f"  Weighted F1:  {metrics['weighted_f1']:.4f}",
            "",
            "Per-Class Metrics:",
        ]
        
        for label, class_metrics in metrics.get("per_class", {}).items():
            lines.append(f"  {label}:")
            lines.append(f"    Precision: {class_metrics['precision']:.4f}")
            lines.append(f"    Recall:    {class_metrics['recall']:.4f}")
            lines.append(f"    F1:        {class_metrics['f1']:.4f}")
            lines.append(f"    Support:   {class_metrics['support']}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    # Create synthetic predictions
    n_samples = 100
    labels = ["entailment", "neutral", "contradiction"]
    
    y_true = np.random.choice(labels, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, 20, replace=False)
    for idx in error_indices:
        y_pred[idx] = np.random.choice([l for l in labels if l != y_true[idx]])
    
    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_metrics(y_true, y_pred)
    
    print(calc.format_report(metrics))
    
    # Confusion analysis
    df = pd.DataFrame({"label": y_true, "predicted_label": y_pred})
    confusion = calc.analyze_confusion_patterns(df)
    
    print("\nConfusion Analysis:")
    print(f"  Total errors: {confusion['total_errors']}")
    print(f"  Error rate: {confusion['error_rate']:.2%}")
    print(f"  Top confusions: {confusion['top_confusions']}")
