#!/usr/bin/env python3
"""
Main Inference Script for Adaptive NLI Classification

Runs the complete NLI classification pipeline including:
1. Data loading and preprocessing
2. Feature extraction
3. Multi-strategy inference (zero-shot, one-shot, few-shot, CoT)
4. Adaptive gatekeeper training and evaluation
5. Results export and visualization
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import MultiNLILoader, DataPreprocessor, FeatureExtractor
from src.prompts import PromptTemplates, PromptManager
from src.llm import MockGeminiClient, BatchProcessor
from src.llm.gemini import create_client
from src.gatekeeper import DifficultyClassifier, AdaptiveRouter
from src.evaluation import MetricsCalculator, CostAnalyzer, Visualizer


def load_and_preprocess_data(
    samples_per_genre: int = 50,
    use_cache: bool = True,
    random_state: int = 42
) -> dict:
    """Load and preprocess the MultiNLI dataset."""
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)
    
    # Load data
    loader = MultiNLILoader()
    df_raw = loader.load_dataset("validation_matched", use_cache=use_cache)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    datasets = preprocessor.process_and_save(
        df_raw,
        samples_per_genre=samples_per_genre,
        test_size=0.3,
        random_state=random_state
    )
    
    # Extract features
    print("\nExtracting linguistic features...")
    extractor = FeatureExtractor()
    
    for key in datasets:
        datasets[key] = extractor.extract_features_batch(datasets[key])
        datasets[key].to_csv(f"data/processed/{key}_with_features.csv", index=False)
    
    print(f"\nData ready:")
    print(f"  Full sample: {len(datasets['full'])} samples")
    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Test: {len(datasets['test'])} samples")
    
    return datasets


def run_strategy_inference(
    df: pd.DataFrame,
    llm_client,
    strategies: list = None,
    show_progress: bool = True
) -> dict:
    """Run inference with multiple strategies."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Multi-Strategy Inference")
    print("=" * 60)
    
    if strategies is None:
        strategies = ["zero-shot", "one-shot", "few-shot", "cot"]
    
    processor = BatchProcessor(llm=llm_client, batch_size=10)
    prompt_manager = PromptManager()
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'─' * 40}")
        print(f"Running {strategy.upper()} strategy...")
        print(f"{'─' * 40}")
        
        # Create prompt function for this strategy
        def make_prompt_fn(strat):
            def prompt_fn(item):
                return prompt_manager.create_prompt(
                    strat,
                    item["premise"],
                    item["hypothesis"]
                )
            return prompt_fn
        
        # Set max_tokens based on strategy
        max_tokens = 600 if "cot" in strategy else 20
        
        result_df = processor.process_dataframe(
            df=df.copy(),
            prompt_fn=make_prompt_fn(strategy),
            strategy=strategy,
            show_progress=show_progress,
            max_tokens=max_tokens
        )
        
        results[strategy] = result_df
        
        # Quick accuracy check
        valid = result_df["predicted_label"].notna()
        if valid.any():
            accuracy = (result_df.loc[valid, "label"] == result_df.loc[valid, "predicted_label"]).mean()
            print(f"  Accuracy: {accuracy:.4f}")
    
    return results


def evaluate_strategies(
    results: dict,
    output_dir: str = "data/predictions"
) -> dict:
    """Evaluate all strategies and generate metrics."""
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating Strategies")
    print("=" * 60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics_calc = MetricsCalculator()
    cost_analyzer = CostAnalyzer()
    
    all_metrics = {}
    
    for strategy, df in results.items():
        # Save predictions
        df.to_csv(f"{output_dir}/{strategy}_predictions.csv", index=False)
        df.to_json(f"{output_dir}/{strategy}_predictions.json", orient="records", indent=2)
        
        # Calculate metrics
        metrics = metrics_calc.calculate_metrics_from_df(df)
        all_metrics[strategy] = metrics
        
        print(f"\n{strategy.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        
        # Confusion analysis
        confusion = metrics_calc.analyze_confusion_patterns(df)
        print(f"  Neutral↔Entailment confusion: {confusion['neutral_entailment_confusion_rate']:.2%}")
    
    # Strategy comparison
    comparison_df = metrics_calc.compare_strategies(results)
    comparison_df.to_csv(f"{output_dir}/strategy_comparison.csv", index=False)
    
    print("\n" + "─" * 40)
    print("Strategy Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Cost analysis
    print("\n" + "─" * 40)
    print("Cost Analysis (per 500 samples):")
    cost_comparison = cost_analyzer.compare_strategies(500)
    print(cost_comparison.to_string(index=False))
    
    return all_metrics


def train_adaptive_gatekeeper(
    results: dict,
    test_df: pd.DataFrame
) -> tuple:
    """Train and evaluate the adaptive gatekeeper."""
    print("\n" + "=" * 60)
    print("STEP 4: Training Adaptive Gatekeeper")
    print("=" * 60)
    
    # Get correctness from zero-shot and CoT
    zero_shot_df = results["zero-shot"]
    cot_df = results["cot"]
    
    zero_shot_correct = zero_shot_df["label"] == zero_shot_df["predicted_label"]
    cot_correct = cot_df["label"] == cot_df["predicted_label"]
    
    # Train classifier
    classifier = DifficultyClassifier(n_folds=5)
    train_results = classifier.fit(
        zero_shot_df,
        zero_shot_correct,
        cot_correct
    )
    
    print(f"\nGatekeeper Training Results:")
    print(f"  CV Accuracy: {train_results['cv_accuracy']:.4f}")
    print(f"  Easy samples: {train_results['n_easy']}")
    print(f"  Hard samples: {train_results['n_hard']}")
    
    print("\nTop Feature Importances:")
    sorted_features = sorted(
        train_results["feature_importance"].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    for feat, imp in sorted_features:
        print(f"  {feat}: {imp:.3f}")
    
    # Create router
    router = AdaptiveRouter(classifier=classifier)
    
    # Analyze per-genre difficulty
    print("\nPer-Genre Difficulty Analysis:")
    genre_analysis = classifier.analyze_by_genre(zero_shot_df)
    print(genre_analysis.to_string())
    
    # Estimate savings
    savings = router.estimate_savings(test_df)
    print(f"\nEstimated Token Savings (vs all-CoT):")
    print(f"  Adaptive tokens: {savings['adaptive_tokens']:,}")
    print(f"  Baseline tokens: {savings['baseline_tokens']:,}")
    print(f"  Savings: {savings['savings_percentage']:.1f}%")
    
    # Save gatekeeper
    classifier.save("data/processed/gatekeeper_model.pkl")
    
    return classifier, router


def generate_visualizations(
    metrics: dict,
    results: dict,
    output_dir: str = "outputs/figures"
) -> None:
    """Generate all visualizations."""
    print("\n" + "=" * 60)
    print("STEP 5: Generating Visualizations")
    print("=" * 60)
    
    visualizer = Visualizer(output_dir=output_dir)
    metrics_calc = MetricsCalculator()
    cost_analyzer = CostAnalyzer()
    
    # Strategy comparison
    comparison_df = metrics_calc.compare_strategies(results)
    visualizer.plot_strategy_comparison(
        comparison_df,
        metric="accuracy",
        title="Strategy Accuracy Comparison",
        save_name="strategy_comparison.png"
    )
    print("  ✓ Strategy comparison chart")
    
    # Confusion matrices
    for strategy, strategy_metrics in metrics.items():
        if "confusion_matrix" in strategy_metrics:
            visualizer.plot_confusion_matrix(
                strategy_metrics["confusion_matrix"],
                strategy_metrics["confusion_matrix_labels"],
                title=f"Confusion Matrix ({strategy})",
                save_name=f"confusion_matrix_{strategy}.png"
            )
    print("  ✓ Confusion matrices")
    
    # Cost-accuracy frontier
    strategy_metrics_dict = {s: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]} 
                            for s, m in metrics.items()}
    frontier_df = cost_analyzer.calculate_cost_accuracy_frontier(strategy_metrics_dict)
    visualizer.plot_cost_accuracy_frontier(
        frontier_df,
        title="Cost-Accuracy Frontier",
        save_name="cost_accuracy_frontier.png"
    )
    print("  ✓ Cost-accuracy frontier")
    
    # Prompt progression
    accuracy_by_strategy = {s: m["accuracy"] for s, m in metrics.items()}
    visualizer.plot_prompt_progression(
        accuracy_by_strategy,
        title="Accuracy Progression: Zero-shot → CoT",
        save_name="prompt_progression.png"
    )
    print("  ✓ Prompt progression chart")
    
    # Summary dashboard
    confusion_matrices = {s: m.get("confusion_matrix") for s, m in metrics.items()
                         if "confusion_matrix" in m}
    visualizer.create_summary_dashboard(
        comparison_df,
        confusion_matrices,
        frontier_df,
        save_name="summary_dashboard.png"
    )
    print("  ✓ Summary dashboard")
    
    print(f"\nVisualizations saved to: {output_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Adaptive NLI Classification Pipeline")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM client")
    parser.add_argument("--samples-per-genre", type=int, default=50, 
                       help="Samples per genre (default: 50)")
    parser.add_argument("--strategies", nargs="+", 
                       default=["zero-shot", "one-shot", "few-shot", "cot"],
                       help="Strategies to run")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference, use cached predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Adaptive NLI Classification System                   ║")
    print("║     Multi-Genre Natural Language Inference               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Mock' if args.mock else 'Production'} LLM")
    print(f"Samples per genre: {args.samples_per_genre}")
    print(f"Strategies: {', '.join(args.strategies)}")
    
    # Step 1: Load and preprocess data
    datasets = load_and_preprocess_data(
        samples_per_genre=args.samples_per_genre,
        random_state=args.seed
    )
    
    # Step 2: Run inference
    if not args.skip_inference:
        # Create LLM client
        llm_client = create_client(use_mock=args.mock, random_seed=args.seed)
        
        # Run inference on test set
        results = run_strategy_inference(
            datasets["test"],
            llm_client,
            strategies=args.strategies
        )
    else:
        # Load cached predictions
        print("\nLoading cached predictions...")
        results = {}
        for strategy in args.strategies:
            path = f"data/predictions/{strategy}_predictions.csv"
            if Path(path).exists():
                results[strategy] = pd.read_csv(path)
                print(f"  Loaded {strategy}: {len(results[strategy])} samples")
    
    # Step 3: Evaluate strategies
    metrics = evaluate_strategies(results)
    
    # Step 4: Train adaptive gatekeeper
    classifier, router = train_adaptive_gatekeeper(results, datasets["test"])
    
    # Step 5: Generate visualizations
    generate_visualizations(metrics, results)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutputs:")
    print("  - Predictions: data/predictions/")
    print("  - Figures: outputs/figures/")
    print("  - Gatekeeper model: data/processed/gatekeeper_model.pkl")


if __name__ == "__main__":
    main()
