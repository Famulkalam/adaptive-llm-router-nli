"""
Final Analysis Script for Real API Results

Loads predictions from all strategies (including cached ones),
trains the adaptive gatekeeper, and generates final visualizations.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FeatureExtractor
from src.gatekeeper import DifficultyClassifier, AdaptiveRouter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def main():
    print("=" * 60)
    print("FINAL ANALYSIS: Real API Results")
    print("=" * 60)
    
    # 1. Load Predictions
    print("\nLoading predictions...")
    try:
        zero_shot_df = pd.read_csv("data/predictions/zero-shot_predictions.csv")
        one_shot_df = pd.read_csv("data/predictions/one-shot_predictions.csv")
        few_shot_df = pd.read_csv("data/predictions/few-shot_predictions.csv")
        cot_df = pd.read_csv("data/predictions/cot_predictions.csv")
        print(f"  Zero-shot: {len(zero_shot_df)} samples")
        print(f"  One-shot:  {len(one_shot_df)} samples")
        print(f"  Few-shot:  {len(few_shot_df)} samples")
        print(f"  CoT:       {len(cot_df)} samples")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Extract Features (needed for gatekeeper)
    print("\nExtracting features for test set...")
    extractor = FeatureExtractor(use_nltk=False)
    # We use zero_shot_df as the base since premises are the same across all
    df_features = extractor.extract_features_batch(zero_shot_df)

    # 3. Train Adaptive Gatekeeper
    print("\nTraining adaptive gatekeeper...")
    # Clean labels (strip whitespace, lower case)
    zero_shot_df['predicted_label'] = zero_shot_df['predicted_label'].astype(str).str.lower().str.strip()
    cot_df['predicted_label'] = cot_df['predicted_label'].astype(str).str.lower().str.strip()
    
    # Determine correctness
    zero_shot_correct = zero_shot_df['label'] == zero_shot_df['predicted_label']
    cot_correct = cot_df['label'] == cot_df['predicted_label']
    
    # Filter to common indices if sizes differ (e.g. if one crashed)
    common_indices = zero_shot_df.index.intersection(cot_df.index)
    if len(common_indices) < len(zero_shot_df):
        print(f"Warning: Aligning dataframes to {len(common_indices)} common samples")
        zero_shot_correct = zero_shot_correct.loc[common_indices]
        cot_correct = cot_correct.loc[common_indices]
        df_features = df_features.loc[common_indices]

    classifier = DifficultyClassifier(n_folds=5, random_state=42)
    train_results = classifier.fit(df_features, zero_shot_correct, cot_correct)
    
    print(f"  Gatekeeper CV Accuracy: {train_results['cv_accuracy']:.4f}")

    # 4. Route Samples
    print("\nRouting samples...")
    router = AdaptiveRouter(classifier=classifier, use_tiered_routing=True)
    df_routed = router.add_routing_to_dataframe(df_features)
    
    routing_dist = df_routed['routed_strategy'].value_counts()
    print("\nRouting Distribution:")
    print(routing_dist)

    # 5. Simulate Adaptive Inference
    print("\nSimulating adaptive inference...")
    adaptive_results = []
    
    # map strategies to dataframes
    strategy_map = {
        'zero-shot': zero_shot_df,
        'one-shot': one_shot_df,
        'few-shot': few_shot_df,
        'cot': cot_df
    }

    for idx, row in df_routed.iterrows():
        if idx not in common_indices:
            continue
            
        strategy = row['routed_strategy']
        # Map strategy name to df key
        # Strategy from router is lower case e.g. 'zero-shot'
        
        target_df = strategy_map.get(strategy)
        
        if target_df is not None and idx in target_df.index:
            pred_row = target_df.loc[idx]
            adaptive_results.append({
                'idx': idx,
                'true_label': row['label'],
                'predicted_label': str(pred_row['predicted_label']).lower().strip(),
                'strategy_used': strategy,
                'difficulty_score': row['difficulty_score']
            })
    
    adaptive_df = pd.DataFrame(adaptive_results)
    adaptive_df.to_csv('data/predictions/adaptive_predictions.csv', index=False)
    
    # 6. Calculate Metrics
    adaptive_acc = (adaptive_df['true_label'] == adaptive_df['predicted_label']).mean()
    
    accuracies = {
        'Zero-shot': (zero_shot_df.loc[common_indices, 'label'] == zero_shot_df.loc[common_indices, 'predicted_label']).mean(),
        'One-shot': (one_shot_df.loc[common_indices, 'label'] == one_shot_df.loc[common_indices, 'predicted_label']).mean(),
        'Few-shot': (few_shot_df.loc[common_indices, 'label'] == few_shot_df.loc[common_indices, 'predicted_label']).mean(),
        'CoT': (cot_df.loc[common_indices, 'label'] == cot_df.loc[common_indices, 'predicted_label']).mean(),
        'ADAPTIVE': adaptive_acc
    }
    
    print('\nTop-line Results:')
    for k, v in accuracies.items():
        print(f'  {k:10}: {v:.4f}')

    # 7. Token Savings
    # Use real token counts if available
    def get_avg_tokens(df):
        if 'prompt_tokens' in df.columns:
            return df['prompt_tokens'].mean() + df['response_tokens'].mean()
        return 0

    token_costs = {
        'zero-shot': get_avg_tokens(zero_shot_df) or 165,
        'one-shot': get_avg_tokens(one_shot_df) or 235,
        'few-shot': get_avg_tokens(few_shot_df) or 465,
        'cot': get_avg_tokens(cot_df) or 600
    }
    
    adaptive_total_tokens = 0
    for _, row in adaptive_df.iterrows():
        adaptive_total_tokens += token_costs.get(row['strategy_used'], 0)
        
    cot_total_tokens = len(adaptive_df) * token_costs['cot']
    savings_pct = (cot_total_tokens - adaptive_total_tokens) / cot_total_tokens * 100
    
    print(f'\nAdaptive Tokens: {adaptive_total_tokens:,.0f}')
    print(f'All-CoT Tokens:  {cot_total_tokens:,.0f}')
    print(f'Savings:         {savings_pct:.1f}%')

    # 8. Visualizations
    fig = plt.figure(figsize=(16, 8))
    
    # Accuracy Bar Chart
    ax1 = fig.add_subplot(1, 2, 1)
    acc_values = list(accuracies.values())
    acc_labels = list(accuracies.keys())
    colors = ['#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#27ae60']
    
    bars = ax1.bar(acc_labels, acc_values, color=colors, edgecolor='black')
    bars[-1].set_hatch('///')
    ax1.set_title(f'Accuracy Comparison (Real API - {len(adaptive_df)} samples)', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    for bar, acc in zip(bars, acc_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
                 
    # Routing Pie Chart
    ax2 = fig.add_subplot(1, 2, 2)
    # Define colors
    c_map = {'zero-shot': '#3498db', 'few-shot': '#f39c12', 'cot': '#e74c3c'}
    p_colors = [c_map.get(s, 'gray') for s in routing_dist.index]
    
    ax2.pie(routing_dist.values, labels=routing_dist.index, autopct='%1.1f%%', 
           colors=p_colors, explode=[0.05]*len(routing_dist))
    ax2.set_title('Adaptive Routing Decisions', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/real_api_dashboard.png', dpi=150)
    print('\nDashboard saved to outputs/figures/real_api_dashboard.png')

if __name__ == "__main__":
    main()
