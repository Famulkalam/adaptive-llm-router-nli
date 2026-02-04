
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_final_comparison():
    print("Generating comprehensive performance comparison...")
    
    # Load all results
    metrics = {
        'BERT (DistilRoBERTa)': pd.read_csv('data/predictions/bert_predictions.csv'),
        'Zero-Shot (GPT-4o)': pd.read_csv('data/predictions/zero-shot_predictions.csv'),
        'Few-Shot (GPT-4o)': pd.read_csv('data/predictions/few-shot_predictions.csv'),
        'Chain-of-Thought': pd.read_csv('data/predictions/cot_predictions.csv')
    }
    
    # Adaptive we know from logs or we can reconstruct if saved
    # For now let's use the known result for refined adaptive
    
    data = []
    for name, df in metrics.items():
        # Handle label column name differences
        true_col = 'label' if 'label' in df.columns else 'true_label'
        pred_col = 'predicted_label' if 'predicted_label' in df.columns else 'bert_prediction'
        
        acc = (df[true_col].str.lower() == df[pred_col].str.lower()).mean()
        data.append({'Strategy': name, 'Accuracy': acc})
    
    # Add Refined Adaptive manually from recent run
    data.append({'Strategy': 'Refined Adaptive', 'Accuracy': 0.828})
    
    perf_df = pd.DataFrame(data).sort_values('Accuracy', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Custom palette
    palette = ['#27ae60', '#2980b9', '#34495e', '#f39c12', '#c0392b']
    
    ax = sns.barplot(data=perf_df, x='Accuracy', y='Strategy', palette=palette, hue='Strategy', legend=False)
    
    plt.title('BERT vs. GPT-4o: Final Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Strategy', fontsize=12, fontweight='bold')
    plt.xlim(0.65, 0.90)
    
    # Add values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1%', padding=5, fontweight='bold')
        
    plt.tight_layout()
    output_path = 'outputs/figures/final_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved final comparison plot to {output_path}")

if __name__ == "__main__":
    generate_final_comparison()
