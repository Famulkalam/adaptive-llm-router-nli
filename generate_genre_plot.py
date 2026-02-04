
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_genre_plot():
    print("Generating genre performance plot...")
    
    # Load data
    try:
        zero_shot_df = pd.read_csv('data/predictions/zero-shot_predictions.csv')
        cot_df = pd.read_csv('data/predictions/cot_predictions.csv')
        few_shot_df = pd.read_csv('data/predictions/few-shot_predictions.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare data for plotting
    strategies = {
        'Zero-shot': zero_shot_df, 
        'Few-shot': few_shot_df,
        'CoT': cot_df
    }

    data = []
    for strat_name, df in strategies.items():
        genre_acc = df.groupby('genre').apply(
            lambda x: (x['label'] == x['predicted_label']).mean()
        ).reset_index(name='accuracy')
        genre_acc['strategy'] = strat_name
        data.append(genre_acc)

    plot_df = pd.concat(data)

    # Plot
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Create barplot
    ax = sns.barplot(
        data=plot_df, 
        x='genre', 
        y='accuracy', 
        hue='strategy', 
        palette=['#3498db', '#f1c40f', '#e74c3c'], # Blue, Yellow/Gold, Red
        edgecolor='black',
        linewidth=1
    )
    
    plt.title('Accuracy by Genre: GPT-4o Strategy Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Genre', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.05)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.legend(title='Strategy', title_fontsize='11', fontsize='10', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = 'outputs/figures/genre_performance.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    generate_genre_plot()
