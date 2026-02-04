"""
Visualization Module

Creates plots and charts for NLI analysis including:
- Accuracy comparison charts
- Cost-accuracy frontier
- Confusion matrices
- Per-genre heatmaps
- Gatekeeper decision distribution
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


class Visualizer:
    """
    Visualization tools for NLI classification analysis.
    
    Creates publication-quality figures for:
    - Strategy comparison
    - Error analysis
    - Cost-benefit analysis
    - Per-genre breakdown
    """
    
    # Color palettes
    STRATEGY_COLORS = {
        "zero-shot": "#3498db",   # Blue
        "one-shot": "#2ecc71",    # Green
        "few-shot": "#f39c12",    # Orange
        "cot": "#e74c3c"          # Red
    }
    
    LABEL_COLORS = {
        "entailment": "#27ae60",      # Green
        "neutral": "#3498db",          # Blue
        "contradiction": "#e74c3c"     # Red
    }
    
    def __init__(
        self,
        output_dir: str = "outputs/figures",
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    def plot_strategy_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = "accuracy",
        title: str = "Strategy Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar chart comparing strategies.
        
        Args:
            metrics_df: DataFrame with strategy metrics
            metric: Metric to plot (accuracy, macro_f1, etc.)
            title: Plot title
            save_name: Optional filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        strategies = metrics_df["strategy"].tolist()
        values = metrics_df[metric].tolist()
        colors = [self.STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]
        
        bars = ax.bar(strategies, values, color=colors, edgecolor="white", linewidth=1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Plot title
            save_name: Optional filename
            normalize: Whether to normalize
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if normalize:
            cm = confusion_matrix.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums != 0)
            fmt = ".2f"
        else:
            cm = confusion_matrix
            fmt = "d"
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_per_genre_heatmap(
        self,
        genre_metrics: pd.DataFrame,
        metric: str = "accuracy",
        title: str = "Performance by Genre",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Heatmap of performance across genres and strategies.
        
        Args:
            genre_metrics: DataFrame with genre and strategy columns
            metric: Metric to plot
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot if needed
        if "strategy" in genre_metrics.columns:
            pivot_df = genre_metrics.pivot(index="genre", columns="strategy", values=metric)
        else:
            pivot_df = genre_metrics.set_index("genre")[[col for col in genre_metrics.columns if col != "genre"]]
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            ax=ax,
            cbar_kws={'label': metric.replace("_", " ").title()}
        )
        
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Genre", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cost_accuracy_frontier(
        self,
        frontier_df: pd.DataFrame,
        title: str = "Cost-Accuracy Frontier",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Scatter plot of cost vs accuracy with frontier.
        
        Args:
            frontier_df: DataFrame with cost_usd and accuracy columns
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for _, row in frontier_df.iterrows():
            color = self.STRATEGY_COLORS.get(row["strategy"], "#95a5a6")
            marker = "o" if row.get("pareto_optimal", False) else "s"
            size = 200 if row.get("pareto_optimal", False) else 100
            
            ax.scatter(
                row["cost_usd"],
                row["accuracy"],
                c=color,
                s=size,
                marker=marker,
                edgecolors="white",
                linewidth=2,
                label=row["strategy"],
                zorder=10
            )
            
            # Add label
            ax.annotate(
                row["strategy"],
                (row["cost_usd"], row["accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10
            )
        
        # Draw Pareto frontier line
        pareto = frontier_df[frontier_df.get("pareto_optimal", pd.Series([True]*len(frontier_df)))]
        if len(pareto) > 1:
            pareto_sorted = pareto.sort_values("cost_usd")
            ax.plot(
                pareto_sorted["cost_usd"],
                pareto_sorted["accuracy"],
                'k--',
                alpha=0.5,
                linewidth=2,
                label="Pareto Frontier"
            )
        
        ax.set_xlabel("Cost (USD)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_gatekeeper_distribution(
        self,
        decisions: pd.DataFrame,
        title: str = "Adaptive Gatekeeper Routing Distribution",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Pie/bar chart of gatekeeper routing decisions.
        
        Args:
            decisions: DataFrame with routed_strategy column
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        strategy_counts = decisions["routed_strategy"].value_counts()
        colors = [self.STRATEGY_COLORS.get(s, "#95a5a6") for s in strategy_counts.index]
        
        axes[0].pie(
            strategy_counts.values,
            labels=strategy_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(strategy_counts),
            shadow=True,
            startangle=90
        )
        axes[0].set_title("Routing Distribution", fontsize=12, fontweight='bold')
        
        # By genre
        if "genre" in decisions.columns:
            genre_routing = pd.crosstab(decisions["genre"], decisions["routed_strategy"], normalize="index")
            genre_routing.plot(
                kind="barh",
                stacked=True,
                ax=axes[1],
                color=[self.STRATEGY_COLORS.get(c, "#95a5a6") for c in genre_routing.columns]
            )
            axes[1].set_xlabel("Proportion", fontsize=11)
            axes[1].set_ylabel("Genre", fontsize=11)
            axes[1].set_title("Routing by Genre", fontsize=12, fontweight='bold')
            axes[1].legend(title="Strategy", bbox_to_anchor=(1.02, 1))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_prompt_progression(
        self,
        strategy_metrics: Dict[str, float],
        metric: str = "accuracy",
        title: str = "Prompt Strategy Progression",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Line plot showing progression from zero to CoT.
        
        Args:
            strategy_metrics: Dict mapping strategy to metric value
            metric: Metric name for y-axis label
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Order strategies
        order = ["zero-shot", "one-shot", "few-shot", "cot"]
        strategies = [s for s in order if s in strategy_metrics]
        values = [strategy_metrics[s] for s in strategies]
        
        # Plot line
        ax.plot(strategies, values, 'bo-', linewidth=2, markersize=10)
        
        # Fill area under curve
        ax.fill_between(strategies, values, alpha=0.3)
        
        # Add value labels
        for x, y in zip(strategies, values):
            ax.annotate(f'{y:.3f}',
                       xy=(x, y),
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center',
                       fontsize=11,
                       fontweight='bold')
        
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(
        self,
        strategy_comparison: pd.DataFrame,
        confusion_matrices: Dict[str, np.ndarray],
        cost_frontier: pd.DataFrame,
        labels: List[str] = None,
        save_name: str = "dashboard.png"
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard figure.
        
        Args:
            strategy_comparison: DataFrame comparing strategies
            confusion_matrices: Dict mapping strategy to confusion matrix
            cost_frontier: Cost-accuracy frontier data
            labels: Class labels
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = ["entailment", "neutral", "contradiction"]
        
        fig = plt.figure(figsize=(16, 12))
        
        # Strategy comparison (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        strategies = strategy_comparison["strategy"].tolist()
        values = strategy_comparison["accuracy"].tolist()
        colors = [self.STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]
        bars = ax1.bar(strategies, values, color=colors)
        ax1.set_title("Strategy Comparison (Accuracy)", fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords="offset points", ha='center')
        
        # Cost-accuracy frontier (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        for _, row in cost_frontier.iterrows():
            color = self.STRATEGY_COLORS.get(row["strategy"], "#95a5a6")
            ax2.scatter(row["cost_usd"], row["accuracy"], c=color, s=150,
                       edgecolors="white", linewidth=2)
            ax2.annotate(row["strategy"], (row["cost_usd"], row["accuracy"]),
                        xytext=(5, 5), textcoords="offset points")
        ax2.set_xlabel("Cost (USD)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Cost-Accuracy Frontier", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Best strategy confusion matrix (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        best_strategy = strategy_comparison.iloc[0]["strategy"]
        if best_strategy in confusion_matrices:
            cm = confusion_matrices[best_strategy]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                       xticklabels=labels, yticklabels=labels, ax=ax3)
            ax3.set_title(f"Confusion Matrix ({best_strategy})", fontsize=12, fontweight='bold')
        
        # Per-class F1 comparison (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        f1_data = []
        for _, row in strategy_comparison.iterrows():
            for label in labels:
                f1_col = f"{label}_f1"
                if f1_col in row:
                    f1_data.append({"strategy": row["strategy"], "label": label, "f1": row[f1_col]})
        
        if f1_data:
            f1_df = pd.DataFrame(f1_data)
            f1_pivot = f1_df.pivot(index="label", columns="strategy", values="f1")
            f1_pivot.plot(kind="bar", ax=ax4, color=[self.STRATEGY_COLORS.get(c, "#95a5a6") 
                                                     for c in f1_pivot.columns])
            ax4.set_title("Per-Class F1 Scores", fontsize=12, fontweight='bold')
            ax4.set_ylim(0, 1.1)
            ax4.legend(title="Strategy")
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.suptitle("NLI Classification Analysis Dashboard", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Demo visualizations
    visualizer = Visualizer()
    
    # Create sample data
    strategy_df = pd.DataFrame({
        "strategy": ["zero-shot", "one-shot", "few-shot", "cot"],
        "accuracy": [0.70, 0.75, 0.80, 0.85],
        "macro_f1": [0.68, 0.73, 0.78, 0.83]
    })
    
    # Plot strategy comparison
    fig = visualizer.plot_strategy_comparison(
        strategy_df,
        metric="accuracy",
        title="Strategy Comparison",
        save_name="strategy_comparison_demo.png"
    )
    
    print("Demo visualizations saved to outputs/figures/")
    plt.show()
