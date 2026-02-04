
import asyncio
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Project components
from src.data.loader import MultiNLILoader
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureExtractor
from src.prompts.templates import PromptTemplates
from src.llm.async_openai import AsyncOpenAIClient
from src.llm.async_batch import AsyncBatchProcessor
from src.gatekeeper.adaptive_gatekeeper import AdaptiveGatekeeper

# Configure MSc level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nli_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NLI_Main")

class NLIPipeline:
    def __init__(self, samples_per_genre: int = 50):
        self.samples_per_genre = samples_per_genre
        self.loader = MultiNLILoader()
        self.preprocessor = DataPreprocessor()
        self.extractor = FeatureExtractor(use_nltk=False) # Use simple for speed
        self.llm = AsyncOpenAIClient()
        self.processor = AsyncBatchProcessor(self.llm, batch_size=20)
        self.gatekeeper = AdaptiveGatekeeper(
            length_threshold=220, 
            negation_threshold=3, 
            overlap_threshold=0.4
        )

    async def run(self):
        logger.info("Initializing NLI Refinement Pipeline...")
        
        # 1. Data Prep
        raw_df = self.loader.load_dataset()
        clean_df = self.preprocessor.clean_data(raw_df)
        sampled_df = self.preprocessor.stratified_sample(clean_df, self.samples_per_genre)
        
        # 2. Feature Extraction
        df_features = self.extractor.extract_features_batch(sampled_df)
        
        # 3. Adaptive Routing
        df_routed = self.gatekeeper.add_routing_to_dataframe(df_features)
        
        # 4. Async Inference
        logger.info("Starting Async Batch Inference...")
        
        # Group by strategy to batch efficiently
        results_dfs = []
        for strategy in df_routed["routed_strategy"].unique():
            strat_df = df_routed[df_routed["routed_strategy"] == strategy].copy()
            logger.info(f"Executing Batch for Strategy: {strategy} ({len(strat_df)} samples)")
            
            # Special kwargs for hybrid or special strategies
            kwargs = {}
            if strategy == "6-shot":
                kwargs["max_tokens"] = 20
            elif strategy == "6-shot-cot":
                kwargs["max_tokens"] = 500
            
            def make_prompt(item):
                return PromptTemplates.get_prompt(strategy, item["premise"], item["hypothesis"])
            
            processed_df = await self.processor.process_dataframe_async(
                strat_df, make_prompt, strategy=strategy, **kwargs
            )
            results_dfs.append(processed_df)
            
        final_df = pd.concat(results_dfs).sort_index()
        
        # 5. Metrics & Analysis
        self.analyze_results(final_df)
        
        # 6. ROI Calculation
        self.calculate_roi(final_df)
        
        logger.info("Pipeline Execution Complete.")
        return final_df

    def analyze_results(self, df: pd.DataFrame):
        logger.info("Performing detailed metric analysis...")
        
        # Clean labels
        y_true = df["label"].astype(str).str.lower().str.strip()
        y_pred = df["predicted_label"].astype(str).str.lower().str.strip()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        logger.info(f"Overall Accuracy: {acc:.4f}")
        logger.info(f"Overall Macro-F1: {f1:.4f}")
        
        # Per-genre heatmap
        genre_metrics = []
        for genre in df["genre"].unique():
            genre_df = df[df["genre"] == genre]
            g_true = genre_df["label"].astype(str).str.lower().str.strip()
            g_pred = genre_df["predicted_label"].astype(str).str.lower().str.strip()
            genre_metrics.append({
                "genre": genre,
                "accuracy": accuracy_score(g_true, g_pred),
                "f1": f1_score(g_true, g_pred, average='macro')
            })
            
        metrics_df = pd.DataFrame(genre_metrics).set_index("genre")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df[["accuracy"]].T, annot=True, cmap="YlGnBu", cbar=False)
        plt.title("NLI Accuracy Heatmap by Genre (GPT-4o Adaptive)")
        plt.savefig("outputs/figures/genre_heatmap.png")
        logger.info("Genre heatmap saved to outputs/figures/genre_heatmap.png")

    def calculate_roi(self, df: pd.DataFrame):
        """
        Calculates ROI: Always Few-Shot vs Adaptive Gatekeeper.
        Estimates cost based on token counts.
        """
        logger.info("Calculating ROI (Always Few-Shot vs Adaptive)...")
        
        # Total tokens used by Adaptive
        adaptive_tokens = df["prompt_tokens"].sum() + df["response_tokens"].sum()
        
        # Estimate "Always Few-Shot" cost
        # We'll use the average tokens of 6-shot samples as a baseline
        few_shot_df = df[df["routed_strategy"] == "6-shot"]
        if len(few_shot_df) > 0:
            avg_fs_tokens = (few_shot_df["prompt_tokens"].mean() + few_shot_df["response_tokens"].mean())
        else:
            # Fallback based on previous runs if 6-shot wasn't triggered
            avg_fs_tokens = 500 
            
        baseline_tokens = len(df) * avg_fs_tokens
        
        savings = (baseline_tokens - adaptive_tokens) / baseline_tokens * 100
        
        logger.info(f"Adaptive Total Tokens: {adaptive_tokens:,}")
        logger.info(f"Always-Few-Shot Baseline (Est): {baseline_tokens:,.0f}")
        logger.info(f"Token ROI (Savings): {savings:.2f}%")
        
        # Price estimation (OpenAI gpt-4o-mini pricing roughly 0.15/1M in, 0.60/1M out)
        # Simplified for demonstration
        cost_unit = 0.0000003 # 0.3USD per 1M tokens approx
        logger.info(f"Estimated Cost Saving: ${(baseline_tokens - adaptive_tokens) * cost_unit:.4f}")

async def main():
    pipeline = NLIPipeline(samples_per_genre=50)
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
