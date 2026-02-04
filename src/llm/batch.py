"""
Batch Processor for LLM Inference

Handles batch processing with:
- Configurable batch sizes
- Retry logic with exponential backoff
- Progress tracking
- Error handling and recovery
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

from .base import BaseLLM, LLMResponse


@dataclass
class BatchResult:
    """Results from a batch processing run."""
    responses: List[LLMResponse]
    successful: int
    failed: int
    total_tokens: int
    total_time_ms: float
    retries: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful + self.failed
        return self.successful / max(total, 1)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert responses to DataFrame."""
        records = []
        for resp in self.responses:
            records.append({
                "text": resp.text,
                "label": resp.get_label(),
                "prompt_tokens": resp.prompt_tokens,
                "response_tokens": resp.response_tokens,
                "total_tokens": resp.total_tokens,
                "latency_ms": resp.latency_ms,
                "success": resp.success,
                "error": resp.error,
                "strategy": resp.strategy
            })
        return pd.DataFrame(records)


class BatchProcessor:
    """
    Batch processor for LLM inference.
    
    Features:
    - Configurable batch processing
    - Exponential backoff retry
    - Progress tracking with tqdm
    - Error recovery
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 1
    ):
        """
        Initialize the batch processor.
        
        Args:
            llm: LLM client to use
            batch_size: Number of items per batch
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
            max_workers: Max parallel workers (1 = sequential)
        """
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
    
    def _process_single(
        self,
        item: Dict[str, Any],
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str,
        **generate_kwargs
    ) -> LLMResponse:
        """
        Process a single item with retry logic.
        
        Args:
            item: Data item to process
            prompt_fn: Function to generate prompt from item
            strategy: Prompting strategy
            **generate_kwargs: Additional args for LLM generation
            
        Returns:
            LLMResponse
        """
        prompt = prompt_fn(item)
        
        for attempt in range(self.max_retries):
            response = self.llm.generate(
                prompt=prompt,
                strategy=strategy,
                true_label=item.get("label"),
                premise=item.get("premise"),
                hypothesis=item.get("hypothesis"),
                **generate_kwargs
            )
            
            if response.success:
                return response
            
            # Check if error is retryable
            if response.error and any(x in response.error.lower() 
                                       for x in ["rate limit", "timeout", "unavailable"]):
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                # Non-retryable error
                break
        
        return response
    
    def process_batch(
        self,
        items: List[Dict[str, Any]],
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str = "zero-shot",
        show_progress: bool = True,
        desc: Optional[str] = None,
        **generate_kwargs
    ) -> BatchResult:
        """
        Process a batch of items.
        
        Args:
            items: List of data items
            prompt_fn: Function to generate prompts
            strategy: Prompting strategy
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            **generate_kwargs: Additional args for LLM generation
            
        Returns:
            BatchResult with all responses
        """
        start_time = time.time()
        responses = []
        successful = 0
        failed = 0
        retries = 0
        
        desc = desc or f"Processing ({strategy})"
        iterator = tqdm(items, desc=desc, disable=not show_progress)
        
        for item in iterator:
            # Pass generate_kwargs to _process_single (which needs update too)
            response = self._process_single(item, prompt_fn, strategy, **generate_kwargs)
            responses.append(response)
            
            if response.success:
                successful += 1
            else:
                failed += 1
            
            # Update progress bar description
            if show_progress:
                iterator.set_postfix({
                    "success": successful,
                    "failed": failed,
                    "tokens": sum(r.total_tokens for r in responses)
                })
        
        total_time = (time.time() - start_time) * 1000
        total_tokens = sum(r.total_tokens for r in responses)
        
        return BatchResult(
            responses=responses,
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            retries=retries
        )
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str = "zero-shot",
        show_progress: bool = True,
        **generate_kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame and return results as a new DataFrame.
        
        Args:
            df: Input DataFrame with premise/hypothesis columns
            prompt_fn: Function to generate prompts
            strategy: Prompting strategy
            show_progress: Whether to show progress
            **generate_kwargs: Additional args for LLM generation
            
        Returns:
            DataFrame with original data plus predictions
        """
        items = df.to_dict("records")
        
        result = self.process_batch(
            items=items,
            prompt_fn=prompt_fn,
            strategy=strategy,
            show_progress=show_progress,
            **generate_kwargs
        )
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result["predicted_label"] = [r.get_label() for r in result.responses]
        df_result["raw_response"] = [r.text for r in result.responses]
        df_result["prompt_tokens"] = [r.prompt_tokens for r in result.responses]
        df_result["response_tokens"] = [r.response_tokens for r in result.responses]
        df_result["latency_ms"] = [r.latency_ms for r in result.responses]
        df_result["success"] = [r.success for r in result.responses]
        df_result["error"] = [r.error for r in result.responses]
        
        return df_result
    
    def compare_strategies(
        self,
        df: pd.DataFrame,
        prompt_fn_factory: Callable[[str], Callable],
        strategies: List[str] = None,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run inference with multiple strategies and compare.
        
        Args:
            df: Input DataFrame
            prompt_fn_factory: Function that takes strategy and returns prompt_fn
            strategies: List of strategies to compare
            show_progress: Whether to show progress
            
        Returns:
            Dictionary mapping strategy to result DataFrame
        """
        if strategies is None:
            strategies = ["zero-shot", "one-shot", "few-shot", "cot"]
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"Running {strategy.upper()} strategy...")
            print('='*50)
            
            prompt_fn = prompt_fn_factory(strategy)
            result_df = self.process_dataframe(
                df=df,
                prompt_fn=prompt_fn,
                strategy=strategy,
                show_progress=show_progress
            )
            results[strategy] = result_df
        
        return results


if __name__ == "__main__":
    # Test batch processing
    from .mock_gemini import MockGeminiClient
    
    # Create mock client
    client = MockGeminiClient(random_seed=42)
    processor = BatchProcessor(llm=client, batch_size=10)
    
    # Test data
    test_items = [
        {"premise": "The cat sat on the mat.", "hypothesis": "An animal was sitting.", "label": "entailment"},
        {"premise": "The store is closed.", "hypothesis": "The store is open.", "label": "contradiction"},
        {"premise": "She went to study.", "hypothesis": "She checked out books.", "label": "neutral"},
    ]
    
    # Prompt function
    def simple_prompt(item):
        return f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nClassification:"
    
    print("Batch Processing Test")
    print("=" * 50)
    
    result = processor.process_batch(
        items=test_items,
        prompt_fn=simple_prompt,
        strategy="zero-shot"
    )
    
    print(f"\nResults:")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Time: {result.total_time_ms:.0f}ms")
    
    df_result = result.to_dataframe()
    print(f"\nPredictions:")
    for _, row in df_result.iterrows():
        print(f"  {row['label']}: latency={row['latency_ms']:.0f}ms")
