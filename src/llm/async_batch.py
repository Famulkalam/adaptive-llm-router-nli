
import asyncio
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
from tqdm.asyncio import tqdm
import time

from .base import LLMResponse
from .async_openai import AsyncOpenAIClient

class AsyncBatchProcessor:
    """Processor for running LLM requests in parallel."""
    
    def __init__(self, llm: AsyncOpenAIClient, batch_size: int = 10):
        self.llm = llm
        self.batch_size = batch_size

    async def _process_single(
        self,
        item: Dict[str, Any],
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str,
        **generate_kwargs
    ) -> LLMResponse:
        prompt = prompt_fn(item)
        return await self.llm.generate_async(
            prompt=prompt,
            strategy=strategy,
            **generate_kwargs
        )

    async def process_batch_async(
        self,
        items: List[Dict[str, Any]],
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str = "zero-shot",
        show_progress: bool = True,
        **generate_kwargs
    ) -> List[LLMResponse]:
        """Process a list of items asynchronously with semi-concurrency control."""
        responses = []
        
        # We can use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.batch_size)
        
        async def sem_process(item):
            async with semaphore:
                return await self._process_single(item, prompt_fn, strategy, **generate_kwargs)
        
        tasks = [sem_process(item) for item in items]
        
        if show_progress:
            responses = await tqdm.gather(*tasks, desc=f"Processing ({strategy})")
        else:
            responses = await asyncio.gather(*tasks)
            
        return responses

    async def process_dataframe_async(
        self,
        df: pd.DataFrame,
        prompt_fn: Callable[[Dict[str, Any]], str],
        strategy: str = "zero-shot",
        show_progress: bool = True,
        **generate_kwargs
    ) -> pd.DataFrame:
        """Process a DataFrame asynchronously and return with results."""
        items = df.to_dict('records')
        responses = await self.process_batch_async(
            items, prompt_fn, strategy, show_progress, **generate_kwargs
        )
        
        df_result = df.copy()
        df_result["predicted_label"] = [r.get_label() for r in responses]
        df_result["raw_response"] = [r.text for r in responses]
        df_result["prompt_tokens"] = [r.prompt_tokens for r in responses]
        df_result["response_tokens"] = [r.response_tokens for r in responses]
        df_result["latency_ms"] = [r.latency_ms for r in responses]
        df_result["success"] = [r.success for r in responses]
        df_result["error"] = [r.error if not r.success else "" for r in responses]
        
        return df_result
