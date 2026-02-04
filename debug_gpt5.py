
import sys
import pandas as pd
from src.llm.gemini import create_client
from src.llm.batch import BatchProcessor
from src.prompts.templates import PromptTemplates

# Setup
client = create_client(provider="openai")
print(f"Testing model: {client.model_name}")

# Prepare data (mock dataframe)
df = pd.DataFrame([
    {"premise": "The cat sat on the mat.", "hypothesis": "The cat is on the mat.", "label": "entailment"},
    {"premise": "He logic is flawed.", "hypothesis": "He is smart.", "label": "neutral"}
])

# Create prompt function
def prompt_fn(item):
    return PromptTemplates.get_zero_shot_prompt(item["premise"], item["hypothesis"])

# Run batch
processor = BatchProcessor(llm=client, batch_size=2, max_retries=1)
result_df = processor.process_dataframe(df, prompt_fn, strategy="zero-shot")

# Inspect results
print("\nResults:")
print(result_df[["success", "error"]])
