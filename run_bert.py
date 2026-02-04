
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os

def run_bert_inference():
    print("🚀 Initializing BERT Baseline (cross-encoder/nli-distilroberta-base)...")
    model_name = "cross-encoder/nli-distilroberta-base"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load dataset (using the same sample as LLM runs)
    try:
        df = pd.read_csv('data/processed/full_sample.csv')
    except:
        print("Error: Could not find processed sample. Run main.py first.")
        return

    # Mapping for this specific model
    # 0: contradiction, 1: entailment, 2: neutral
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    
    results = []
    print(f"Running inference on {len(df)} samples using {device}...")
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            inputs = tokenizer(
                row['premise'], 
                row['hypothesis'], 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(device)
            
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            results.append(id2label[pred_id])
    
    df['bert_prediction'] = results
    
    # Metrics
    y_true = df['label'].str.lower().str.strip()
    y_pred = df['bert_prediction'].str.lower().str.strip()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*40)
    print("BERT (Fine-tuned MNLI) PERFORMANCE")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    # Save results
    output_path = "data/predictions/bert_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved BERT predictions to {output_path}")

    # Sector breakdown
    genre_metrics = df.groupby('genre').apply(
        lambda x: accuracy_score(x['label'], x['bert_prediction'])
    )
    print("\nAccuracy by Genre:")
    print(genre_metrics)

if __name__ == "__main__":
    run_bert_inference()
