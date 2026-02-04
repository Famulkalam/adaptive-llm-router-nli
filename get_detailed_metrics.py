
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

def print_metrics(name, y_true, y_pred):
    print(f"\n{'='*40}")
    print(f"STRATEGY: {name}")
    print(f"{'='*40}")
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Macro F1
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1: {f1:.4f}")
    
    # Confusion Matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\nConfusion Matrix:")
    print(f"{'':>15} {'Pred ' + ' '.join([l[:4] for l in labels])}")
    for i, label in enumerate(labels):
        print(f"True {label:>10}: {cm[i]}")

def main():
    strategies = ['zero-shot', 'one-shot', 'few-shot', 'cot', 'adaptive']
    
    for strategy in strategies:
        try:
            # Load data
            path = f"data/predictions/{strategy}_predictions.csv"
            df = pd.read_csv(path)
            
            # Helper to find label column
            label_col = 'true_label' if 'true_label' in df.columns else 'label'
            
            # Clean labels
            y_true = df[label_col].astype(str).str.lower().str.strip()
            y_pred = df['predicted_label'].astype(str).str.lower().str.strip()
            
            # Remove NaNs if any (shouldn't be for valid runs, but safety first)
            valid_mask = y_pred != 'nan'
            if (~valid_mask).any():
                print(f"Warning: {strategy} has {(~valid_mask).sum()} invalid predictions (NaN)")
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
            print_metrics(strategy.upper(), y_true, y_pred)
            
        except FileNotFoundError:
            print(f"\nCould not find results for {strategy}")
        except Exception as e:
            print(f"\nError analyzing {strategy}: {e}")

if __name__ == "__main__":
    main()
