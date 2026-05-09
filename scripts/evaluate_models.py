"""
Evaluate both the baseline and DistilBERT models on the test set.
Run after training notebooks have completed.

Usage:
    python scripts/evaluate_models.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from quickserve.nlp.intent_classifier import predict_intent, LABELS


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def evaluate(model_type: str, df: pd.DataFrame) -> dict:
    use_baseline = (model_type == 'baseline')
    preds = [predict_intent(t, use_baseline=use_baseline)['intent'] for t in df['text']]
    macro_f1 = f1_score(df['intent'], preds, average='macro')
    print(f"\n=== {model_type.upper()} ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(df['intent'], preds))
    return {"preds": preds, "macro_f1": macro_f1}


def plot_comparison(baseline_f1: float, distilbert_f1: float):
    plt.figure(figsize=(6, 4))
    models = ['TF-IDF + LogReg\n(Baseline)', 'DistilBERT\n(Fine-tuned)']
    scores = [baseline_f1, distilbert_f1]
    bars = plt.bar(models, scores, color=['#4c72b0', '#dd8452'])
    plt.ylim(0, 1.05)
    plt.ylabel('Macro F1 Score')
    plt.title('Intent Classifier — Model Comparison')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('docs/diagrams/model_comparison.png', dpi=150)
    plt.show()
    print("Saved comparison chart to docs/diagrams/model_comparison.png")


if __name__ == '__main__':
    test_df = load_jsonl('data/intents_test.jsonl')

    baseline_result = evaluate('baseline', test_df)
    distilbert_result = evaluate('distilbert', test_df)
    plot_comparison(baseline_result['macro_f1'], distilbert_result['macro_f1'])
