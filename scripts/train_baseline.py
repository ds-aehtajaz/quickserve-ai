"""Train the baseline TF-IDF + Logistic Regression intent classifier and save it."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

TRAIN_PATH = 'data/intents_train.jsonl'
TEST_PATH = 'data/intents_test.jsonl'
MODEL_OUT = 'models/baseline.joblib'


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


train_df = load_jsonl(TRAIN_PATH)
test_df = load_jsonl(TEST_PATH)
print(f'Train: {len(train_df)} | Test: {len(test_df)}')

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs'))
])

model.fit(train_df['text'], train_df['intent'])

y_pred = model.predict(test_df['text'])
macro_f1 = f1_score(test_df['intent'], y_pred, average='macro')
print(f'\nMacro F1: {macro_f1:.4f}')
print(classification_report(test_df['intent'], y_pred))

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)
print(f'Model saved to {MODEL_OUT}')
