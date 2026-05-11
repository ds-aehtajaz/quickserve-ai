import os
import joblib
from .preprocess import clean_text

# torch and transformers are imported lazily inside _load_distilbert()
# to avoid the ~700 MB memory hit when only the baseline is used (e.g. on Streamlit Cloud).

LABELS = [
    "greeting", "goodbye", "place_order", "track_order",
    "cancel_order", "modify_order", "faq", "talk_to_human", "fallback"
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_distilbert_model = None
_distilbert_tokenizer = None
_baseline_model = None


def _load_distilbert(model_path: str):
    global _distilbert_model, _distilbert_tokenizer
    if _distilbert_model is None:
        # Lazy imports — only loaded when DistilBERT is actually used
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _distilbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _distilbert_model.eval()


def _load_baseline(model_path: str):
    global _baseline_model
    if _baseline_model is None:
        _baseline_model = joblib.load(model_path)


def predict_intent(
    text: str,
    model_path: str | None = None,
    use_baseline: bool = False,
) -> dict:
    """
    Returns {"intent": str, "confidence": float}.

    Falls back to the baseline TF-IDF model if the DistilBERT model directory
    is not found, or if use_baseline=True.
    """
    cleaned = clean_text(text)

    distilbert_path = model_path or os.getenv("INTENT_MODEL_PATH", "models/distilbert-intent")
    baseline_path = os.getenv("BASELINE_MODEL_PATH", "models/baseline.joblib")

    distilbert_ready = os.path.isfile(os.path.join(distilbert_path, "config.json"))
    if use_baseline or not distilbert_ready:
        _load_baseline(baseline_path)
        proba = _baseline_model.predict_proba([cleaned])[0]
        label_idx = int(proba.argmax())
        # baseline pipeline stores classes in the same order as LABELS
        classes = list(_baseline_model.classes_)
        intent = classes[label_idx]
        confidence = float(proba[label_idx])
        return {"intent": intent, "confidence": confidence}

    _load_distilbert(distilbert_path)
    # Lazy import — only when DistilBERT path is actually used
    import torch
    inputs = _distilbert_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        logits = _distilbert_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_idx = int(probs.argmax())
    # Use the id2label stored in the model config — it may differ from the
    # hardcoded LABELS order if the model was trained with sorted labels.
    model_id2label = _distilbert_model.config.id2label
    return {
        "intent": model_id2label[label_idx],
        "confidence": float(probs[label_idx]),
    }
