import os
import re
import spacy
from .preprocess import extract_order_ref

_nlp = None


def _load_model(model_path: str):
    global _nlp
    if _nlp is None:
        if os.path.isdir(model_path):
            _nlp = spacy.load(model_path)
        else:
            # Fall back to base English model before custom NER is trained
            _nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str, model_path: str | None = None) -> dict:
    """
    Returns {"ITEM": str | None, "QUANTITY": int | None, "ORDER_ID": str | None}.

    Uses the trained spaCy model when available, with a regex fallback for
    ORDER_ID so tracking/cancellation works even before training is done.
    """
    path = model_path or os.getenv("NER_MODEL_PATH", "models/ner")
    _load_model(path)

    doc = _nlp(text)
    entities: dict = {"ITEM": None, "QUANTITY": None, "ORDER_ID": None}

    for ent in doc.ents:
        label = ent.label_.upper()
        if label in entities:
            if label == "QUANTITY":
                try:
                    entities[label] = int(ent.text)
                except ValueError:
                    entities[label] = ent.text
            elif label == "ORDER_ID":
                entities[label] = ent.text.upper()
            else:
                entities[label] = ent.text

    # Regex fallback for ORDER_ID regardless of model state
    if entities["ORDER_ID"] is None:
        entities["ORDER_ID"] = extract_order_ref(text)

    # Regex fallback for QUANTITY (e.g. "2 pizzas", "three sandwiches")
    if entities["QUANTITY"] is None:
        qty_match = re.search(r"\b(\d+)\b", text)
        if qty_match:
            entities["QUANTITY"] = int(qty_match.group(1))

    return entities
