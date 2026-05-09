import re
import unicodedata


def clean_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove most punctuation."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_order_ref(text: str) -> str | None:
    """Pull the first ORD-style reference out of user text."""
    match = re.search(r"\b(ord\d+)\b", text.lower())
    return match.group(1).upper() if match else None
