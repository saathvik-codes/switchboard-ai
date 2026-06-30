"""Inference helper: classifies a customer message into one of the 77
banking77 intents, loading the trained artifact once."""
import json
from pathlib import Path

import joblib

from embeddings import embed

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

_clf = None
_labels = None


def _load():
    global _clf, _labels
    if _clf is None:
        _clf = joblib.load(ARTIFACTS_DIR / "intent_classifier.joblib")
        _labels = json.loads((ARTIFACTS_DIR / "label_names.json").read_text())
    return _clf, _labels


def classify_intent(message: str, top_k: int = 3) -> list[dict]:
    """Returns the top-k predicted intents with confidence scores, ranked."""
    clf, labels = _load()
    vec = embed([message])
    proba = clf.predict_proba(vec)[0]
    ranked = sorted(zip(clf.classes_, proba), key=lambda x: -x[1])[:top_k]
    return [{"intent": labels[label_idx], "confidence": round(float(p), 4)} for label_idx, p in ranked]


if __name__ == "__main__":
    for msg in ["I lost my card, please help", "why is my exchange rate so bad", "can I top up automatically"]:
        print(msg, "->", classify_intent(msg))
