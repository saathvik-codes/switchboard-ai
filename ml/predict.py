"""Inference helper: classifies a customer message into one of the 77
banking77 intents, loading the trained artifact once."""
import json
import os
import tempfile
from pathlib import Path

import joblib

from embeddings import embed

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILES = ("intent_classifier.joblib", "label_names.json")

_clf = None
_labels = None


def _artifacts_dir() -> Path:
    """Where the trained classifier lives. With MODEL_BUCKET set (the AWS deployment), the artifacts are downloaded
    once from s3://MODEL_BUCKET/MODEL_PREFIX/ into temporary storage, so a retrained model ships without a redeploy."""
    bucket = os.getenv("MODEL_BUCKET")
    if not bucket:
        return ARTIFACTS_DIR
    import boto3
    prefix = os.getenv("MODEL_PREFIX", "switchboard/v1").strip("/")
    target = Path(tempfile.gettempdir()) / "switchboard-model" / prefix
    target.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for name in ARTIFACT_FILES:
        if not (target / name).exists():
            s3.download_file(bucket, f"{prefix}/{name}", str(target / name))
    return target


def _load():
    global _clf, _labels
    if _clf is None:
        where = _artifacts_dir()
        _clf = joblib.load(where / "intent_classifier.joblib")
        _labels = json.loads((where / "label_names.json").read_text())
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
