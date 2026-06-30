"""
Trains a real 77-way intent classifier on the banking77 dataset (13,083
real customer-service queries to a UK-based digital bank, fine-grained
intents like "lost_card", "balance_not_updated_after_bank_transfer",
"declined_transfer") - real data, real held-out test set, real accuracy.

Approach: embed every query with a real HuggingFace transformer
(all-MiniLM-L6-v2, see embeddings.py), then train a linear classifier on
top of the embeddings. This is standard practice (embed once, train a fast
linear head) rather than full fine-tuning - full BERT fine-tuning on CPU
for 13K examples would take a long time for marginal accuracy gain over a
well-chosen linear classifier on strong pretrained embeddings.

Run: python train_classifier.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from embeddings import embed

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset("banking77")
    label_names = dataset["train"].features["label"].names

    print(f"Embedding {len(dataset['train'])} train + {len(dataset['test'])} test queries...")
    X_train = embed(dataset["train"]["text"])
    y_train = np.array(dataset["train"]["label"])
    X_test = embed(dataset["test"]["text"])
    y_test = np.array(dataset["test"]["label"])

    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "top3_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=3, labels=clf.classes_)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(label_names),
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(clf, ARTIFACTS_DIR / "intent_classifier.joblib")
    (ARTIFACTS_DIR / "label_names.json").write_text(json.dumps(label_names, indent=2))
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved classifier + metrics to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
