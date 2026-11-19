"""The trained classifier artifact and the agent eval cases: guard the numbers the README states."""
import json
from pathlib import Path

import numpy as np
import pytest

import predict
import tools

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "ml" / "artifacts"


def test_committed_metrics_support_the_readme_claims():
    m = json.loads((ART / "metrics.json").read_text())
    assert m["n_classes"] == 77 and m["n_test"] == 3080
    assert m["accuracy"] >= 0.92 and m["top3_accuracy"] >= 0.98 and m["macro_f1"] >= 0.92


def test_label_file_matches_the_classifier():
    labels = json.loads((ART / "label_names.json").read_text())
    assert len(labels) == 77 and len(set(labels)) == 77


def test_predict_returns_ranked_labelled_intents(monkeypatch):
    """Classifier on a stand-in embedding (the real one needs the 90 MB transformer)."""
    clf, labels = predict._load()
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(1, clf.n_features_in_))
    monkeypatch.setattr(predict, "embed", lambda texts: vec / np.linalg.norm(vec))
    top = predict.classify_intent("anything", top_k=5)
    conf = [t["confidence"] for t in top]
    assert len(top) == 5 and conf == sorted(conf, reverse=True) and 0 < sum(conf) <= 1.0001
    assert all(t["intent"] in labels for t in top)


def test_eval_cases_are_well_formed():
    cases = json.loads((ROOT / "eval" / "eval_cases.json").read_text())
    ids = [c["id"] for c in cases]
    names = {t.name for t in tools.ALL_TOOLS}
    assert len(ids) == len(set(ids)) and len(cases) >= 12
    assert all(c["message"].strip() and (c["expected_tool"] is None or c["expected_tool"] in names) for c in cases)
    assert {c["expected_tool"] for c in cases} >= names | {None}        # every tool, and the "no tool" case, is exercised
