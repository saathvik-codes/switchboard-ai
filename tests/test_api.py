import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_and_static_assets_are_served(client):
    assert "<html" in client.get("/").text.lower()
    assert client.get("/style.css").headers["content-type"].startswith("text/css")
    assert client.get("/script.js").status_code == 200


def test_metadata_reports_the_committed_training_metrics(client):
    committed = json.loads((ROOT / "ml" / "artifacts" / "metrics.json").read_text())
    body = client.get("/metadata").json()
    assert body["accuracy"] == committed["accuracy"] and body["dataset"] == "banking77"


def test_presets_cover_distinct_tools_and_are_capped_at_five(client):
    presets = client.get("/presets").json()
    assert 1 <= len(presets) <= 5 and all({"id", "message", "expected"} <= p.keys() for p in presets)


def test_state_exposes_the_mock_bank(client):
    body = client.get("/state").json()
    assert body["cards"]["CARD-4521"]["status"] == "active" and body["accounts"]["ACC-1001"]["currency"] == "GBP"


def test_chat_returns_what_the_agent_returns(client, monkeypatch):
    monkeypatch.setattr(main, "handle_message", lambda message: {"message": message, "response": "hi", "tool_calls": []})
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200 and r.json()["response"] == "hi"


@pytest.mark.parametrize("payload", [{"message": ""}, {}, {"message": "x" * 2001}])
def test_chat_rejects_empty_missing_or_oversized_messages(client, monkeypatch, payload):
    monkeypatch.setattr(main, "handle_message", lambda message: pytest.fail("the agent must not be called"))
    assert client.post("/chat", json=payload).status_code == 422
