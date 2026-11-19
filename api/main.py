"""FastAPI wrapper around the agent. Run with:
    uvicorn api.main:app --reload --port 8020
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from agent import handle_message  # noqa: E402

app = FastAPI(title="Switchboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/style.css")
def style():
    css_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
    return HTMLResponse(content=css_path.read_text(encoding="utf-8"), media_type="text/css")


@app.get("/script.js")
def script():
    js_path = Path(__file__).resolve().parent.parent / "static" / "script.js"
    return HTMLResponse(content=js_path.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/metadata")
def get_metadata():
    import json
    from pathlib import Path
    metrics_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["dataset"] = "banking77"
        metrics["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        return metrics
    return {
        "dataset": "banking77",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "accuracy": 0.9295,
        "n_classes": 77,
        "n_train": 10003
    }


@app.get("/presets")
def get_presets():

    import json
    cases_path = Path(__file__).resolve().parent.parent / "eval" / "eval_cases.json"
    if cases_path.exists():
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        presets = []
        seen_tools = set()
        for case in cases:
            expected = case["expected_tool"]
            if expected not in seen_tools or expected is None:
                if expected is not None:
                    seen_tools.add(expected)
                presets.append({
                    "id": case["id"],
                    "message": case["message"],
                    "expected": expected
                })
        return presets[:5]
    return []


@app.get("/state")
def get_state():
    from tools import _MOCK_ACCOUNTS, _MOCK_CARDS, _MOCK_TRANSACTIONS
    return {
        "accounts": _MOCK_ACCOUNTS,
        "cards": _MOCK_CARDS,
        "transactions": _MOCK_TRANSACTIONS
    }


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/chat")
def chat(req: MessageRequest):
    return handle_message(req.message)

