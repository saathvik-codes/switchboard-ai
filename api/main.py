"""FastAPI wrapper around the agent. Run with:
    uvicorn api.main:app --reload --port 8020
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from agent import handle_message  # noqa: E402

app = FastAPI(title="Switchboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MessageRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: MessageRequest):
    return handle_message(req.message)
