# Switchboard — Intent-Classified Banking Service Agent

A customer-service agent for banking queries: a real HuggingFace
transformer classifies what the customer wants (93% accuracy, 77 fine-grained
intents), and a LangChain tool-calling agent decides what to do about it —
check a balance, freeze a card, pull transaction history, or escalate to a
human — extracting the right parameters from plain English.

## Why this exists, and why it's two models instead of one

It would be simpler to just hand the whole message to an LLM and ask it to
both understand the intent and pick a tool. The reason this project
doesn't do that: a classifier trained on 10,000 real labeled examples is
measurably more reliable at intent recognition than an LLM guessing
zero-shot across 77 fine-grained categories, and that's a claim this repo
backs with a held-out test set rather than asserting it. But the
classifier alone can't extract "the card number is 4521" from a sentence,
hold context, or decide *what to do* with an identified intent — that
needs an LLM. So the classifier's top prediction is handed to the agent as
a hint ("this looks like `lost_or_stolen_card`, 91% confidence"), and the
LLM is explicitly told to trust the actual message over the hint if they
disagree. Combining a real classifier with a real tool-calling agent, each
doing the part it's actually good at.

## Real, verified numbers

**Intent classification** (`ml/train_classifier.py`) — banking77 dataset,
13,083 real customer messages to a digital bank, 77 intents:

| Metric | Score |
|---|---|
| Accuracy (3,080 held-out test examples) | **93.0%** |
| Top-3 accuracy | **98.4%** |
| Macro F1 | **0.93** |

Real HuggingFace transformer embeddings (`all-MiniLM-L6-v2`, mean-pooled
token embeddings via `transformers.AutoModel` — same weights the
`sentence-transformers` package wraps, loaded directly), with a linear
classifier on top rather than full fine-tuning: full BERT fine-tuning on
CPU for 13K examples buys little accuracy over a well-chosen linear head
on strong pretrained embeddings, for a lot more training time.

**Agent tool-selection** (`eval/run_eval.py`) — 12 hand-written test cases
covering account actions (should call a tool) and general policy
questions (shouldn't call any tool), run end-to-end against the real
classifier and a real local LLM (Ollama, `llama3.1:8b`):

- First pass: **8/12 (67%)** — found a real, common agentic-LLM failure
  mode: the model called a tool even for purely informational questions
  ("which currencies do you support" → checked the account balance).
- After tightening the system prompt to explicitly forbid tool calls for
  account-unrelated questions: **10/12 (83%)**.

Both numbers are reported, not just the improved one — the eval script is
in the repo and reproducible (`python eval/run_eval.py`), and the two
remaining failures are documented as real, current limitations rather than
hidden.

## Tech stack

- **HuggingFace**: `transformers` (embeddings), `datasets` (banking77)
- **LangChain**: tool-calling agent (`langchain-ollama`)
- **LLM**: Ollama, local, no API key required
- **ML**: scikit-learn (logistic regression on top of embeddings)
- **API**: FastAPI

## Running it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1:8b
ollama serve                     # if not already running

cd ml && python train_classifier.py   # trains the intent classifier (~2 min)
cd ../eval && python run_eval.py      # runs the agent eval
cd .. && uvicorn api.main:app --reload --port 8020
```

```bash
curl -X POST http://localhost:8020/chat -H "Content-Type: application/json" \
  -d '{"message": "my card was stolen, please freeze it"}'
```

## Changing the Dataset

By default, the intent classifier is trained on the `banking77` dataset. However, you can change the training dataset to any other text classification dataset (such as `DeepPavlov/clinc_oos` or a custom dataset).

To change the dataset:
1. Open [`ml/train_classifier.py`](file:///c:/Users/saath/Downloads/switchboard-ai/ml/train_classifier.py).
2. Modify the `load_dataset` call to load your target dataset (e.g., `load_dataset("DeepPavlov/clinc_oos", "plus")`).
3. Update the fields extracted (e.g., text, labels) to match your dataset's structure.
4. Run `python train_classifier.py` to train the new model and generate new classifier assets (`intent_classifier.joblib`, `label_names.json`, and `metrics.json`).

## Project structure

```
ml/
  embeddings.py          Real HuggingFace transformer embeddings
  train_classifier.py    Trains + evaluates the 77-way intent classifier
  predict.py              Inference helper
  artifacts/                 Trained classifier (committed - small, ~120KB)
agent/
  tools.py                Mock banking tools (clearly labeled as simulated)
  agent.py                 Orchestrator: classify intent -> LangChain tool-calling agent
eval/
  eval_cases.json           Curated tool-selection test cases
  run_eval.py                 Runs them end-to-end, reports real pass rate
api/
  main.py                      FastAPI wrapper
```

## What's simulated vs. real

The banking backend (`agent/tools.py`) is an in-memory mock — one fake
account, one fake card, a handful of fake transactions. That's stated
plainly rather than left for someone to discover. Everything else — the
training data, the classifier's accuracy, the LLM's tool-calling behavior,
the eval results — is real and reproducible.

## Data source

banking77 dataset (Casanueva et al., 2020, "Efficient Intent Detection
with Dual Sentence Encoders"), via HuggingFace Datasets.

## Tests

```bash
pip install -r requirements-dev.txt     # light: no torch/transformers needed
python -m pytest -q                     # 27 tests
```

The suite covers the banking tools, the agent (with a scripted stand-in for the LLM: tool dispatch, the classifier hint,
a model that names a tool that doesn't exist, arguments of the wrong shape), the API (including rejecting empty or
oversized messages), the classifier artifact (ranking, labels, and that the committed metrics support the numbers above),
and the eval cases. The real transformer and the real Ollama model are only used by `ml/train_classifier.py` and
`eval/run_eval.py`. CI runs the suite on Python 3.11, 3.12 and 3.13.
