"""
The orchestrator: real ML intent classification (ml/predict.py) hands the
LLM a hint about what the customer wants, then a LangChain tool-calling
agent (Ollama, local, no API key) decides which banking tool to invoke and
extracts the parameters from the message itself.

Why combine both rather than picking one: intent classification alone
can't extract "card 4521" from "my card was stolen, the number is 4521" or
hold a conversation - that needs an LLM. An LLM alone, asked to freely
guess which of 77 fine-grained banking intents a message matches, is
noticeably less reliable than a classifier trained on 10,000 real labeled
examples (see ml/train_classifier.py - 93% accuracy). Giving the LLM the
classifier's top intent as a hint, then letting it handle the part it's
actually good at (parameter extraction, tool selection, natural response),
plays to each component's strength.
"""
import io
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from predict import classify_intent  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

SYSTEM_PROMPT = """You are Switchboard, a banking customer-service agent.

You have tools for: checking account balance, freezing a lost/stolen card,
looking up recent transactions, and escalating to a human agent.

The customer's message has been classified by a trained intent model as
likely about: "{intent}" (confidence: {confidence:.0%}). Use this as a
hint, not a hard rule - if the message clearly needs a different tool,
trust the message over the hint.

Call a tool ONLY when the customer is asking you to act on or look up
THEIR OWN account (their balance, their transactions, freezing their own
card, a dispute about a charge they were actually billed). Use sensible
defaults like ACC-1001 or CARD-4521 if they don't give a specific ID -
this is a demo environment with one mock account/card.

Do NOT call a tool for general policy/product questions that aren't about
this customer's own account or money - questions like "what are your
fees", "how long does a replacement card take", "which currencies do you
support". For those, answer briefly from general knowledge of how banks
typically work, or say you don't have that specific policy detail - do
not check a balance, freeze a card, or escalate just because a tool is
available. Calling a tool when none is needed is a worse outcome than
giving a short direct answer.
"""


TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call from the model. A model can name a tool that doesn't exist or pass arguments of the wrong
    shape; either becomes a message the model can read and recover from, never an exception that fails the request."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"Error: there is no tool called '{name}'. Available tools: {', '.join(TOOLS_BY_NAME)}."
    try:
        return tool.invoke(args)
    except Exception as e:
        return f"Error: the {name} tool could not run with those arguments ({type(e).__name__})."


def handle_message(message: str, llm=None) -> dict:
    intent_predictions = classify_intent(message, top_k=3)
    top_intent = intent_predictions[0]

    chat = llm or ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0,
                             base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    chat_with_tools = chat.bind_tools(ALL_TOOLS)

    system = SystemMessage(content=SYSTEM_PROMPT.format(intent=top_intent["intent"], confidence=top_intent["confidence"]))
    response = chat_with_tools.invoke([system, HumanMessage(content=message)])

    tool_results = []
    for call in response.tool_calls:
        tool_results.append({"tool": call["name"], "args": call["args"], "result": run_tool(call["name"], call["args"])})

    if tool_results:
        followup = chat.invoke([
            system,
            HumanMessage(content=message),
            response,
            *[HumanMessage(content=f"Tool '{r['tool']}' returned: {r['result']}") for r in tool_results],
            HumanMessage(content="Summarize this for the customer in one or two sentences."),
        ])
        final_text = followup.content
    else:
        final_text = response.content

    return {
        "message": message,
        "classified_intent": top_intent["intent"],
        "intent_confidence": top_intent["confidence"],
        "all_intents": intent_predictions,
        "tool_calls": tool_results,
        "response": final_text,
    }


if __name__ == "__main__":
    if sys.platform == "win32":     # only for this demo's printed output; never at import time
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    for msg in [
        "my card was stolen, the number is CARD-4521",
        "what's my balance?",
        "show me my last few transactions",
        "I was charged twice for the same order, I want to dispute this",
        "why is your exchange rate worse than my old bank",
    ]:
        result = handle_message(msg)
        print(f"\n> {msg}")
        print(f"  intent: {result['classified_intent']} ({result['intent_confidence']:.0%})")
        print(f"  tool calls: {[t['tool'] for t in result['tool_calls']]}")
        print(f"  response: {result['response']}")
