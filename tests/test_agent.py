"""The agent with a scripted stand-in for the LLM, so tool dispatch, the classifier hint and error handling are tested
without Ollama or the transformer model."""
import pytest
from langchain_core.messages import AIMessage

import agent
import tools


class FakeLLM:
    """bind_tools() returns itself; invoke() pops the next scripted reply and records what it was asked."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def call(name, args, id="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


@pytest.fixture(autouse=True)
def fixed_intent(monkeypatch):
    monkeypatch.setattr(agent, "classify_intent", lambda message, top_k=3: [
        {"intent": "lost_or_stolen_card", "confidence": 0.91}, {"intent": "card_arrival", "confidence": 0.04},
        {"intent": "card_linking", "confidence": 0.01}])


def test_tool_call_runs_the_tool_and_summarises_the_result():
    llm = FakeLLM(call("freeze_card", {"card_id": "CARD-4521"}), AIMessage(content="Your card is frozen."))
    out = agent.handle_message("my card was stolen", llm=llm)
    assert out["tool_calls"] == [{"tool": "freeze_card", "args": {"card_id": "CARD-4521"},
                                  "result": "Card CARD-4521 has been frozen. A replacement will be issued within 5-7 business days."}]
    assert out["response"] == "Your card is frozen."
    assert tools._MOCK_CARDS["CARD-4521"]["status"] == "frozen"


def test_classifier_prediction_is_given_to_the_model_as_a_hint():
    llm = FakeLLM(AIMessage(content="ok"))
    agent.handle_message("my card was stolen", llm=llm)
    system_prompt = llm.seen[0][0].content
    assert "lost_or_stolen_card" in system_prompt and "91%" in system_prompt


def test_no_tool_call_returns_the_model_text_directly():
    llm = FakeLLM(AIMessage(content="Fees depend on your plan."))
    out = agent.handle_message("what are your fees", llm=llm)
    assert out["tool_calls"] == [] and out["response"] == "Fees depend on your plan."
    assert out["classified_intent"] == "lost_or_stolen_card" and len(out["all_intents"]) == 3
    assert len(llm.seen) == 1                                  # no second summarising call


def test_a_tool_the_model_invented_becomes_an_error_message_not_a_crash():
    llm = FakeLLM(call("wire_money", {"amount": 5}), AIMessage(content="Sorry, I can't do that."))
    out = agent.handle_message("send 5 pounds", llm=llm)
    assert "no tool called 'wire_money'" in out["tool_calls"][0]["result"]
    assert out["response"] == "Sorry, I can't do that."


def test_arguments_of_the_wrong_shape_become_an_error_message():
    llm = FakeLLM(call("transaction_history", {"limit": "many"}), AIMessage(content="Could you say how many?"))
    out = agent.handle_message("show transactions", llm=llm)
    assert out["tool_calls"][0]["result"].startswith("Error: the transaction_history tool could not run")


def test_several_tool_calls_in_one_reply_all_run():
    two = AIMessage(content="", tool_calls=[{"name": "check_balance", "args": {}, "id": "1"},
                                            {"name": "transaction_history", "args": {"limit": 1}, "id": "2"}])
    out = agent.handle_message("balance and last transaction", llm=FakeLLM(two, AIMessage(content="Done.")))
    assert [t["tool"] for t in out["tool_calls"]] == ["check_balance", "transaction_history"]


def test_importing_the_agent_leaves_stdout_alone():
    """On Windows the module used to wrap sys.stdout at import, which breaks anything that imports it (pytest, a server)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(agent.__file__).resolve().parents[1]
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']; real = sys.stdout; import agent; "
            "sys.exit(0 if sys.stdout is real else 1)") % (root / "agent", root / "ml")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
