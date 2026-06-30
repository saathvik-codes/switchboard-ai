"""
Evaluates the agent's tool-selection decisions against a curated set of
real-world-phrased messages, run end-to-end (real intent classifier +
real LLM tool-calling, no mocking) - the same honesty standard as Pulse,
RouteForge, and the RAG assistant: report what actually happened, not
what was hoped for.

Run: python eval/run_eval.py
"""
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
from agent import handle_message  # noqa: E402

CASES_PATH = Path(__file__).parent / "eval_cases.json"


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())
    passed = 0
    for case in cases:
        result = handle_message(case["message"])
        called_tools = [t["tool"] for t in result["tool_calls"]]
        expected = case["expected_tool"]

        if expected is None:
            ok = len(called_tools) == 0
        else:
            ok = called_tools == [expected]

        status = "PASS" if ok else "FAIL"
        passed += ok
        print(f"[{status}] {case['id']}: expected={expected!r} got={called_tools!r} (intent={result['classified_intent']})")

    print(f"\n{passed}/{len(cases)} passed ({passed/len(cases):.0%})")


if __name__ == "__main__":
    main()
