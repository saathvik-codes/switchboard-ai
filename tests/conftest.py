import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ml", "agent", "api"):                 # the modules import each other by bare name (see agent.py, api/main.py)
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def fresh_bank():
    """Tools mutate an in-memory bank (freeze_card); restore it so tests don't depend on order."""
    import tools
    snapshot = (copy.deepcopy(tools._MOCK_ACCOUNTS), copy.deepcopy(tools._MOCK_CARDS), copy.deepcopy(tools._MOCK_TRANSACTIONS))
    yield
    tools._MOCK_ACCOUNTS.clear(); tools._MOCK_ACCOUNTS.update(snapshot[0])
    tools._MOCK_CARDS.clear(); tools._MOCK_CARDS.update(snapshot[1])
    tools._MOCK_TRANSACTIONS[:] = snapshot[2]
