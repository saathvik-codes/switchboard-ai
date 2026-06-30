"""
Banking tools the agent can call. Backed by an in-memory mock data store -
this is a portfolio demo, not a connection to a real bank, and that's
stated here rather than left for someone to discover. The tools themselves
(balance lookup, card freeze, transaction history, escalation) are the
real, standard shape of what a production banking chatbot's tool layer
looks like; only the backend behind them is simulated.
"""
from langchain_core.tools import tool

_MOCK_ACCOUNTS = {
    "ACC-1001": {"balance": 1250.40, "currency": "GBP", "owner": "default"},
}

_MOCK_CARDS = {
    "CARD-4521": {"status": "active", "account": "ACC-1001"},
}

_MOCK_TRANSACTIONS = [
    {"date": "2026-06-28", "amount": -42.50, "merchant": "Tesco"},
    {"date": "2026-06-27", "amount": -12.00, "merchant": "TfL"},
    {"date": "2026-06-25", "amount": 2500.00, "merchant": "Salary"},
    {"date": "2026-06-20", "amount": -89.99, "merchant": "Amazon"},
]


@tool
def check_balance(account_id: str = "ACC-1001") -> str:
    """Check the current balance of a bank account given its account ID."""
    account = _MOCK_ACCOUNTS.get(account_id)
    if not account:
        return f"No account found with ID {account_id}."
    return f"Account {account_id} balance: {account['currency']} {account['balance']:,.2f}"


@tool
def freeze_card(card_id: str = "CARD-4521") -> str:
    """Freeze a card given its card ID - used when a card is reported lost or stolen."""
    card = _MOCK_CARDS.get(card_id)
    if not card:
        return f"No card found with ID {card_id}."
    card["status"] = "frozen"
    return f"Card {card_id} has been frozen. A replacement will be issued within 5-7 business days."


@tool
def transaction_history(account_id: str = "ACC-1001", limit: int = 5) -> str:
    """Look up recent transactions for an account."""
    rows = _MOCK_TRANSACTIONS[:limit]
    lines = [f"{t['date']}: {t['merchant']} {t['amount']:+.2f}" for t in rows]
    return "\n".join(lines)


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent when the request needs
    judgment the automated system shouldn't make alone (disputes, fraud
    claims, complaints, anything involving discretion over money)."""
    return f"Escalated to a human agent. Reason logged: {reason}"


ALL_TOOLS = [check_balance, freeze_card, transaction_history, escalate_to_human]
