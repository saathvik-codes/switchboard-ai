import tools


def test_balance_is_formatted_with_currency_and_thousands_separator():
    assert tools.check_balance.invoke({"account_id": "ACC-1001"}) == "Account ACC-1001 balance: GBP 1,250.40"


def test_unknown_account_is_reported_not_raised():
    assert "No account found" in tools.check_balance.invoke({"account_id": "ACC-0000"})


def test_freezing_a_card_changes_its_status():
    assert tools._MOCK_CARDS["CARD-4521"]["status"] == "active"
    assert "frozen" in tools.freeze_card.invoke({"card_id": "CARD-4521"})
    assert tools._MOCK_CARDS["CARD-4521"]["status"] == "frozen"


def test_freezing_an_unknown_card_changes_nothing():
    assert "No card found" in tools.freeze_card.invoke({"card_id": "CARD-0000"})
    assert tools._MOCK_CARDS["CARD-4521"]["status"] == "active"


def test_transaction_history_respects_the_limit_and_signs_amounts():
    lines = tools.transaction_history.invoke({"limit": 2}).splitlines()
    assert lines == ["2026-06-28: Tesco -42.50", "2026-06-27: TfL -12.00"]
    assert "+2500.00" in tools.transaction_history.invoke({"limit": 4})


def test_escalation_logs_the_reason():
    assert "double charge" in tools.escalate_to_human.invoke({"reason": "double charge"})


def test_every_tool_has_a_description_the_model_can_read():
    assert {t.name for t in tools.ALL_TOOLS} == {"check_balance", "freeze_card", "transaction_history", "escalate_to_human"}
    assert all(t.description for t in tools.ALL_TOOLS)
