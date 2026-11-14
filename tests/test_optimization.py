"""Assignment acceptance checks; the original baseline tests stay unchanged."""
from __future__ import annotations

import json

import pytest

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline
from shared.kb import article_ids

PUBLIC = ("T-1001", "T-1002", "T-1003", "T-1004")


@pytest.mark.parametrize("agent_type,limit,label", [
    (SupportCopilot, 3000, "handle_ticket"),
    (SupportPipeline, 2500, "compose"),
])
@pytest.mark.parametrize("ticket_id", (*PUBLIC, "T-1091"))
def test_cold_ticket_correctness_and_budget(agent_type, limit, label, ticket_id):
    agent = agent_type()
    result = agent.handle_ticket(ticket_id)
    expected = TICKET_EXPECTATIONS[ticket_id]
    assert (result.action, result.kb_article) == (expected.action, expected.kb_article)
    assert [call.label for call in agent.session.calls] == [label]
    assert agent.session.calls[0].cached_prefix_tokens == 0
    if ticket_id in PUBLIC:
        assert agent.session.total_points() <= limit


@pytest.mark.parametrize("agent_type,limit", [
    (SupportCopilot, 3000), (SupportPipeline, 2500),
])
def test_reused_agent_keeps_each_ticket_under_budget(agent_type, limit):
    agent = agent_type()
    for ticket_id in PUBLIC:
        previous_points = agent.session.total_points()
        agent.handle_ticket(ticket_id)
        assert agent.session.total_points() - previous_points <= limit


def test_unknown_product_retains_all_policies():
    assert set(SupportPipeline()._triage({"product_type": "new-product"})) == set(article_ids())


def test_customized_physical_item_retrieves_the_custom_exception():
    pipeline = SupportPipeline()
    candidates = pipeline._triage({"product_type": "physical", "customized": True})
    assert "custom-items-no-replace" in pipeline._retrieve(candidates)


@pytest.mark.parametrize("draft", [
    "invalid JSON",
    json.dumps({"action": "DELETE_ORDER", "kb_article": "returns-within-30-days"}),
    json.dumps({"action": "REFUND", "kb_article": "invented-policy"}),
])
def test_audit_rejects_invalid_output(draft):
    pipeline = SupportPipeline()
    articles = pipeline._retrieve(("returns-within-30-days",))
    with pytest.raises(ValueError):
        pipeline._audit(draft, articles)
    assert not pipeline.session.calls
