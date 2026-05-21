import pytest
from schemas.models import ClassificationOutput, RoutingDecision
from pipeline.router import route_ticket, CONFIDENCE_THRESHOLD
from pipeline.loader import clean_text
from pipeline.evaluator import compute_evaluation


# ── Test: Text cleaning ──
def test_clean_text_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_repeated_punctuation():
    assert clean_text("help!!!") == "help!"
    assert clean_text("what???") == "what?"


def test_clean_text_lowercase():
    assert clean_text("Hello World") == "hello world"


# ── Test: Routing threshold ──
def test_route_auto_triage():
    classification = ClassificationOutput(
        category="billing",
        urgency="high",
        confidence=0.9,
        reasoning_summary="Clear billing issue",
        needs_human_review=False,
    )
    decision = route_ticket("T1", classification)
    assert decision.route == "auto_triage"


def test_route_human_review_low_confidence():
    classification = ClassificationOutput(
        category="billing",
        urgency="high",
        confidence=0.5,
        reasoning_summary="Unclear ticket",
        needs_human_review=False,
    )
    decision = route_ticket("T1", classification)
    assert decision.route == "human_review"


def test_route_human_review_parse_failure():
    classification = ClassificationOutput(
        category="other",
        urgency="medium",
        confidence=0.0,
        reasoning_summary="Parse failure after 2 attempts: no JSON found",
        needs_human_review=True,
    )
    decision = route_ticket("T1", classification)
    assert decision.route == "human_review"


def test_route_boundary():
    """Confidence exactly at threshold should be auto_triage."""
    classification = ClassificationOutput(
        category="technical_issue",
        urgency="low",
        confidence=CONFIDENCE_THRESHOLD,
        reasoning_summary="Borderline case",
        needs_human_review=False,
    )
    decision = route_ticket("T1", classification)
    assert decision.route == "auto_triage"


# ── Test: Metrics ──
def test_evaluation_perfect():
    tickets = [
        {"ticket_id": "T1", "expected_category": "billing", "expected_urgency": "high"},
    ]
    classifications = {
        "T1": ClassificationOutput(
            category="billing", urgency="high", confidence=0.9,
            reasoning_summary="test", needs_human_review=False,
        )
    }
    routing = {
        "T1": RoutingDecision(
            ticket_id="T1", route="auto_triage", confidence=0.9,
            routing_reason="test",
        )
    }
    report = compute_evaluation(tickets, classifications, routing)
    assert report.category_accuracy == 1.0
    assert report.urgency_accuracy == 1.0