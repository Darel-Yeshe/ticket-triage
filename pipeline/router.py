from schemas.models import ClassificationOutput, RoutingDecision

CONFIDENCE_THRESHOLD = 0.65


def route_ticket(ticket_id: str, classification: ClassificationOutput) -> RoutingDecision:
    """Deterministic routing — never relies on LLM's needs_human_review alone."""

    # Check for parse failure (confidence 0.0 from fallback)
    if classification.confidence == 0.0 and "Parse failure" in classification.reasoning_summary:
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=classification.confidence,
            routing_reason="LLM output was invalid or unparsable",
        )

    # Confidence threshold check
    if classification.confidence < CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=classification.confidence,
            routing_reason=f"Confidence {classification.confidence} is below threshold {CONFIDENCE_THRESHOLD}",
        )

    # Passed all checks
    return RoutingDecision(
        ticket_id=ticket_id,
        route="auto_triage",
        confidence=classification.confidence,
        routing_reason=f"Confidence {classification.confidence} meets threshold",
    )