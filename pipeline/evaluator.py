from schemas.models import EvaluationReport, PredictionComparison


def compute_evaluation(tickets: list, classifications: dict, routing_decisions: dict) -> EvaluationReport:
    """Compute accuracy metrics by comparing predictions to expected labels."""
    total = len(tickets)
    category_correct = 0
    urgency_correct = 0
    human_review_count = 0
    auto_triage_count = 0
    parse_failures = 0

    for ticket in tickets:
        tid = ticket["ticket_id"]
        classification = classifications.get(tid)
        routing = routing_decisions.get(tid)

        if classification is None:
            parse_failures += 1
            continue

        if classification.confidence == 0.0 and "Parse failure" in classification.reasoning_summary:
            parse_failures += 1

        if classification.category == ticket["expected_category"]:
            category_correct += 1

        if classification.urgency == ticket["expected_urgency"]:
            urgency_correct += 1

        if routing and routing.route == "human_review":
            human_review_count += 1
        else:
            auto_triage_count += 1

    return EvaluationReport(
        total_tickets=total,
        category_accuracy=round(category_correct / total, 4) if total > 0 else 0.0,
        urgency_accuracy=round(urgency_correct / total, 4) if total > 0 else 0.0,
        human_review_count=human_review_count,
        auto_triage_count=auto_triage_count,
        parse_failures=parse_failures,
    )


def build_comparisons(tickets: list, classifications: dict, routing_decisions: dict) -> list:
    """Build per-ticket comparison report."""
    comparisons = []

    for ticket in tickets:
        tid = ticket["ticket_id"]
        classification = classifications.get(tid)
        routing = routing_decisions.get(tid)

        if classification is None:
            comparisons.append(
                PredictionComparison(
                    ticket_id=tid,
                    expected_category=ticket["expected_category"],
                    predicted_category="PARSE_FAILURE",
                    category_match=False,
                    expected_urgency=ticket["expected_urgency"],
                    predicted_urgency="PARSE_FAILURE",
                    urgency_match=False,
                    confidence=0.0,
                    route="human_review",
                )
            )
            continue

        comparisons.append(
            PredictionComparison(
                ticket_id=tid,
                expected_category=ticket["expected_category"],
                predicted_category=classification.category,
                category_match=classification.category == ticket["expected_category"],
                expected_urgency=ticket["expected_urgency"],
                predicted_urgency=classification.urgency,
                urgency_match=classification.urgency == ticket["expected_urgency"],
                confidence=classification.confidence,
                route=routing.route if routing else "human_review",
            )
        )

    return comparisons


def build_confusion_summary(comparisons: list) -> dict:
    """Build a simple confusion summary showing which categories get mixed up."""
    confusion = {}

    for comp in comparisons:
        if not comp.category_match:
            key = f"{comp.expected_category} -> {comp.predicted_category}"
            confusion[key] = confusion.get(key, 0) + 1

    return {
        "total_mismatches": sum(confusion.values()),
        "confusions": confusion,
    }