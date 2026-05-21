import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from pipeline.loader import load_tickets, load_label_schema, preprocess_tickets, save_preprocessed
from pipeline.classifier import classify_ticket
from pipeline.router import route_ticket
from pipeline.responder import generate_reply, generate_internal_note
from pipeline.evaluator import compute_evaluation, build_comparisons, build_confusion_summary
from pipeline.logger import LLMLogger
from schemas.models import TriageResult


def save_json(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_pipeline(tickets_path: str, schema_path: str, output_dir: str):
    # ── INIT ──
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    logger = LLMLogger(f"{output_dir}/llm_calls.jsonl")

    print("=" * 50)
    print("TICKET TRIAGE PIPELINE")
    print("=" * 50)

    # ── STAGE 1: INPUTS_LOADED + TEXT_PREPROCESSED ──
    print("\n[1/7] Loading inputs...")
    tickets = load_tickets(tickets_path)
    schema = load_label_schema(schema_path)
    categories = schema["categories"]
    urgency_levels = schema["urgency_levels"]
    print(f"  Loaded {len(tickets)} tickets, {len(categories)} categories, {len(urgency_levels)} urgency levels")

    print("\n[2/7] Preprocessing tickets...")
    preprocessed = preprocess_tickets(tickets)
    save_preprocessed(preprocessed, f"{output_dir}/preprocessed_tickets.json")
    print(f"  Saved preprocessed_tickets.json")

    # ── STAGE 2-3: MODEL_PROMPTED + STRUCTURED_OUTPUT_PARSED ──
    print("\n[3/7] Classifying tickets via Groq...")
    classifications = {}
    for ticket in tickets:
        tid = ticket["ticket_id"]
        print(f"  Classifying {tid}...")
        result = classify_ticket(
            ticket_id=tid,
            ticket_text=ticket["customer_message"],
            categories=categories,
            urgency_levels=urgency_levels,
            logger=logger,
            output_dir=output_dir,
        )
        classifications[tid] = result
        print(f"    -> {result.category} | {result.urgency} | conf: {result.confidence}")

    # ── STAGE 4: CONFIDENCE_CHECKED + ROUTED ──
    print("\n[4/7] Routing tickets...")
    routing_decisions = {}
    for tid, classification in classifications.items():
        decision = route_ticket(tid, classification)
        routing_decisions[tid] = decision
        print(f"  {tid} -> {decision.route} ({decision.routing_reason})")

    save_json(
        [d.model_dump() for d in routing_decisions.values()],
        f"{output_dir}/routing_decisions.json",
    )
    print(f"  Saved routing_decisions.json")

    # ── STAGE 5: RESPONSE_GENERATED ──
    print("\n[5/7] Generating responses...")
    triage_results = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        classification = classifications[tid]
        routing = routing_decisions[tid]

        customer_reply = None
        internal_note = None

        if routing.route == "auto_triage":
            print(f"  {tid}: generating customer reply...")
            customer_reply = generate_reply(
                ticket_id=tid,
                ticket_text=ticket["customer_message"],
                category=classification.category,
                urgency=classification.urgency,
                logger=logger,
                output_dir=output_dir,
            )
        else:
            print(f"  {tid}: generating internal note...")
            internal_note = generate_internal_note(
                ticket_id=tid,
                routing_reason=routing.routing_reason,
                category=classification.category,
                confidence=classification.confidence,
            )

        triage_results.append(
            TriageResult(
                ticket_id=tid,
                predicted_category=classification.category,
                predicted_urgency=classification.urgency,
                confidence=classification.confidence,
                route=routing.route,
                customer_reply=customer_reply,
                internal_note=internal_note,
            )
        )

    save_json(
        [r.model_dump() for r in triage_results],
        f"{output_dir}/triage_results.json",
    )
    print(f"  Saved triage_results.json")

    # ── STAGE 6: EVALUATION_COMPUTED ──
    print("\n[6/7] Computing evaluation metrics...")
    report = compute_evaluation(tickets, classifications, routing_decisions)
    save_json(report.model_dump(), f"{output_dir}/evaluation_report.json")
    print(f"  Category accuracy: {report.category_accuracy}")
    print(f"  Urgency accuracy:  {report.urgency_accuracy}")
    print(f"  Auto-triaged:      {report.auto_triage_count}")
    print(f"  Human review:      {report.human_review_count}")
    print(f"  Parse failures:    {report.parse_failures}")

    comparisons = build_comparisons(tickets, classifications, routing_decisions)
    save_json(
        [c.model_dump() for c in comparisons],
        f"{output_dir}/prediction_comparison.json",
    )
    print(f"  Saved prediction_comparison.json")

    confusion = build_confusion_summary(comparisons)
    save_json(confusion, f"{output_dir}/confusion_summary.json")
    print(f"  Saved confusion_summary.json")

    # ── STAGE 7: VALIDATION_COMPLETED ──
    print("\n[7/7] Pipeline complete!")
    print("=" * 50)
    print(f"All artifacts saved to: {output_dir}/")
    print("Run 'python validate.py' to verify outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket Triage Pipeline")
    parser.add_argument("--tickets", default="data/tickets.json", help="Path to tickets JSON file")
    parser.add_argument("--schema", default="data/label_schema.json", help="Path to label schema JSON file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.tickets, args.schema, args.output)