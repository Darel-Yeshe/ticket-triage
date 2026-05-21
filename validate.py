import json
import sys
from pathlib import Path


OUTPUT_DIR = "output"

REQUIRED_FILES = [
    "preprocessed_tickets.json",
    "routing_decisions.json",
    "triage_results.json",
    "prediction_comparison.json",
    "evaluation_report.json",
    "llm_calls.jsonl",
]

OPTIONAL_FILES = [
    "confusion_summary.json",
]


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate():
    errors = []
    warnings = []
    checks_passed = 0

    print("=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)

    # ── CHECK 1: Required files exist ──
    print("\n[1] Checking required files...")
    for filename in REQUIRED_FILES:
        filepath = Path(OUTPUT_DIR) / filename
        if not filepath.exists():
            errors.append(f"Missing required file: {filename}")
            print(f"  FAIL: {filename} not found")
        else:
            print(f"  OK:   {filename}")
            checks_passed += 1

    for filename in OPTIONAL_FILES:
        filepath = Path(OUTPUT_DIR) / filename
        if filepath.exists():
            print(f"  OK:   {filename} (optional)")
            checks_passed += 1
        else:
            warnings.append(f"Optional file not found: {filename}")
            print(f"  SKIP: {filename} (optional)")

    # Stop early if core files missing
    if errors:
        print_summary(checks_passed, errors, warnings)
        return len(errors) == 0

    # ── CHECK 2: JSON files are valid ──
    print("\n[2] Checking JSON validity...")
    try:
        preprocessed = load_json(Path(OUTPUT_DIR) / "preprocessed_tickets.json")
        routing = load_json(Path(OUTPUT_DIR) / "routing_decisions.json")
        triage = load_json(Path(OUTPUT_DIR) / "triage_results.json")
        comparison = load_json(Path(OUTPUT_DIR) / "prediction_comparison.json")
        evaluation = load_json(Path(OUTPUT_DIR) / "evaluation_report.json")
        llm_calls = load_jsonl(Path(OUTPUT_DIR) / "llm_calls.jsonl")
        print("  OK:   All JSON files parse successfully")
        checks_passed += 1
    except Exception as e:
        errors.append(f"JSON parse error: {str(e)}")
        print(f"  FAIL: {str(e)}")
        print_summary(checks_passed, errors, warnings)
        return False

    # ── CHECK 3: All tickets have routing decisions ──
    print("\n[3] Checking routing coverage...")
    triage_ids = {t["ticket_id"] for t in triage}
    routing_ids = {r["ticket_id"] for r in routing}

    if triage_ids != routing_ids:
        missing = triage_ids - routing_ids
        errors.append(f"Tickets missing routing decisions: {missing}")
        print(f"  FAIL: Missing routing for {missing}")
    else:
        print(f"  OK:   All {len(routing_ids)} tickets have routing decisions")
        checks_passed += 1

    # ── CHECK 4: Auto-triage tickets have customer_reply ──
    print("\n[4] Checking auto-triage replies...")
    routing_map = {r["ticket_id"]: r["route"] for r in routing}
    auto_missing_reply = []
    for t in triage:
        if routing_map.get(t["ticket_id"]) == "auto_triage" and not t.get("customer_reply"):
            auto_missing_reply.append(t["ticket_id"])

    if auto_missing_reply:
        errors.append(f"Auto-triage tickets missing reply: {auto_missing_reply}")
        print(f"  FAIL: Missing replies for {auto_missing_reply}")
    else:
        auto_count = sum(1 for r in routing if r["route"] == "auto_triage")
        print(f"  OK:   All {auto_count} auto-triage tickets have replies")
        checks_passed += 1

    # ── CHECK 5: Human-review tickets have internal_note ──
    print("\n[5] Checking human-review notes...")
    human_missing_note = []
    for t in triage:
        if routing_map.get(t["ticket_id"]) == "human_review" and not t.get("internal_note"):
            human_missing_note.append(t["ticket_id"])

    if human_missing_note:
        errors.append(f"Human-review tickets missing internal note: {human_missing_note}")
        print(f"  FAIL: Missing notes for {human_missing_note}")
    else:
        human_count = sum(1 for r in routing if r["route"] == "human_review")
        print(f"  OK:   All {human_count} human-review tickets have notes")
        checks_passed += 1

    # ── CHECK 6: Labels belong to allowed schema ──
    print("\n[6] Checking label validity...")
    schema = load_json("data/label_schema.json")
    allowed_categories = schema["categories"]
    allowed_urgency = schema["urgency_levels"]
    invalid_labels = []

    for t in triage:
        if t["predicted_category"] not in allowed_categories:
            invalid_labels.append(f"{t['ticket_id']}: bad category '{t['predicted_category']}'")
        if t["predicted_urgency"] not in allowed_urgency:
            invalid_labels.append(f"{t['ticket_id']}: bad urgency '{t['predicted_urgency']}'")

    if invalid_labels:
        errors.append(f"Invalid labels found: {invalid_labels}")
        for label in invalid_labels:
            print(f"  FAIL: {label}")
    else:
        print(f"  OK:   All predicted labels are valid")
        checks_passed += 1

    # ── CHECK 7: Evaluation metrics exist ──
    print("\n[7] Checking evaluation metrics...")
    required_metrics = ["total_tickets", "category_accuracy", "urgency_accuracy", "human_review_count", "auto_triage_count", "parse_failures"]
    missing_metrics = [m for m in required_metrics if m not in evaluation]

    if missing_metrics:
        errors.append(f"Missing evaluation metrics: {missing_metrics}")
        print(f"  FAIL: Missing {missing_metrics}")
    else:
        print(f"  OK:   All metrics present")
        print(f"         Category accuracy: {evaluation['category_accuracy']}")
        print(f"         Urgency accuracy:  {evaluation['urgency_accuracy']}")
        checks_passed += 1

    # ── SUMMARY ──
    print_summary(checks_passed, errors, warnings)
    return len(errors) == 0


def print_summary(passed, errors, warnings):
    print("\n" + "=" * 50)
    print(f"CHECKS PASSED: {passed}")
    print(f"ERRORS:        {len(errors)}")
    print(f"WARNINGS:      {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if not errors:
        print("\n*** VALIDATION PASSED ***")
    else:
        print("\n*** VALIDATION FAILED ***")
    print("=" * 50)


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)