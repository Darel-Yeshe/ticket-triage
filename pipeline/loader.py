import json
import re
from pathlib import Path
from schemas.models import PreprocessedTicket


def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tickets(filepath: str) -> list:
    return load_json(filepath)


def load_label_schema(filepath: str) -> dict:
    schema = load_json(filepath)
    if "categories" not in schema or "urgency_levels" not in schema:
        raise ValueError("label_schema.json must contain 'categories' and 'urgency_levels'")
    return schema


def clean_text(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize repeated punctuation (e.g. "!!!" -> "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    # Lowercase
    text = text.lower()
    return text


def preprocess_tickets(tickets: list) -> list:
    preprocessed = []
    for ticket in tickets:
        original = ticket["customer_message"]
        cleaned = clean_text(original)
        preprocessed.append(
            PreprocessedTicket(
                ticket_id=ticket["ticket_id"],
                original_text=original,
                cleaned_text=cleaned,
                char_count=len(cleaned),
                word_count=len(cleaned.split()),
            )
        )
    return preprocessed


def save_preprocessed(preprocessed: list, output_path: str):
    data = [p.model_dump() for p in preprocessed]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)