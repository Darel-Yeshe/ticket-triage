import json
import re
import os
from groq import Groq
from schemas.models import ClassificationOutput
from pipeline.logger import LLMLogger


MODEL_NAME = "llama-3.3-70b-versatile"


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_classification_prompt(ticket_text: str, categories: list, urgency_levels: list) -> str:
    return f"""You are a customer support ticket classifier.

Classify the following ticket into one category and one urgency level.

ALLOWED CATEGORIES: {json.dumps(categories)}
ALLOWED URGENCY LEVELS: {json.dumps(urgency_levels)}

TICKET:
\"{ticket_text}\"

Respond with ONLY a valid JSON object in this exact format, nothing else:
{{
  "category": "one of the allowed categories",
  "urgency": "one of the allowed urgency levels",
  "confidence": 0.0 to 1.0,
  "reasoning_summary": "short explanation",
  "needs_human_review": false
}}"""


def extract_json_from_text(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No valid JSON found in response")


def validate_classification(raw: dict, categories: list, urgency_levels: list) -> ClassificationOutput:
    if raw.get("category") not in categories:
        raise ValueError(f"Invalid category: {raw.get('category')}")
    if raw.get("urgency") not in urgency_levels:
        raise ValueError(f"Invalid urgency: {raw.get('urgency')}")
    return ClassificationOutput(**raw)


def classify_ticket(
    ticket_id: str,
    ticket_text: str,
    categories: list,
    urgency_levels: list,
    logger: LLMLogger,
    output_dir: str,
    max_retries: int = 2,
) -> ClassificationOutput:

    client = get_client()
    prompt = build_classification_prompt(ticket_text, categories, urgency_levels)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            raw_text = response.choices[0].message.content

            logger.log_call(
                stage="classification",
                ticket_id=ticket_id,
                prompt=prompt,
                model=MODEL_NAME,
                output_artifact=f"{output_dir}/triage_results.json",
            )

            raw_dict = extract_json_from_text(raw_text)
            result = validate_classification(raw_dict, categories, urgency_levels)
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                prompt = build_classification_prompt(ticket_text, categories, urgency_levels)
                prompt += "\n\nIMPORTANT: Return ONLY raw JSON. No markdown, no explanation, no code fences."
            else:
                return ClassificationOutput(
                    category="other",
                    urgency="medium",
                    confidence=0.0,
                    reasoning_summary=f"Parse failure after {max_retries} attempts: {str(e)}",
                    needs_human_review=True,
                )