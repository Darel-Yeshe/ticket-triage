from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List


class PreprocessedTicket(BaseModel):
    ticket_id: str
    original_text: str
    cleaned_text: str
    char_count: int
    word_count: int


class ClassificationOutput(BaseModel):
    category: str
    urgency: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    needs_human_review: bool

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v):
        return round(v, 4)


class RoutingDecision(BaseModel):
    ticket_id: str
    route: Literal["auto_triage", "human_review"]
    confidence: float
    routing_reason: str


class TriageResult(BaseModel):
    ticket_id: str
    predicted_category: str
    predicted_urgency: str
    confidence: float
    route: str
    customer_reply: Optional[str] = None
    internal_note: Optional[str] = None


class LLMCallLog(BaseModel):
    stage: Literal["classification", "reply_generation"]
    ticket_id: str
    timestamp: str
    provider: str
    model: str
    prompt_hash: str
    output_artifact: str


class EvaluationReport(BaseModel):
    total_tickets: int
    category_accuracy: float
    urgency_accuracy: float
    human_review_count: int
    auto_triage_count: int
    parse_failures: int


class PredictionComparison(BaseModel):
    ticket_id: str
    expected_category: str
    predicted_category: str
    category_match: bool
    expected_urgency: str
    predicted_urgency: str
    urgency_match: bool
    confidence: float
    route: str