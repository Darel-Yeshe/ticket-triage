import os
from groq import Groq
from pipeline.logger import LLMLogger

MODEL_NAME = "llama-3.3-70b-versatile"


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_reply_prompt(ticket_text: str, category: str, urgency: str) -> str:
    return f"""You are a customer support agent. Write a short draft reply (2-4 sentences) to the following ticket.

Rules:
- Acknowledge the customer's issue
- Do NOT make up account-specific facts (names, balances, dates)
- Do NOT promise actions not stated in the ticket
- Be consistent with the category ({category}) and urgency ({urgency})
- Be professional and empathetic

TICKET:
\"{ticket_text}\"

Reply:"""


def generate_reply(
    ticket_id: str,
    ticket_text: str,
    category: str,
    urgency: str,
    logger: LLMLogger,
    output_dir: str,
) -> str:
    client = get_client()
    prompt = build_reply_prompt(ticket_text, category, urgency)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )
        reply = response.choices[0].message.content.strip()

        logger.log_call(
            stage="reply_generation",
            ticket_id=ticket_id,
            prompt=prompt,
            model=MODEL_NAME,
            output_artifact=f"{output_dir}/triage_results.json",
        )

        return reply

    except Exception as e:
        return f"[Error generating reply: {str(e)}]"


def generate_internal_note(ticket_id: str, routing_reason: str, category: str, confidence: float) -> str:
    return (
        f"Ticket {ticket_id} escalated to human review. "
        f"Predicted category: {category}, confidence: {confidence}. "
        f"Reason: {routing_reason}"
    )