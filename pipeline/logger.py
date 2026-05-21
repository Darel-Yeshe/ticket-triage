import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from schemas.models import LLMCallLog


class LLMLogger:
    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Clear file at start
        open(output_path, "w").close()

    def log_call(self, stage: str, ticket_id: str, prompt: str, model: str, output_artifact: str):
        entry = LLMCallLog(
            stage=stage,
            ticket_id=ticket_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider="google",
            model=model,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            output_artifact=output_artifact,
        )
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")