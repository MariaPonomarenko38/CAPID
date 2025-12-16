from __future__ import annotations
import time
from typing import Callable, Optional

class LLM:
    def __init__(self, caller: Callable[..., str | BaseModel], max_retries: int = 3, backoff: float = 1.5):
        self.caller = caller
        self.max_retries = max_retries
        self.backoff = backoff

    def ask(self, prompt: str, model="gpt-4.1-mini", schema: Optional[type] = None) -> str:
        delay = 1.0
        last_err: Optional[Exception] = None
        for _ in range(1, self.max_retries + 1):
            try:
                resp = self.caller(prompt, model, schema=schema) if schema else self.caller(prompt, model)
                if isinstance(resp, str):
                    return resp.strip()
                return str(resp).strip()
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= self.backoff
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_err