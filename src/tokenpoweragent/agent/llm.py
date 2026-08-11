"""Minimal OpenAI-compatible completion client for the bounded planner."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class PlannerClientError(RuntimeError):
    """Raised when an external planner endpoint cannot return usable text."""


@dataclass(frozen=True)
class PlannerCompletionResult:
    """Text and reproducibility metadata returned by a planner endpoint."""

    text: str
    model: Optional[str] = None
    response_id: Optional[str] = None
    system_fingerprint: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class OpenAICompatibleCompletion:
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(self, prompt: str) -> PlannerCompletionResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the requested JSON schema exactly. Do not "
                        "select or execute concrete system actions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PlannerClientError("planner request failed: %s" % exc) from exc
        try:
            body = json.loads(raw)
            content = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise PlannerClientError("planner response schema is invalid") from exc
        if not isinstance(content, str) or not content.strip():
            raise PlannerClientError("planner response content is empty")
        usage = body.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return PlannerCompletionResult(
            text=content,
            model=_optional_text(body.get("model")),
            response_id=_optional_text(body.get("id")),
            system_fingerprint=_optional_text(body.get("system_fingerprint")),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers


def _optional_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
