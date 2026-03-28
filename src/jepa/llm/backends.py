"""Commentary backends for grounded LLM explanations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from jepa.commentary import DeterministicCommentaryGenerator, LLMReadyCommentaryContext

from .schema import (
    COMMENTARY_JSON_SCHEMA,
    validate_commentary_payload,
)


class CommentaryBackend(Protocol):
    """Backend protocol for commentary generation."""

    name: str

    def available(self) -> bool: ...

    def generate(
        self,
        evidence: Mapping[str, Any],
        *,
        prompt_package: Optional[LLMReadyCommentaryContext] = None,
        expected_selected_index: int | None = None,
    ) -> Dict[str, Any]: ...


@dataclass(slots=True)
class BackendStatus:
    """Describe whether an LLM backend is available."""

    name: str
    available: bool
    reason: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "available": self.available, "reason": self.reason}


class DeterministicCommentaryBackend:
    """Always-available fallback backend built from the deterministic commentator."""

    name = "deterministic"

    def __init__(self) -> None:
        self._generator = DeterministicCommentaryGenerator()

    def available(self) -> bool:
        return True

    def generate(
        self,
        evidence: Mapping[str, Any],
        *,
        prompt_package: Optional[LLMReadyCommentaryContext] = None,
        expected_selected_index: int | None = None,
    ) -> Dict[str, Any]:
        commentary = self._generator.generate(evidence)
        return {
            "selected_candidate_index": commentary.selected_candidate_index,
            "anticipation_summary": commentary.anticipated_future_summary,
            "why_selected": commentary.why_selected,
            "uncertainty_note": commentary.uncertainty_statement,
            "baseline_note": commentary.baseline_disagreement_note or "",
        }


class OpenAIResponsesCommentaryBackend:
    """Structured-output backend using the OpenAI Responses API."""

    name = "openai_responses"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Any = None,
        max_output_tokens: int = 256,
    ) -> None:
        self.model = model or os.environ.get("JEPA_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-mini"
        self.max_output_tokens = max_output_tokens
        self._client = client
        self._status = BackendStatus(name=self.name, available=False, reason="not initialized")
        self._initialize_client()

    def _initialize_client(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._status = BackendStatus(name=self.name, available=False, reason="OPENAI_API_KEY not set")
            return

        if self._client is not None:
            self._status = BackendStatus(name=self.name, available=True)
            return

        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI()
            self._status = BackendStatus(name=self.name, available=True)
        except Exception as exc:  # pragma: no cover - import/runtime dependent
            self._client = None
            self._status = BackendStatus(name=self.name, available=False, reason=str(exc))

    def available(self) -> bool:
        return bool(self._status.available and self._client is not None)

    @property
    def status(self) -> BackendStatus:
        return self._status

    def generate(
        self,
        evidence: Mapping[str, Any],
        *,
        prompt_package: Optional[LLMReadyCommentaryContext] = None,
        expected_selected_index: int | None = None,
    ) -> Dict[str, Any]:
        if not self.available():
            raise RuntimeError(f"OpenAI commentary backend unavailable: {self._status.reason}")
        if prompt_package is None:
            raise ValueError("prompt_package is required for the OpenAI backend.")

        messages = self._build_messages(prompt_package, evidence)
        response = self._client.responses.create(  # type: ignore[union-attr]
            model=self.model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grounded_commentary",
                    "strict": True,
                    "schema": COMMENTARY_JSON_SCHEMA,
                }
            },
            max_output_tokens=self.max_output_tokens,
        )

        payload = self._extract_json_payload(response)
        normalized = validate_commentary_payload(
            payload,
            expected_selected_index=expected_selected_index,
            valid_candidate_indices=self._candidate_indices(evidence),
        )
        return normalized.as_dict()

    @staticmethod
    def _build_messages(
        prompt_package: LLMReadyCommentaryContext,
        evidence: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        return [
            {"role": "system", "content": prompt_package.system_instructions},
            {"role": "user", "content": prompt_package.user_prompt},
        ]

    @staticmethod
    def _extract_json_payload(response: Any) -> Dict[str, Any]:
        text = ""
        if hasattr(response, "output_text") and getattr(response, "output_text"):
            text = str(response.output_text)
        else:
            output = getattr(response, "output", None) or []
            pieces: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, Mapping):
                    content = item.get("content")
                content = content or []
                for part in content:
                    part_text = getattr(part, "text", None)
                    if part_text is None and isinstance(part, Mapping):
                        part_text = part.get("text")
                    if part_text:
                        pieces.append(str(part_text))
            text = "".join(pieces).strip()
        if not text:
            raise RuntimeError("OpenAI response did not contain a JSON payload.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI response was not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI response JSON must be an object.")
        return payload

    @staticmethod
    def _candidate_indices(evidence: Mapping[str, Any]) -> list[int]:
        table = list(evidence.get("candidate_score_table", []))
        indices: list[int] = []
        for row in table:
            try:
                indices.append(int(row.get("candidate_index")))
            except Exception:
                continue
        return indices


def build_default_commentary_backend() -> CommentaryBackend:
    """Prefer OpenAI when available, otherwise fall back to the deterministic backend."""

    backend = OpenAIResponsesCommentaryBackend()
    if backend.available():
        return backend
    return DeterministicCommentaryBackend()
