"""Commentary backends for grounded LLM explanations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

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


def _first_nonempty(values: Sequence[Optional[str]]) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _read_colab_secret(*names: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None

    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            continue
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_gemini_api_key() -> Optional[str]:
    value = _first_nonempty(
        [
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("JEPA_GEMINI_API_KEY"),
        ]
    )
    if value:
        os.environ.setdefault("GEMINI_API_KEY", value)
        return value

    value = _read_colab_secret(
        "google-api-key",
        "google_api_key",
        "gemini-api-key",
        "gemini_api_key",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    )
    if value:
        os.environ.setdefault("GEMINI_API_KEY", value)
        os.environ.setdefault("GOOGLE_API_KEY", value)
    return value


class GeminiCommentaryBackend:
    """Structured-output backend using the Gemini API via the Google GenAI SDK."""

    name = "gemini"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Any = None,
        max_output_tokens: int = 256,
    ) -> None:
        self.model = (
            model
            or os.environ.get("JEPA_LLM_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-2.5-flash"
        )
        self.max_output_tokens = max_output_tokens
        self._client = client
        self._types = None
        self._status = BackendStatus(name=self.name, available=False, reason="not initialized")
        self._initialize_client()

    def _initialize_client(self) -> None:
        api_key = _resolve_gemini_api_key()
        if not api_key and self._client is None:
            self._status = BackendStatus(
                name=self.name,
                available=False,
                reason="GEMINI_API_KEY / GOOGLE_API_KEY not set and no Colab secret was found",
            )
            return

        if self._client is not None:
            self._status = BackendStatus(name=self.name, available=True)
            return

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            self._client = genai.Client(api_key=api_key)
            self._types = types
            self._status = BackendStatus(name=self.name, available=True)
        except Exception as exc:  # pragma: no cover - import/runtime dependent
            self._client = None
            self._types = None
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
            raise RuntimeError(f"Gemini commentary backend unavailable: {self._status.reason}")
        if prompt_package is None:
            raise ValueError("prompt_package is required for the Gemini backend.")

        config = {
            "system_instruction": prompt_package.system_instructions,
            "response_mime_type": "application/json",
            "response_schema": COMMENTARY_JSON_SCHEMA,
            "temperature": 0.1,
            "max_output_tokens": self.max_output_tokens,
        }
        if self._types is not None:
            try:
                config["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

        response = self._client.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=prompt_package.user_prompt,
            config=config,
        )

        payload = self._extract_json_payload(response)
        normalized = validate_commentary_payload(
            payload,
            expected_selected_index=expected_selected_index,
            valid_candidate_indices=self._candidate_indices(evidence),
        )
        return normalized.as_dict()

    @staticmethod
    def _extract_json_payload(response: Any) -> Dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Mapping):
            return dict(parsed)
        text = getattr(response, "text", None)
        if text is None and isinstance(response, Mapping):
            text = response.get("text")
        if text is None:
            raise RuntimeError("Gemini response did not contain JSON text.")
        try:
            payload = json.loads(str(text))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini response JSON must be an object.")
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
    """Prefer Gemini, then OpenAI, then deterministic fallback."""

    for backend in (GeminiCommentaryBackend(), OpenAIResponsesCommentaryBackend()):
        if backend.available():
            return backend
    return DeterministicCommentaryBackend()
