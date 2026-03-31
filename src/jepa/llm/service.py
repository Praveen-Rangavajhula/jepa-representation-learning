"""High-level commentary service with validation and deterministic fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from jepa.commentary import (
    DeterministicCommentaryGenerator,
    GroundedCommentary,
    LLMReadyCommentaryBuilder,
    LLMReadyCommentaryContext,
)

from .backends import (
    CommentaryBackend,
    DeterministicCommentaryBackend,
    build_default_commentary_backend,
)
from .schema import ValidatedCommentaryPayload, validate_commentary_payload


@dataclass(slots=True)
class CommentaryGenerationResult:
    """Validated commentary result returned by the LLM service."""

    commentary: GroundedCommentary
    payload: Dict[str, Any]
    backend_name: str
    used_fallback: bool
    validation_error: str | None = None
    llm_context: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "commentary": self.commentary.as_dict(),
            "payload": dict(self.payload),
            "backend_name": self.backend_name,
            "used_fallback": self.used_fallback,
            "validation_error": self.validation_error,
            "llm_context": dict(self.llm_context),
        }


class LLMCommentaryService:
    """Produce grounded commentary with a live LLM backend when available."""

    def __init__(
        self,
        *,
        backend: CommentaryBackend | None = None,
        fallback_backend: CommentaryBackend | None = None,
        prompt_builder: Optional[LLMReadyCommentaryBuilder] = None,
        deterministic_generator: Optional[DeterministicCommentaryGenerator] = None,
    ) -> None:
        self.backend = backend or build_default_commentary_backend()
        self.fallback_backend = fallback_backend or DeterministicCommentaryBackend()
        self.prompt_builder = prompt_builder or LLMReadyCommentaryBuilder()
        self.deterministic_generator = deterministic_generator or DeterministicCommentaryGenerator()

    def generate(
        self,
        evidence: Mapping[str, Any],
        *,
        expected_selected_index: int | None = None,
    ) -> CommentaryGenerationResult:
        prompt_context = self._build_prompt_context(evidence)
        valid_indices = self._candidate_indices(evidence)
        selected_index = self._selected_index(evidence, fallback=expected_selected_index)

        raw_payload: Dict[str, Any] | None = None
        backend_name = getattr(self.backend, "name", type(self.backend).__name__)
        validation_error: str | None = None
        used_fallback = False

        try:
            if not self.backend.available():
                raise RuntimeError(f"Backend unavailable: {backend_name}")
            raw_payload = self.backend.generate(
                evidence,
                prompt_package=prompt_context,
                expected_selected_index=selected_index,
            )
            validated = validate_commentary_payload(
                raw_payload,
                expected_selected_index=selected_index,
                valid_candidate_indices=valid_indices,
            )
        except Exception as exc:
            validation_error = str(exc)
            used_fallback = True
            backend_name = getattr(self.fallback_backend, "name", type(self.fallback_backend).__name__)
            raw_payload = self.fallback_backend.generate(
                evidence,
                prompt_package=prompt_context,
                expected_selected_index=selected_index,
            )
            validated = validate_commentary_payload(
                raw_payload,
                expected_selected_index=selected_index,
                valid_candidate_indices=valid_indices,
            )

        commentary = self._build_grounded_commentary(
            evidence,
            validated,
        )
        return CommentaryGenerationResult(
            commentary=commentary,
            payload=validated.as_dict(),
            backend_name=backend_name,
            used_fallback=used_fallback,
            validation_error=validation_error,
            llm_context=prompt_context.as_dict(),
        )

    def _build_prompt_context(self, evidence: Mapping[str, Any]) -> LLMReadyCommentaryContext:
        base_context = self.prompt_builder.build(evidence)
        extra = self._extra_prompt_evidence(evidence)
        user_prompt = base_context.user_prompt.rstrip() + "\n\nAdditional grounded evidence:\n"
        user_prompt += json.dumps(extra, indent=2, ensure_ascii=True)
        user_prompt += (
            "\n\nReturn only JSON that matches the required schema. "
            "Do not add markdown, bullets, or code fences."
        )
        return LLMReadyCommentaryContext(
            system_instructions=base_context.system_instructions,
            user_prompt=user_prompt,
            evidence={**base_context.evidence, **extra},
        )

    @staticmethod
    def _extra_prompt_evidence(evidence: Mapping[str, Any]) -> Dict[str, Any]:
        ranking_summary = dict(evidence.get("ranking_summary", {}))
        selected = dict(evidence.get("selected_candidate") or {})
        runner_up = dict(evidence.get("runner_up_candidate") or {})
        return {
            "observed_description": evidence.get("observed_description"),
            "candidate_descriptions": evidence.get("candidate_descriptions"),
            "selected_candidate": selected,
            "runner_up_candidate": runner_up,
            "candidate_score_table": evidence.get("candidate_score_table"),
            "score_margin": evidence.get("score_margin")
            or ranking_summary.get("score_margin")
            or ranking_summary.get("score_gap_to_runner_up"),
            "confidence_margin": evidence.get("confidence_margin")
            or ranking_summary.get("confidence_margin")
            or ranking_summary.get("confidence"),
            "confidence_tier": evidence.get("confidence_tier")
            or ranking_summary.get("confidence_tier")
            or ranking_summary.get("uncertainty"),
            "heuristic_comparison": evidence.get("baseline_comparison"),
        }

    def _build_grounded_commentary(
        self,
        evidence: Mapping[str, Any],
        payload: ValidatedCommentaryPayload,
    ) -> GroundedCommentary:
        deterministic = self.deterministic_generator.generate(evidence)
        ranking_summary = dict(evidence.get("ranking_summary", {}))
        selected = dict(evidence.get("selected_candidate") or {})

        selected_generation_type = str(
            selected.get("generation_type")
            or ranking_summary.get("selected_generation_type")
            or "unknown"
        )
        evaluator_name = str(ranking_summary.get("evaluator_name") or evidence.get("evaluator_name") or "unknown")

        confidence_margin = (
            ranking_summary.get("confidence_margin")
            or ranking_summary.get("confidence")
        )
        confidence_tier = ranking_summary.get("confidence_tier") or ranking_summary.get("uncertainty")
        score_margin = ranking_summary.get("score_margin") or ranking_summary.get("score_gap_to_runner_up")

        text_parts = [
            payload.anticipation_summary.strip(),
            payload.why_selected.strip(),
            payload.uncertainty_note.strip(),
        ]
        if payload.baseline_note.strip():
            text_parts.append(payload.baseline_note.strip())
        text = " ".join(part for part in text_parts if part)

        evidence_highlights: list[str] = []

        def _append_highlight(item: str | None) -> None:
            if item is None:
                return
            text = str(item).strip()
            if text and text not in evidence_highlights:
                evidence_highlights.append(text)

        for item in deterministic.evidence_highlights:
            _append_highlight(item)
        if score_margin is not None:
            _append_highlight(f"Score margin = {float(score_margin):.4f}")
        if confidence_margin is not None:
            label = confidence_tier if confidence_tier is not None else "unknown"
            _append_highlight(f"Confidence margin = {float(confidence_margin):.4f} ({label})")
        _append_highlight(payload.baseline_note)

        return GroundedCommentary(
            evaluator_name=evaluator_name,
            selected_candidate_index=int(payload.selected_candidate_index),
            selected_generation_type=selected_generation_type,
            anticipated_future_summary=payload.anticipation_summary.strip(),
            why_selected=payload.why_selected.strip(),
            uncertainty_statement=payload.uncertainty_note.strip(),
            close_alternative_warning=deterministic.close_alternative_warning,
            baseline_disagreement_note=payload.baseline_note.strip() or deterministic.baseline_disagreement_note,
            evidence_highlights=evidence_highlights,
            text=text,
        )

    @staticmethod
    def _candidate_indices(evidence: Mapping[str, Any]) -> Sequence[int]:
        table = list(evidence.get("candidate_score_table", []))
        indices: list[int] = []
        for row in table:
            try:
                indices.append(int(row.get("candidate_index")))
            except Exception:
                continue
        return indices

    @staticmethod
    def _selected_index(
        evidence: Mapping[str, Any],
        *,
        fallback: int | None = None,
    ) -> int:
        ranking_summary = dict(evidence.get("ranking_summary", {}))
        selected = evidence.get("selected_candidate") or {}
        for candidate in (
            fallback,
            ranking_summary.get("selected_index"),
            selected.get("candidate_index"),
        ):
            if candidate is not None:
                return int(candidate)
        raise ValueError("Evidence does not include a selected candidate index.")


def build_default_commentary_service() -> LLMCommentaryService:
    """Convenience factory for notebook and demo usage."""

    return LLMCommentaryService()


def generate_commentary(
    evidence: Mapping[str, Any],
    *,
    service: Optional[LLMCommentaryService] = None,
    backend: CommentaryBackend | None = None,
    expected_selected_index: int | None = None,
) -> CommentaryGenerationResult:
    """Convenience wrapper that validates and returns grounded commentary."""

    commentary_service = service or LLMCommentaryService(backend=backend)
    return commentary_service.generate(
        evidence,
        expected_selected_index=expected_selected_index,
    )
