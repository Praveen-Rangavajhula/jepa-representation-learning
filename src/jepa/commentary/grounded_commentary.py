"""Grounded commentary generation for future-selection outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


GENERATION_TYPE_LABELS = {
    "true_continuation": "the continuation that best preserves the observed trajectory",
    "shuffled_temporal_order": "a continuation with plausible frames but weakened temporal order",
    "wrong_velocity_continuation": "a continuation that changes speed relative to the observed prefix",
    "wrong_direction_continuation": "a continuation that redirects the trajectory",
    "future_segment_from_other_sample": "a visually plausible continuation borrowed from another sequence",
    "mirrored_or_perturbed_continuation": "a continuation with mirrored or perturbed motion cues",
    "unknown": "the continuation pattern represented by the candidate",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _component_highlights(components: Mapping[str, Any], *, top_n: int = 2) -> List[Dict[str, float]]:
    ordered = sorted(
        ((str(name), _as_float(value)) for name, value in components.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"name": name, "value": value} for name, value in ordered[:top_n]]


def _describe_generation_type(generation_type: str) -> str:
    return GENERATION_TYPE_LABELS.get(generation_type, GENERATION_TYPE_LABELS["unknown"])


@dataclass(slots=True)
class GroundedCommentary:
    """Structured grounded commentary for a future-selection example."""

    evaluator_name: str
    selected_candidate_index: int
    selected_generation_type: str
    anticipated_future_summary: str
    why_selected: str
    uncertainty_statement: str
    close_alternative_warning: Optional[str] = None
    baseline_disagreement_note: Optional[str] = None
    evidence_highlights: List[str] = field(default_factory=list)
    text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "selected_candidate_index": self.selected_candidate_index,
            "selected_generation_type": self.selected_generation_type,
            "anticipated_future_summary": self.anticipated_future_summary,
            "why_selected": self.why_selected,
            "uncertainty_statement": self.uncertainty_statement,
            "close_alternative_warning": self.close_alternative_warning,
            "baseline_disagreement_note": self.baseline_disagreement_note,
            "evidence_highlights": list(self.evidence_highlights),
            "text": self.text,
        }

    def to_markdown(self) -> str:
        lines = [
            "### Grounded Commentary",
            "",
            self.text.strip(),
            "",
            "**Evidence highlights**",
        ]
        for item in self.evidence_highlights:
            lines.append(f"- {item}")
        return "\n".join(lines).strip()


@dataclass(slots=True)
class LLMReadyCommentaryContext:
    """Tensor-free prompt package for a future LLM commentary layer."""

    system_instructions: str
    user_prompt: str
    evidence: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "system_instructions": self.system_instructions,
            "user_prompt": self.user_prompt,
            "evidence": dict(self.evidence),
        }


class DeterministicCommentaryGenerator:
    """Render grounded commentary from structured scoring evidence."""

    def __init__(
        self,
        *,
        close_score_gap_threshold: float = 0.05,
        low_confidence_threshold: float = 0.10,
    ) -> None:
        self.close_score_gap_threshold = close_score_gap_threshold
        self.low_confidence_threshold = low_confidence_threshold

    def generate(self, evidence: Mapping[str, Any]) -> GroundedCommentary:
        ranking_summary = dict(evidence.get("ranking_summary", {}))
        selected = dict(evidence.get("selected_candidate") or {})
        if not selected:
            candidates = list(evidence.get("candidate_score_table", []))
            if candidates:
                selected = dict(candidates[0])
        if not selected:
            raise ValueError("Commentary evidence must include a selected candidate.")

        runner_up = dict(evidence.get("runner_up_candidate") or {})
        candidate_count = int(evidence.get("candidate_clip_summary", {}).get("candidate_count", 0))
        observed_frames = int(evidence.get("observed_clip_summary", {}).get("frame_count", 0))

        selected_index = int(selected.get("candidate_index", ranking_summary.get("selected_index", -1)))
        selected_generation_type = str(
            selected.get("generation_type", ranking_summary.get("selected_generation_type", "unknown"))
        )
        evaluator_name = str(ranking_summary.get("evaluator_name", evidence.get("evaluator_name", "unknown")))
        score_gap = _as_float(
            ranking_summary.get("score_margin", ranking_summary.get("score_gap_to_runner_up")),
            _as_float(selected.get("score")) - _as_float(runner_up.get("score")),
        )
        confidence = ranking_summary.get("confidence")
        confidence_tier = str(
            ranking_summary.get("confidence_tier")
            or ranking_summary.get("uncertainty")
            or evidence.get("confidence_tier")
            or "unknown"
        )

        top_components = _component_highlights(selected.get("components", {}))
        runner_up_index = runner_up.get("candidate_index")
        selected_label = self._candidate_label(selected, selected_generation_type)
        runner_up_label = self._candidate_label(
            runner_up,
            str(runner_up.get("generation_type", "unknown")),
        )

        anticipated_future_summary = (
            f"For this example, the strongest predicted next step is candidate {selected_index}"
            f"{selected_label}. After the {observed_frames}-frame observed prefix, that is the future "
            "the system would present as its best guess."
        )

        component_phrase = self._component_phrase(top_components)
        why_selected = (
            f"It finished first under {evaluator_name}, driven mainly by {component_phrase}. "
            f"The lead over the next-best option is {score_gap:.4f}."
        )
        if runner_up_index is not None:
            why_selected += f" The closest alternative was candidate {int(runner_up_index)}{runner_up_label}."

        uncertainty_statement = self._uncertainty_statement(
            confidence=confidence,
            confidence_tier=confidence_tier,
            score_gap=score_gap,
            candidate_count=candidate_count,
        )

        close_warning = self._close_alternative_warning(
            runner_up=runner_up,
            confidence=confidence,
            confidence_tier=confidence_tier,
            score_gap=score_gap,
        )

        baseline_note = self._baseline_note(evidence)
        evidence_highlights = self._evidence_highlights(
            selected=selected,
            ranking_summary=ranking_summary,
            top_components=top_components,
            score_gap=score_gap,
            close_warning=close_warning,
            baseline_note=baseline_note,
        )
        text = self._render_text(
            anticipated_future_summary=anticipated_future_summary,
            why_selected=why_selected,
            uncertainty_statement=uncertainty_statement,
            close_warning=close_warning,
            baseline_note=baseline_note,
        )

        return GroundedCommentary(
            evaluator_name=evaluator_name,
            selected_candidate_index=selected_index,
            selected_generation_type=selected_generation_type,
            anticipated_future_summary=anticipated_future_summary,
            why_selected=why_selected,
            uncertainty_statement=uncertainty_statement,
            close_alternative_warning=close_warning,
            baseline_disagreement_note=baseline_note,
            evidence_highlights=evidence_highlights,
            text=text,
        )

    @staticmethod
    def _component_phrase(highlights: List[Dict[str, float]]) -> str:
        if not highlights:
            return "the available score evidence"
        if len(highlights) == 1:
            item = highlights[0]
            return f"`{item['name']}` ({item['value']:.3f})"
        first, second = highlights[:2]
        return (
            f"`{first['name']}` ({first['value']:.3f}) and "
            f"`{second['name']}` ({second['value']:.3f})"
        )

    def _uncertainty_statement(
        self,
        *,
        confidence: Any,
        confidence_tier: str,
        score_gap: float,
        candidate_count: int,
    ) -> str:
        if confidence is None:
            return (
                f"This evaluator does not expose a calibrated confidence margin, so treat the ranking over "
                f"{candidate_count} candidates as directional evidence rather than a hard probability claim."
            )

        confidence_value = _as_float(confidence)
        if confidence_tier == "high":
            return (
                f"This is a strong call: the top-two margin is {confidence_value:.4f}, so the selected "
                "future is meaningfully separated from the alternatives."
            )
        if confidence_tier == "medium":
            return (
                f"This is a reasonable call with a top-two margin of {confidence_value:.4f}; the leader "
                "looks plausible, but an alternative still remains competitive."
            )
        if score_gap <= self.close_score_gap_threshold:
            return (
                f"This is a close call: the top-two margin is {confidence_value:.4f}, and the gap to the "
                f"runner-up is only {score_gap:.4f}. I would present it as a tentative but defensible pick."
            )
        return (
            f"Confidence is still limited with a top-two margin of {confidence_value:.4f}, so frame this "
            "as a tentative preference rather than a decisive win."
        )

    def _close_alternative_warning(
        self,
        *,
        runner_up: Mapping[str, Any],
        confidence: Any,
        confidence_tier: str,
        score_gap: float,
    ) -> Optional[str]:
        if not runner_up:
            return None
        if (
            confidence_tier != "low"
            and _as_float(confidence, default=1.0) >= self.low_confidence_threshold
            and score_gap > self.close_score_gap_threshold
        ):
            return None

        runner_up_index = int(runner_up.get("candidate_index", -1))
        runner_up_type = str(runner_up.get("generation_type", "unknown"))
        runner_up_label = self._candidate_label(runner_up, runner_up_type)
        return (
            f"Candidate {runner_up_index}{runner_up_label} is still very close, so this should be framed "
            f"as a near tie rather than a clean separation. That alternative represents "
            f"{_describe_generation_type(runner_up_type)}."
        )

    @staticmethod
    def _baseline_note(evidence: Mapping[str, Any]) -> Optional[str]:
        comparison = list(evidence.get("baseline_comparison", []))
        if len(comparison) < 2:
            return None

        primary = comparison[0]
        baseline = comparison[1]
        if primary.get("selected_index") == baseline.get("selected_index"):
            return None

        primary_label = str(primary.get("evaluator", primary.get("evaluator_name", "primary")))
        baseline_label = str(baseline.get("evaluator", baseline.get("evaluator_name", "baseline")))
        return (
            f"The {baseline_label} baseline disagrees and selects candidate "
            f"{baseline.get('selected_index')}, while the {primary_label} path selects "
            f"candidate {primary.get('selected_index')}."
        )

    @staticmethod
    def _evidence_highlights(
        *,
        selected: Mapping[str, Any],
        ranking_summary: Mapping[str, Any],
        top_components: List[Dict[str, float]],
        score_gap: float,
        close_warning: Optional[str],
        baseline_note: Optional[str],
    ) -> List[str]:
        highlights = [
            f"Selected candidate: {int(selected.get('candidate_index', -1))} "
            f"({selected.get('generation_type', 'unknown')})",
            f"Evaluator: {ranking_summary.get('evaluator_name', 'unknown')}",
        ]

        for item in top_components:
            highlights.append(f"Top component `{item['name']}` = {item['value']:.4f}")

        confidence = ranking_summary.get("confidence")
        if confidence is not None:
            confidence_tier = ranking_summary.get("confidence_tier", ranking_summary.get("uncertainty", "unknown"))
            highlights.append(
                f"Confidence margin = {_as_float(confidence):.4f} ({confidence_tier})"
            )
        highlights.append(f"Score gap to runner-up = {score_gap:.4f}")

        if close_warning:
            highlights.append(close_warning)
        if baseline_note:
            highlights.append(baseline_note)
        return highlights

    @staticmethod
    def _render_text(
        *,
        anticipated_future_summary: str,
        why_selected: str,
        uncertainty_statement: str,
        close_warning: Optional[str],
        baseline_note: Optional[str],
    ) -> str:
        parts = [anticipated_future_summary, why_selected, uncertainty_statement]
        if close_warning:
            parts.append(close_warning)
        if baseline_note:
            parts.append(baseline_note)
        return " ".join(part.strip() for part in parts if part).strip()

    @staticmethod
    def _candidate_label(candidate: Mapping[str, Any], generation_type: str) -> str:
        description = (
            candidate.get("candidate_description")
            or candidate.get("selected_description")
            or candidate.get("description")
        )
        if description is not None and str(description).strip():
            return f" ({str(description).strip()})"
        return f" ({_describe_generation_type(generation_type)})"


class LLMReadyCommentaryBuilder:
    """Prepare a grounded prompt package for a later LLM commentary stage."""

    def build(self, evidence: Mapping[str, Any]) -> LLMReadyCommentaryContext:
        minimal_evidence = {
            "task_type": evidence.get("task_type"),
            "observed_clip_summary": evidence.get("observed_clip_summary"),
            "candidate_clip_summary": evidence.get("candidate_clip_summary"),
            "ranking_summary": evidence.get("ranking_summary"),
            "selected_candidate": evidence.get("selected_candidate"),
            "runner_up_candidate": evidence.get("runner_up_candidate"),
            "candidate_score_table": evidence.get("candidate_score_table"),
            "candidate_metadata": evidence.get("candidate_metadata"),
            "candidate_descriptions": evidence.get("candidate_descriptions"),
            "score_margin": evidence.get("score_margin"),
            "confidence_margin": evidence.get("confidence_margin"),
            "confidence_tier": evidence.get("confidence_tier"),
            "baseline_comparison": evidence.get("baseline_comparison"),
            "baseline_agreement": evidence.get("baseline_agreement"),
            "evidence_highlights": evidence.get("evidence_highlights"),
            "notes": evidence.get("notes"),
        }

        system_instructions = (
            "You are writing grounded, presentation-ready commentary for a class demo of a future-selection benchmark. "
            "Use only the provided evidence. Sound clear, confident, and helpful, but stay honest about uncertainty. "
            "Avoid robotic repetition, avoid restating the same score twice, prefer human-readable candidate descriptions "
            "over raw candidate numbers when available, and do not invent storylines, object identities, or causes."
        )
        user_prompt = (
            "Return only JSON.\n"
            "Write concise presenter notes that someone could comfortably read aloud in class.\n"
            "Field guidance:\n"
            "- anticipation_summary: 1-2 sentences saying what future the system favors and why that is the main takeaway.\n"
            "- why_selected: 1 sentence naming the strongest evidence without repeating the full summary.\n"
            "- uncertainty_note: 1 sentence that frames confidence honestly but constructively; if the margin is small, call it a close call instead of apologizing.\n"
            "- baseline_note: mention baseline agreement or disagreement only if it is genuinely useful; otherwise return an empty string.\n\n"
            f"Evidence JSON:\n{json.dumps(minimal_evidence, indent=2)}"
        )

        return LLMReadyCommentaryContext(
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            evidence=minimal_evidence,
        )
