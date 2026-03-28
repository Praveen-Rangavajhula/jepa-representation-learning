"""Tensor-free context builders for the later LLM-agentic layer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .vjepa_tools import (
    build_baseline_comparison_table,
    build_candidate_score_table,
    build_evidence_highlights,
    build_ranking_summary,
)


def build_future_selection_context(
    example: Any,
    score_bundle: Any,
    baseline_bundle: Optional[Any] = None,
) -> Dict[str, Any]:
    correct_index = int(getattr(example, "correct_index", -1))
    candidate_table = build_candidate_score_table(score_bundle, correct_index=correct_index)
    ranking_summary = _strip_legacy_uncertainty(build_ranking_summary(score_bundle, correct_index=correct_index))
    evidence_highlights = build_evidence_highlights(score_bundle, correct_index=correct_index)
    metadata_items = list(getattr(example, "metadata", {}).get("candidate_strategies", []))
    candidate_count = int(getattr(example, "candidates").shape[0])
    candidate_descriptions = [None] * candidate_count
    for row in candidate_table:
        description = row.get("candidate_description")
        candidate_index = int(row.get("candidate_index", -1))
        if description is None and 0 <= candidate_index < len(metadata_items):
            details = metadata_items[candidate_index].get("details", {})
            description = details.get("description") or metadata_items[candidate_index].get("description")
        if 0 <= candidate_index < len(candidate_descriptions):
            candidate_descriptions[candidate_index] = description

    context: Dict[str, Any] = {
        "task_type": "future_selection",
        "observed_clip_summary": {
            "shape": list(getattr(example, "observed").shape),
            "frame_count": int(getattr(example, "observed").shape[0]),
        },
        "candidate_clip_summary": {
            "shape": list(getattr(example, "candidates").shape),
            "candidate_count": int(getattr(example, "candidates").shape[0]),
            "future_frames": int(getattr(example, "candidates").shape[1]),
        },
        "correct_index": correct_index,
        "candidate_score_table": candidate_table,
        "ranking_summary": ranking_summary,
        "selected_candidate": evidence_highlights["selected_candidate"],
        "runner_up_candidate": evidence_highlights["runner_up_candidate"],
        "evidence_highlights": evidence_highlights,
        "candidate_metadata": metadata_items,
        "candidate_descriptions": candidate_descriptions,
        "observed_description": getattr(example, "metadata", {}).get("observed_description"),
        "score_margin": ranking_summary.get("score_margin"),
        "confidence_margin": ranking_summary.get("confidence_margin"),
        "confidence_tier": ranking_summary.get("confidence_tier"),
        "notes": list(getattr(score_bundle, "notes", [])),
    }

    if baseline_bundle is not None:
        baseline_comparison = [
            _strip_legacy_uncertainty(row)
            for row in build_baseline_comparison_table(
                score_bundle,
                baseline_bundle,
                correct_index=correct_index,
            )
        ]
        context["baseline_comparison"] = baseline_comparison
        if len(baseline_comparison) >= 2:
            context["baseline_agreement"] = (
                baseline_comparison[0]["selected_index"] == baseline_comparison[1]["selected_index"]
            )
        else:
            context["baseline_agreement"] = None
        context["baseline_selected_index"] = baseline_comparison[1]["selected_index"] if len(baseline_comparison) > 1 else None

    return context


def _strip_legacy_uncertainty(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("uncertainty", None)
    return cleaned
