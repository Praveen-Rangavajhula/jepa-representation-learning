"""Tensor-free context builders for the later LLM-agentic layer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .vjepa_tools import build_baseline_comparison_table, build_candidate_score_table, build_ranking_summary


def build_future_selection_context(
    example: Any,
    score_bundle: Any,
    baseline_bundle: Optional[Any] = None,
) -> Dict[str, Any]:
    correct_index = int(getattr(example, "correct_index", -1))
    candidate_table = build_candidate_score_table(score_bundle, correct_index=correct_index)
    ranking_summary = build_ranking_summary(score_bundle, correct_index=correct_index)

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
        "candidate_metadata": list(getattr(example, "metadata", {}).get("candidate_strategies", [])),
        "notes": list(getattr(score_bundle, "notes", [])),
    }

    if baseline_bundle is not None:
        context["baseline_comparison"] = build_baseline_comparison_table(
            score_bundle,
            baseline_bundle,
            correct_index=correct_index,
        )

    return context
