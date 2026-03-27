"""Notebook-friendly tables and summaries for V-JEPA future selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _candidate_rows_from_bundle(bundle: Any) -> List[Dict[str, Any]]:
    if hasattr(bundle, "candidate_scores"):
        rows = []
        for item in bundle.candidate_scores:
            rows.append(
                {
                    "candidate_index": item.candidate_index,
                    "rank": item.rank,
                    "score": item.score,
                    "probability": getattr(item, "probability", None),
                    "generation_type": item.generation_type,
                    "rationale": item.rationale,
                    "components": dict(item.components),
                    "is_true": getattr(item, "is_true", None),
                }
            )
        return rows

    if hasattr(bundle, "ranked_candidates"):
        rows = []
        for rank, item in enumerate(bundle.ranked_candidates, start=1):
            rows.append(
                {
                    "candidate_index": item.candidate_index,
                    "rank": rank,
                    "score": item.score,
                    "probability": None,
                    "generation_type": item.generation_type,
                    "rationale": item.rationale,
                    "components": dict(item.components),
                    "is_true": None,
                }
            )
        return rows

    raise TypeError(f"Unsupported bundle type for candidate table: {type(bundle)!r}")


def build_candidate_score_table(bundle: Any, *, correct_index: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = _candidate_rows_from_bundle(bundle)
    for row in rows:
        row["is_correct"] = bool(correct_index is not None and row["candidate_index"] == int(correct_index))
    rows.sort(key=lambda item: item["rank"])
    return rows


def build_ranking_summary(bundle: Any, *, correct_index: Optional[int] = None) -> Dict[str, Any]:
    rows = build_candidate_score_table(bundle, correct_index=correct_index)
    selected = rows[0] if rows else None
    correct_rank = None
    if correct_index is not None:
        for row in rows:
            if row["candidate_index"] == int(correct_index):
                correct_rank = int(row["rank"])
                break

    return {
        "evaluator_name": getattr(bundle, "evaluator_name", type(bundle).__name__),
        "selected_index": selected["candidate_index"] if selected else -1,
        "selected_generation_type": selected["generation_type"] if selected else "unknown",
        "confidence": getattr(bundle, "confidence", None),
        "uncertainty": getattr(bundle, "uncertainty", None),
        "correct_index": int(correct_index) if correct_index is not None else None,
        "correct_rank": correct_rank,
        "was_correct": bool(correct_index is not None and selected is not None and selected["candidate_index"] == int(correct_index)),
    }


def build_baseline_comparison_table(
    vjepa_bundle: Any,
    baseline_bundle: Any,
    *,
    correct_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    vjepa_summary = build_ranking_summary(vjepa_bundle, correct_index=correct_index)
    baseline_summary = build_ranking_summary(baseline_bundle, correct_index=correct_index)
    return [
        {"evaluator": "vjepa", **vjepa_summary},
        {"evaluator": "heuristic", **baseline_summary},
    ]
