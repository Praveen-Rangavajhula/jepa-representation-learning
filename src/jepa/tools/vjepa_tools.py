"""Notebook-friendly tables and summaries for future-selection evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


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


def build_component_highlights(components: Mapping[str, Any], *, top_n: int = 2) -> List[Dict[str, float]]:
    ordered = sorted(
        ((str(name), float(value)) for name, value in components.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"name": name, "value": value} for name, value in ordered[:top_n]]


def _evaluator_label(bundle: Any) -> str:
    name = str(getattr(bundle, "evaluator_name", type(bundle).__name__)).lower()
    if "latent_predictor" in name:
        return "latent_predictor"
    if name.startswith("vjepa2"):
        return "vjepa"
    if name in {"heuristic", "futureselectionresult"}:
        return "heuristic"
    return name


def build_evidence_highlights(bundle: Any, *, correct_index: Optional[int] = None) -> Dict[str, Any]:
    rows = build_candidate_score_table(bundle, correct_index=correct_index)
    selected = dict(rows[0]) if rows else None
    runner_up = dict(rows[1]) if len(rows) > 1 else None
    score_gap = None
    probability_gap = None
    if selected is not None and runner_up is not None:
        score_gap = float(selected["score"]) - float(runner_up["score"])
        if selected.get("probability") is not None and runner_up.get("probability") is not None:
            probability_gap = float(selected["probability"]) - float(runner_up["probability"])

    return {
        "selected_candidate": selected,
        "runner_up_candidate": runner_up,
        "selected_component_highlights": build_component_highlights(selected["components"]) if selected else [],
        "runner_up_component_highlights": build_component_highlights(runner_up["components"]) if runner_up else [],
        "score_gap_to_runner_up": score_gap,
        "probability_gap_to_runner_up": probability_gap,
    }


def build_ranking_summary(bundle: Any, *, correct_index: Optional[int] = None) -> Dict[str, Any]:
    rows = build_candidate_score_table(bundle, correct_index=correct_index)
    selected = rows[0] if rows else None
    runner_up = rows[1] if len(rows) > 1 else None
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
        "selected_score": selected["score"] if selected else None,
        "selected_probability": selected["probability"] if selected else None,
        "selected_component_highlights": build_component_highlights(selected["components"]) if selected else [],
        "runner_up_index": runner_up["candidate_index"] if runner_up else None,
        "runner_up_generation_type": runner_up["generation_type"] if runner_up else None,
        "runner_up_score": runner_up["score"] if runner_up else None,
        "runner_up_probability": runner_up["probability"] if runner_up else None,
        "score_gap_to_runner_up": (
            float(selected["score"]) - float(runner_up["score"])
            if selected is not None and runner_up is not None
            else None
        ),
        "probability_gap_to_runner_up": (
            float(selected["probability"]) - float(runner_up["probability"])
            if selected is not None
            and runner_up is not None
            and selected.get("probability") is not None
            and runner_up.get("probability") is not None
            else None
        ),
        "confidence": getattr(bundle, "confidence", None),
        "uncertainty": getattr(bundle, "uncertainty", None),
        "correct_index": int(correct_index) if correct_index is not None else None,
        "correct_rank": correct_rank,
        "was_correct": bool(correct_index is not None and selected is not None and selected["candidate_index"] == int(correct_index)),
    }


def build_baseline_comparison_table(
    primary_bundle: Any,
    baseline_bundle: Any,
    *,
    correct_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return build_multi_evaluator_comparison_table(
        {
            _evaluator_label(primary_bundle): primary_bundle,
            _evaluator_label(baseline_bundle): baseline_bundle,
        },
        correct_index=correct_index,
    )


def build_multi_evaluator_comparison_table(
    bundles: Mapping[str, Any],
    *,
    correct_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    comparison_rows: List[Dict[str, Any]] = []
    for evaluator_label, bundle in bundles.items():
        comparison_rows.append(
            {
                "evaluator": evaluator_label,
                **build_ranking_summary(bundle, correct_index=correct_index),
            }
        )
    return comparison_rows
