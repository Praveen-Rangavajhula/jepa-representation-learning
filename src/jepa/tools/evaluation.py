"""Benchmark helpers for future-selection evaluators."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from jepa.agents import run_future_selection_pipeline
from jepa.scoring import average_correct_rank, mean_reciprocal_rank

from .vjepa_tools import build_candidate_score_table, build_ranking_summary


@dataclass(slots=True)
class FutureSelectionBenchmarkResult:
    """Aggregated benchmark outputs for multiple evaluators."""

    evaluation_count: int
    summary: Dict[str, Any]
    per_negative_type: Dict[str, Dict[str, Dict[str, float]]]
    confusion_report: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    candidate_rows: List[Dict[str, Any]] = field(default_factory=list)
    per_example: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_count": self.evaluation_count,
            "summary": dict(self.summary),
            "per_negative_type": {
                evaluator: {strategy: dict(values) for strategy, values in strategy_map.items()}
                for evaluator, strategy_map in self.per_negative_type.items()
            },
            "confusion_report": {
                evaluator: {strategy: dict(values) for strategy, values in strategy_map.items()}
                for evaluator, strategy_map in self.confusion_report.items()
            },
            "candidate_rows": list(self.candidate_rows),
            "per_example": list(self.per_example),
        }


def run_future_selection_benchmark(
    dataset: Sequence[Any],
    evaluators: Mapping[str, Any],
    *,
    evaluation_count: Optional[int] = None,
    show_progress: bool = False,
    progress_interval: int = 1,
) -> FutureSelectionBenchmarkResult:
    if not evaluators:
        raise ValueError("At least one evaluator is required.")

    count = min(len(dataset), int(evaluation_count or len(dataset)))
    correct_counts: Dict[str, int] = {name: 0 for name in evaluators}
    correct_ranks: Dict[str, List[int]] = {name: [] for name in evaluators}
    score_margins: Dict[str, List[float]] = {name: [] for name in evaluators}
    confidence_margins: Dict[str, List[float]] = {name: [] for name in evaluators}
    per_negative: Dict[str, MutableMapping[str, Dict[str, int]]] = {
        name: defaultdict(lambda: {"count": 0, "correct": 0})
        for name in evaluators
    }
    confusion_counts: Dict[str, MutableMapping[str, Dict[str, int]]] = {
        name: defaultdict(lambda: {"count": 0})
        for name in evaluators
    }
    candidate_rows: List[Dict[str, Any]] = []
    per_example: List[Dict[str, Any]] = []

    for index in range(count):
        example = dataset[index]
        metadata = list(example.metadata["candidate_strategies"])
        negative_types = [item["strategy"] for item in metadata if not item.get("is_true", False)]
        example_record: Dict[str, Any] = {
            "example_index": index,
            "correct_index": int(example.correct_index),
            "evaluators": {},
        }

        for evaluator_name, evaluator in evaluators.items():
            bundle = _score_example(example, evaluator_name=evaluator_name, evaluator=evaluator)
            summary = build_ranking_summary(bundle, correct_index=example.correct_index)
            table = build_candidate_score_table(bundle, correct_index=example.correct_index)
            stored_summary = dict(summary)
            stored_summary.pop("uncertainty", None)
            candidate_descriptions = [row.get("candidate_description") for row in table]
            score_margin = summary.get("score_margin")
            if score_margin is None:
                score_margin = summary.get("score_gap_to_runner_up")
            confidence_margin = summary.get("confidence_margin")
            if confidence_margin is None:
                confidence_margin = summary.get("probability_gap_to_runner_up")
            confidence_tier = summary.get("confidence_tier")
            if confidence_tier is None:
                confidence_tier = summary.get("uncertainty")
            if confidence_tier is None:
                confidence_tier = "unknown"

            if summary["was_correct"]:
                correct_counts[evaluator_name] += 1
            if summary["correct_rank"] is not None:
                correct_ranks[evaluator_name].append(int(summary["correct_rank"]))
            if score_margin is not None:
                score_margins[evaluator_name].append(float(score_margin))
            if confidence_margin is not None:
                confidence_margins[evaluator_name].append(float(confidence_margin))

            for strategy in negative_types:
                per_negative[evaluator_name][strategy]["count"] += 1
                if summary["was_correct"]:
                    per_negative[evaluator_name][strategy]["correct"] += 1
            if not summary["was_correct"]:
                selected_type = str(summary.get("selected_generation_type") or "unknown")
                confusion_counts[evaluator_name][selected_type]["count"] += 1

            for row in table:
                candidate_rows.append(
                    {
                        "example_index": index,
                        "evaluator": evaluator_name,
                        "candidate_index": row["candidate_index"],
                        "rank": row["rank"],
                        "score": row["score"],
                        "probability": row["probability"],
                        "generation_type": row["generation_type"],
                        "candidate_description": row.get("candidate_description"),
                        "is_correct_candidate": row["is_correct"],
                        "selected_index": summary["selected_index"],
                        "correct_index": int(example.correct_index),
                        "score_margin": score_margin,
                        "confidence_margin": confidence_margin,
                        "confidence_tier": confidence_tier,
                    }
                )

            example_record["evaluators"][evaluator_name] = {
                "ranking_summary": stored_summary,
                "candidate_score_table": table,
                "score_margin": score_margin,
                "confidence_margin": confidence_margin,
                "confidence_tier": confidence_tier,
                "candidate_descriptions": candidate_descriptions,
            }

        per_example.append(example_record)
        if show_progress and (
            index == 0
            or index + 1 == count
            or ((index + 1) % max(progress_interval, 1) == 0)
        ):
            print(f"Benchmark progress: {index + 1}/{count} examples evaluated.")

    summary = {
        "evaluation_count": count,
    }
    for evaluator_name in evaluators:
        score_margin_mean, score_margin_variance = _mean_and_variance(score_margins[evaluator_name])
        confidence_margin_mean, confidence_margin_variance = _mean_and_variance(
            confidence_margins[evaluator_name]
        )
        summary[evaluator_name] = {
            "top1_accuracy": correct_counts[evaluator_name] / max(count, 1),
            "mean_reciprocal_rank": mean_reciprocal_rank(correct_ranks[evaluator_name]),
            "average_correct_rank": average_correct_rank(correct_ranks[evaluator_name]),
            "score_margin_mean": score_margin_mean,
            "score_margin_variance": score_margin_variance,
            "confidence_margin_mean": confidence_margin_mean,
            "confidence_margin_variance": confidence_margin_variance,
        }

    per_negative_type: Dict[str, Dict[str, Dict[str, float]]] = {}
    confusion_report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for evaluator_name, strategy_map in per_negative.items():
        per_negative_type[evaluator_name] = {}
        for strategy, payload in strategy_map.items():
            strategy_count = int(payload["count"])
            per_negative_type[evaluator_name][strategy] = {
                "count": strategy_count,
                "accuracy_when_present": (
                    float(payload["correct"]) / float(strategy_count) if strategy_count else 0.0
                ),
            }
        evaluator_errors = max(count - correct_counts[evaluator_name], 0)
        confusion_report[evaluator_name] = {}
        for strategy, payload in confusion_counts[evaluator_name].items():
            confusion_count = int(payload["count"])
            confusion_report[evaluator_name][strategy] = {
                "count": confusion_count,
                "fraction_of_errors": (
                    float(confusion_count) / float(evaluator_errors) if evaluator_errors else 0.0
                ),
                "fraction_of_examples": float(confusion_count) / float(max(count, 1)),
            }

    return FutureSelectionBenchmarkResult(
        evaluation_count=count,
        summary=summary,
        per_negative_type=per_negative_type,
        confusion_report=confusion_report,
        candidate_rows=candidate_rows,
        per_example=per_example,
    )


def save_future_selection_benchmark_artifacts(
    benchmark: FutureSelectionBenchmarkResult,
    output_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    title: str = "Future-Selection Evaluation Summary",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary_path = output_dir / "benchmark_summary.json"
    per_negative_type_path = output_dir / "per_negative_type.json"
    confusion_report_path = output_dir / "confusion_report.json"
    candidate_rankings_path = output_dir / "candidate_rankings.csv"
    per_example_path = output_dir / "per_example_rankings.json"

    benchmark_summary_path.write_text(json.dumps(benchmark.summary, indent=2), encoding="utf-8")
    per_negative_type_path.write_text(json.dumps(benchmark.per_negative_type, indent=2), encoding="utf-8")
    confusion_report_path.write_text(json.dumps(benchmark.confusion_report, indent=2), encoding="utf-8")
    per_example_path.write_text(json.dumps(benchmark.per_example, indent=2), encoding="utf-8")

    with candidate_rankings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_index",
                "evaluator",
                "candidate_index",
                "rank",
                "score",
                "probability",
                "generation_type",
                "candidate_description",
                "is_correct_candidate",
                "selected_index",
                "correct_index",
                "score_margin",
                "confidence_margin",
                "confidence_tier",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark.candidate_rows)

    saved = {
        "benchmark_summary": benchmark_summary_path,
        "per_negative_type": per_negative_type_path,
        "confusion_report": confusion_report_path,
        "candidate_rankings": candidate_rankings_path,
        "per_example_rankings": per_example_path,
    }

    if report_path is not None:
        report_target = Path(report_path)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(
            _render_markdown_summary(
                title=title,
                summary=benchmark.summary,
                metadata=metadata,
                artifact_paths=saved,
            ),
            encoding="utf-8",
        )
        saved["report"] = report_target

    return saved


def _score_example(example: Any, *, evaluator_name: str, evaluator: Any) -> Any:
    if evaluator_name == "heuristic" or evaluator is None:
        return run_future_selection_pipeline(
            observed=example.observed,
            candidates=example.candidates,
            candidate_metadata=example.metadata["candidate_strategies"],
            evaluation_mode="heuristic",
        )

    if hasattr(evaluator, "score_example"):
        return evaluator.score_example(
            example.observed,
            example.candidates,
            candidate_metadata=example.metadata["candidate_strategies"],
        )

    if hasattr(evaluator, "score_future_candidates"):
        return evaluator.score_future_candidates(
            example.observed,
            example.candidates,
            candidate_metadata=example.metadata["candidate_strategies"],
        )

    raise TypeError(
        f"Evaluator {evaluator_name!r} must be None, expose score_example(), or expose score_future_candidates()."
    )


def _render_markdown_summary(
    *,
    title: str,
    summary: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]],
    artifact_paths: Mapping[str, Path],
) -> str:
    lines = [f"# {title}", ""]
    if metadata:
        lines.append("## Run metadata")
        for key, value in metadata.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## Metrics")
    lines.append(f"- Evaluation count: {summary.get('evaluation_count', 0)}")
    for evaluator_name, payload in summary.items():
        if evaluator_name == "evaluation_count":
            continue
        lines.append(f"- {evaluator_name} Top-1 accuracy: {payload['top1_accuracy']:.4f}")
        lines.append(f"- {evaluator_name} MRR: {payload['mean_reciprocal_rank']:.4f}")
        lines.append(f"- {evaluator_name} average correct rank: {payload['average_correct_rank']:.4f}")
        lines.append(f"- {evaluator_name} score margin mean: {payload['score_margin_mean']:.4f}")
        lines.append(f"- {evaluator_name} score margin variance: {payload['score_margin_variance']:.4f}")
        lines.append(
            f"- {evaluator_name} confidence margin mean: {payload['confidence_margin_mean']:.4f}"
        )
        lines.append(
            f"- {evaluator_name} confidence margin variance: {payload['confidence_margin_variance']:.4f}"
        )
    lines.append("")

    if metadata and metadata.get("confusion_report"):
        lines.append("## Focused Confusions")
        confusion_report = metadata["confusion_report"]
        for evaluator_name, payload in confusion_report.items():
            if not payload:
                lines.append(f"- {evaluator_name}: no wrong selections recorded")
                continue
            ordered = sorted(payload.items(), key=lambda item: item[1].get("count", 0), reverse=True)
            for strategy, values in ordered:
                lines.append(
                    f"- {evaluator_name} -> {strategy}: count={values['count']}, "
                    f"fraction_of_errors={values['fraction_of_errors']:.4f}"
                )
        lines.append("")

    lines.append("## Artifact paths")
    for name, path in artifact_paths.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines).strip() + "\n"


def _mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    floats = [float(value) for value in values]
    if len(floats) == 1:
        return floats[0], 0.0
    return float(fmean(floats)), float(pvariance(floats))
