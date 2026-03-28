"""Live example-by-example demo loop for the future-selection agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from jepa.agents import run_future_selection_pipeline
from jepa.llm import LLMCommentaryService, build_default_commentary_service
from jepa.tools.context_builder import build_future_selection_context
from jepa.tools.vjepa_tools import build_baseline_comparison_table, build_candidate_score_table, build_ranking_summary


@dataclass(slots=True)
class LiveAgentTranscriptEntry:
    """One live-demo example transcript."""

    example_index: int
    correct_index: int
    selected_index: int
    evaluator_name: str
    confidence_margin: float | None
    confidence_tier: str | None
    score_margin: float | None
    candidate_score_table: List[Dict[str, Any]]
    ranking_summary: Dict[str, Any]
    commentary: Dict[str, Any]
    backend_name: str
    used_fallback: bool
    baseline_comparison: Optional[List[Dict[str, Any]]] = None
    observed_description: str | None = None
    candidate_descriptions: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "example_index": self.example_index,
            "correct_index": self.correct_index,
            "selected_index": self.selected_index,
            "evaluator_name": self.evaluator_name,
            "confidence_margin": self.confidence_margin,
            "confidence_tier": self.confidence_tier,
            "score_margin": self.score_margin,
            "candidate_score_table": list(self.candidate_score_table),
            "ranking_summary": dict(self.ranking_summary),
            "commentary": dict(self.commentary),
            "backend_name": self.backend_name,
            "used_fallback": self.used_fallback,
            "baseline_comparison": list(self.baseline_comparison) if self.baseline_comparison is not None else None,
            "observed_description": self.observed_description,
            "candidate_descriptions": list(self.candidate_descriptions),
        }

    def to_markdown(self) -> str:
        commentary_text = self.commentary.get("commentary", {}).get("text", "")
        lines = [
            f"## Example {self.example_index}",
            f"- Selected index: {self.selected_index}",
            f"- Correct index: {self.correct_index}",
            f"- Evaluator: {self.evaluator_name}",
        ]
        if self.confidence_margin is not None:
            lines.append(
                f"- Confidence margin: {self.confidence_margin:.4f} ({self.confidence_tier or 'unknown'})"
            )
        if self.score_margin is not None:
            lines.append(f"- Score margin: {self.score_margin:.4f}")
        if self.observed_description:
            lines.append(f"- Observed: {self.observed_description}")
        if self.candidate_descriptions:
            lines.append("- Candidate descriptions:")
            for index, description in enumerate(self.candidate_descriptions):
                lines.append(f"  - {index}: {description}")
        if self.candidate_score_table:
            lines.append("- Ranked candidates:")
            for row in self.candidate_score_table:
                probability = row.get("probability")
                probability_text = (
                    f", probability={float(probability):.4f}"
                    if probability is not None
                    else ""
                )
                description = row.get("candidate_description") or "no description"
                lines.append(
                    "  - "
                    f"{row['candidate_index']}: rank={row['rank']}, score={float(row['score']):.4f}"
                    f"{probability_text}, type={row['generation_type']}, description={description}"
                )
        if self.baseline_comparison:
            lines.append("- Evaluator comparison:")
            for row in self.baseline_comparison:
                lines.append(
                    "  - "
                    f"{row.get('evaluator', 'unknown')}: selected={row.get('selected_index')}, "
                    f"correct={row.get('was_correct')}, "
                    f"score_margin={row.get('score_margin')}, "
                    f"confidence_margin={row.get('confidence_margin')}"
                )
        lines.append("")
        lines.append(str(commentary_text).strip())
        return "\n".join(lines).strip()


@dataclass(slots=True)
class LiveAgentRunResult:
    """Artifact paths and transcript records from a live agent run."""

    count: int
    records: List[LiveAgentTranscriptEntry]
    jsonl_path: Path
    markdown_path: Path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "records": [record.as_dict() for record in self.records],
            "jsonl_path": str(self.jsonl_path),
            "markdown_path": str(self.markdown_path),
        }


def run_live_agent_loop(
    dataset: Sequence[Any],
    evaluator: Any,
    *,
    count: int = 5,
    output_dir: str | Path = "results/agent_live",
    commentary_backend: Any = None,
    commentary_service: Optional[LLMCommentaryService] = None,
    include_heuristic_baseline: bool = True,
    progress: bool = True,
) -> LiveAgentRunResult:
    """Process examples one at a time and save transcript artifacts."""

    if commentary_service is not None:
        service = commentary_service
    elif commentary_backend is not None:
        service = LLMCommentaryService(backend=commentary_backend)
    else:
        service = build_default_commentary_service()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_count = min(int(count), len(dataset))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"live_agent_transcript_{timestamp}.jsonl"
    markdown_path = output_dir / f"live_agent_transcript_{timestamp}.md"

    records: List[LiveAgentTranscriptEntry] = []
    markdown_lines = ["# Live Agent Transcript", ""]

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for index in range(run_count):
            example = dataset[index]
            score_bundle = _score_example_with_evaluator(
                evaluator,
                example.observed,
                example.candidates,
                candidate_metadata=example.metadata.get("candidate_strategies"),
            )
            baseline_bundle = None
            if include_heuristic_baseline:
                baseline_bundle = run_future_selection_pipeline(
                    observed=example.observed,
                    candidates=example.candidates,
                    candidate_metadata=example.metadata.get("candidate_strategies"),
                    evaluation_mode="heuristic",
                )

            context = build_future_selection_context(example, score_bundle, baseline_bundle=baseline_bundle)
            context = _augment_context_with_descriptions(context, example.metadata)
            commentary_result = service.generate(
                context,
                expected_selected_index=int(score_bundle.selected_index),
            )

            ranking_summary = build_ranking_summary(score_bundle, correct_index=example.correct_index)
            candidate_table = build_candidate_score_table(score_bundle, correct_index=example.correct_index)
            baseline_comparison = (
                build_baseline_comparison_table(score_bundle, baseline_bundle, correct_index=example.correct_index)
                if baseline_bundle is not None
                else None
            )

            record = LiveAgentTranscriptEntry(
                example_index=index,
                correct_index=int(example.correct_index),
                selected_index=int(score_bundle.selected_index),
                evaluator_name=str(score_bundle.evaluator_name),
                confidence_margin=_confidence_margin_from_summary(ranking_summary),
                confidence_tier=_confidence_tier_from_summary(ranking_summary),
                score_margin=_score_margin_from_summary(ranking_summary),
                candidate_score_table=candidate_table,
                ranking_summary=ranking_summary,
                commentary=commentary_result.as_dict(),
                backend_name=commentary_result.backend_name,
                used_fallback=commentary_result.used_fallback,
                baseline_comparison=baseline_comparison,
                observed_description=_observed_description(example.metadata),
                candidate_descriptions=_candidate_descriptions(example.metadata),
            )
            records.append(record)
            jsonl_handle.write(json.dumps(record.as_dict(), ensure_ascii=True) + "\n")

            if progress:
                print(
                    f"[{index + 1}/{run_count}] selected={record.selected_index} "
                    f"correct={record.correct_index} backend={record.backend_name} "
                    f"confidence={record.confidence_margin if record.confidence_margin is not None else 'n/a'} "
                    f"tier={record.confidence_tier or 'unknown'}"
                )
                if record.observed_description:
                    print(f"Observed: {record.observed_description}")
                if record.candidate_descriptions:
                    print("Candidates:")
                    for candidate_index, description in enumerate(record.candidate_descriptions):
                        print(f"  - {candidate_index}: {description}")
                if record.candidate_score_table:
                    print("Ranked scores:")
                    for row in record.candidate_score_table:
                        probability = row.get("probability")
                        probability_text = (
                            f", p={float(probability):.4f}"
                            if probability is not None
                            else ""
                        )
                        print(
                            "  - "
                            f"{row['candidate_index']}: rank={row['rank']}, "
                            f"score={float(row['score']):.4f}{probability_text}, "
                            f"type={row['generation_type']}"
                        )
                if record.baseline_comparison:
                    print("Baseline comparison:")
                    for row in record.baseline_comparison:
                        print(
                            "  - "
                            f"{row.get('evaluator', 'unknown')}: selected={row.get('selected_index')}, "
                            f"correct={row.get('was_correct')}, "
                            f"score_margin={row.get('score_margin')}, "
                            f"confidence_margin={row.get('confidence_margin')}"
                        )
                print(str(record.commentary.get("commentary", {}).get("text", "")).strip())
                print("-" * 72)

            markdown_lines.append(record.to_markdown())
            markdown_lines.append("")

    markdown_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return LiveAgentRunResult(
        count=run_count,
        records=records,
        jsonl_path=jsonl_path,
        markdown_path=markdown_path,
    )


def _score_example_with_evaluator(
    evaluator: Any,
    observed: Any,
    candidates: Any,
    *,
    candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Any:
    if hasattr(evaluator, "score_example"):
        return evaluator.score_example(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )
    if hasattr(evaluator, "score_future_candidates"):
        return evaluator.score_future_candidates(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )
    if hasattr(evaluator, "evaluate"):
        return evaluator.evaluate(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )
    raise TypeError(
        "Evaluator must expose score_example(), score_future_candidates(), or evaluate()."
    )


def _augment_context_with_descriptions(
    context: Dict[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    enriched = dict(context)
    candidate_meta = list(metadata.get("candidate_strategies", []))
    enriched["candidate_descriptions"] = [
        _candidate_description(item, index)
        for index, item in enumerate(candidate_meta)
    ]
    enriched["observed_description"] = _observed_description(metadata)
    if "confidence_tier" not in enriched and "uncertainty" in enriched.get("ranking_summary", {}):
        enriched["confidence_tier"] = enriched["ranking_summary"]["uncertainty"]
    return enriched


def _candidate_descriptions(metadata: Mapping[str, Any]) -> List[str]:
    candidate_meta = list(metadata.get("candidate_strategies", []))
    return [_candidate_description(item, index) for index, item in enumerate(candidate_meta)]


def _candidate_description(candidate_metadata: Mapping[str, Any], index: int) -> str:
    description = candidate_metadata.get("description")
    if description:
        return str(description)
    label_template = candidate_metadata.get("label_template")
    if label_template:
        return str(label_template)
    strategy = candidate_metadata.get("strategy") or candidate_metadata.get("generation_type")
    return f"candidate {index}: {strategy or 'unknown'}"


def _observed_description(metadata: Mapping[str, Any]) -> str | None:
    return (
        metadata.get("observed_description")
        or metadata.get("clip_description")
        or metadata.get("description")
    )


def _score_margin_from_summary(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("score_margin", summary.get("score_gap_to_runner_up"))
    if value is None:
        return None
    return float(value)


def _confidence_margin_from_summary(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("confidence_margin", summary.get("confidence"))
    if value is None:
        return None
    return float(value)


def _confidence_tier_from_summary(summary: Mapping[str, Any]) -> str | None:
    value = summary.get("confidence_tier") or summary.get("uncertainty")
    return str(value) if value is not None else None
