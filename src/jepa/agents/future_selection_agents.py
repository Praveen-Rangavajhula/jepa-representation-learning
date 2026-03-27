"""Agent pipeline for the future-selection task.

Inputs:
- observed clip: ``(T_obs, C, H, W)``
- candidate futures: ``(K, T_future, C, H, W)``

The pipeline supports three evaluator modes:
- ``heuristic`` for the original geometric baseline
- ``representation_only`` for V-JEPA-backed scoring
- ``hybrid`` for optional additive comparisons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Any


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Convert common tensor-like inputs to a NumPy array."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    return np.asarray(value)


def _validate_inputs(observed: np.ndarray, candidates: np.ndarray) -> None:
    if observed.ndim != 4:
        raise ValueError(
            f"Observed clip must have shape (T_obs, C, H, W); received {tuple(observed.shape)}"
        )
    if candidates.ndim != 5:
        raise ValueError(
            f"Candidate futures must have shape (K, T_future, C, H, W); received {tuple(candidates.shape)}"
        )
    if observed.shape[1:] != candidates.shape[2:]:
        raise ValueError(
            "Observed and candidate clips must share channel and spatial dimensions."
        )
    if candidates.shape[0] < 1:
        raise ValueError("At least one candidate future is required.")
    if observed.shape[0] < 2 or candidates.shape[1] < 2:
        raise ValueError("Observed and candidate clips must each contain at least 2 frames.")


def _frame_centroid(frame: np.ndarray) -> np.ndarray:
    """Estimate the brightness centroid of a single frame."""

    frame_2d = np.asarray(frame, dtype=np.float64)
    if frame_2d.ndim == 3:
        frame_2d = frame_2d.mean(axis=0)
    total = float(frame_2d.sum())
    height, width = frame_2d.shape[-2], frame_2d.shape[-1]
    if total <= 1e-8:
        return np.asarray([height / 2.0, width / 2.0], dtype=np.float64)

    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    row_mass = frame_2d.sum(axis=-1)
    col_mass = frame_2d.sum(axis=-2)
    row = float((rows * row_mass).sum() / total)
    col = float((cols * col_mass).sum() / total)
    return np.asarray([row, col], dtype=np.float64)


def _trajectory(frames: np.ndarray) -> Dict[str, np.ndarray]:
    centroids = np.stack([_frame_centroid(frame) for frame in frames], axis=0)
    velocities = np.diff(centroids, axis=0)
    speeds = np.linalg.norm(velocities, axis=-1)
    return {
        "centroids": centroids,
        "velocities": velocities,
        "speeds": speeds,
    }


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = _safe_norm(a) * _safe_norm(b)
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _candidate_generation_type(metadata: Optional[Sequence[Mapping[str, Any]]], index: int) -> str:
    if metadata is None or index >= len(metadata):
        return "unknown"
    item = metadata[index]
    generation_type = None
    if isinstance(item, Mapping):
        generation_type = item.get("generation_type")
        if generation_type is None:
            generation_type = item.get("strategy")
    return str(generation_type) if generation_type is not None else "unknown"


@dataclass(slots=True)
class TaskPlan:
    """Lightweight plan describing the future-selection task."""

    observed_shape: Tuple[int, ...]
    candidate_shape: Tuple[int, ...]
    context_frames: int
    future_frames: int
    candidate_count: int
    evaluator_mode: str = "heuristic"
    candidate_generation_types: Tuple[str, ...] = ()
    heuristic_weights: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observed_shape": list(self.observed_shape),
            "candidate_shape": list(self.candidate_shape),
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "candidate_count": self.candidate_count,
            "evaluator_mode": self.evaluator_mode,
            "candidate_generation_types": list(self.candidate_generation_types),
            "heuristic_weights": dict(self.heuristic_weights),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class CandidateEvaluation:
    """Score and diagnostics for a single candidate future."""

    candidate_index: int
    score: float
    components: Dict[str, float]
    generation_type: str = "unknown"
    rationale: str = ""
    evaluator_name: str = "heuristic"
    confidence: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "score": self.score,
            "components": dict(self.components),
            "generation_type": self.generation_type,
            "rationale": self.rationale,
            "evaluator_name": self.evaluator_name,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CritiqueMessage:
    """Structured critique for a candidate future."""

    candidate_index: int
    severity: str
    message: str
    generation_type: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "severity": self.severity,
            "message": self.message,
            "generation_type": self.generation_type,
        }


@dataclass(slots=True)
class PipelineTraceEvent:
    """Structured trace event emitted by the pipeline."""

    stage: str
    message: str
    candidate_index: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "candidate_index": self.candidate_index,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class FutureSelectionResult:
    """Final output of the future-selection pipeline."""

    selected_index: int
    ranked_candidates: List[CandidateEvaluation]
    critique_messages: List[CritiqueMessage]
    trace: List[PipelineTraceEvent]
    plan: TaskPlan

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "ranked_candidates": [item.as_dict() for item in self.ranked_candidates],
            "critique_messages": [item.as_dict() for item in self.critique_messages],
            "trace": [item.as_dict() for item in self.trace],
            "plan": self.plan.as_dict(),
        }


class PlannerAgent:
    """Build a lightweight task plan from shapes and optional metadata."""

    def __init__(self, *, context_frames: int = 8, future_frames: int = 8) -> None:
        self.context_frames = context_frames
        self.future_frames = future_frames

    def plan(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        representation_model: Any = None,
        evaluation_mode: str = "heuristic",
    ) -> TaskPlan:
        observed_np = _to_numpy(observed)
        candidates_np = _to_numpy(candidates)
        _validate_inputs(observed_np, candidates_np)

        generation_types = tuple(
            _candidate_generation_type(candidate_metadata, index)
            for index in range(candidates_np.shape[0])
        )

        base_weights = {
            "continuity": 0.26,
            "extrapolation": 0.26,
            "direction": 0.18,
            "speed": 0.15,
            "smoothness": 0.15,
        }
        if evaluation_mode == "representation_only":
            heuristic_weights = {name: 0.0 for name in base_weights}
            notes = [
                "Observed clip and candidate futures are compared with V-JEPA latent compatibility scores.",
            ]
        elif evaluation_mode == "hybrid":
            heuristic_weights = dict(base_weights)
            notes = [
                "Observed clip and candidate futures are scored with both motion heuristics and V-JEPA latent compatibility.",
            ]
        else:
            heuristic_weights = dict(base_weights)
            notes = [
                "Observed clip is treated as context and candidate futures are compared against its motion profile.",
            ]

        if representation_model is not None:
            notes.append("A representation model was supplied to the evaluator.")

        return TaskPlan(
            observed_shape=tuple(observed_np.shape),
            candidate_shape=tuple(candidates_np.shape),
            context_frames=int(observed_np.shape[0]),
            future_frames=int(candidates_np.shape[1]),
            candidate_count=int(candidates_np.shape[0]),
            evaluator_mode=evaluation_mode,
            candidate_generation_types=generation_types,
            heuristic_weights=heuristic_weights,
            notes=notes,
        )


class EvaluatorAgent:
    """Rank candidates using a heuristic baseline, V-JEPA scorer, or both."""

    def __init__(self, *, representation_model: Any = None, mode: str = "heuristic") -> None:
        self.representation_model = representation_model
        self.mode = mode

    def evaluate(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        plan: Optional[TaskPlan] = None,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[CandidateEvaluation]:
        observed_np = _to_numpy(observed)
        candidates_np = _to_numpy(candidates)
        _validate_inputs(observed_np, candidates_np)

        if plan is None:
            plan = PlannerAgent().plan(
                observed_np,
                candidates_np,
                candidate_metadata=candidate_metadata,
                representation_model=self.representation_model,
                evaluation_mode=self.mode,
            )

        representation_bundle = None
        representation_lookup: Dict[int, Any] = {}
        if self.mode in {"representation_only", "hybrid"}:
            representation_bundle = self._representation_bundle(
                observed,
                candidates,
                candidate_metadata=candidate_metadata,
            )
            if representation_bundle is None:
                raise ValueError(
                    "EvaluatorAgent in representation_only/hybrid mode requires a representation model "
                    "that exposes score_future_candidates() or score_example()."
                )
            for item in representation_bundle.candidate_scores:
                representation_lookup[int(item.candidate_index)] = item

        heuristic_lookup = self._heuristic_components_by_candidate(observed_np, candidates_np)
        evaluations: List[CandidateEvaluation] = []
        for index in range(candidates_np.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            heuristic_components = heuristic_lookup[index]
            heuristic_score = self._combine_scores(plan.heuristic_weights, heuristic_components)
            rep_item = representation_lookup.get(index)

            if self.mode == "heuristic":
                total_score = heuristic_score
                components = dict(heuristic_components)
                rationale = self._build_heuristic_rationale(heuristic_components, generation_type)
                evaluator_name = "heuristic"
                confidence = None
            elif self.mode == "representation_only":
                if rep_item is None or representation_bundle is None:
                    raise ValueError("Missing representation scores for representation_only evaluation.")
                total_score = float(rep_item.score)
                components = dict(rep_item.components)
                rationale = rep_item.rationale
                evaluator_name = getattr(representation_bundle, "evaluator_name", "representation")
                confidence = getattr(representation_bundle, "confidence", None)
            elif self.mode == "hybrid":
                if rep_item is None or representation_bundle is None:
                    raise ValueError("Missing representation scores for hybrid evaluation.")
                total_score = float(heuristic_score + rep_item.score)
                components = {
                    **{f"heuristic_{key}": float(value) for key, value in heuristic_components.items()},
                    **{f"vjepa_{key}": float(value) for key, value in rep_item.components.items()},
                    "heuristic_total": float(heuristic_score),
                    "vjepa_total": float(rep_item.score),
                }
                rationale = (
                    f"{generation_type} candidate under hybrid scoring; heuristic={heuristic_score:.3f}, "
                    f"vjepa={rep_item.score:.3f}."
                )
                evaluator_name = f"hybrid::{getattr(representation_bundle, 'evaluator_name', 'representation')}"
                confidence = getattr(representation_bundle, "confidence", None)
            else:
                raise ValueError(f"Unsupported evaluator mode: {self.mode}")

            evaluations.append(
                CandidateEvaluation(
                    candidate_index=index,
                    score=float(total_score),
                    components=components,
                    generation_type=generation_type,
                    rationale=rationale,
                    evaluator_name=evaluator_name,
                    confidence=confidence,
                )
            )

        evaluations.sort(key=lambda item: item.score, reverse=True)
        return evaluations

    def _representation_bundle(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> Any:
        model = self.representation_model
        if model is None:
            return None
        if hasattr(model, "score_future_candidates"):
            return model.score_future_candidates(
                observed,
                candidates,
                candidate_metadata=candidate_metadata,
            )
        if hasattr(model, "score_example"):
            return model.score_example(
                observed,
                candidates,
                candidate_metadata=candidate_metadata,
            )
        return None

    def _heuristic_components_by_candidate(
        self,
        observed: np.ndarray,
        candidates: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        observed_traj = _trajectory(observed)
        observed_last_position = observed_traj["centroids"][-1]
        observed_last_velocity = observed_traj["velocities"][-1]
        expected_next_position = observed_last_position + observed_last_velocity
        observed_speed = _mean_or_zero(observed_traj["speeds"][-3:])

        lookup: Dict[int, Dict[str, float]] = {}
        for index, candidate in enumerate(candidates):
            candidate_traj = _trajectory(candidate)
            first_position = candidate_traj["centroids"][0]
            first_velocity = candidate_traj["velocities"][0]
            candidate_speed = _mean_or_zero(candidate_traj["speeds"][:3])
            candidate_smoothness = float(np.std(candidate_traj["velocities"], axis=0).mean())

            continuity = 1.0 / (1.0 + float(np.linalg.norm(first_position - observed_last_position)))
            extrapolation = 1.0 / (1.0 + float(np.linalg.norm(first_position - expected_next_position)))
            direction = (_safe_cosine_similarity(observed_last_velocity, first_velocity) + 1.0) / 2.0
            speed = 1.0 / (1.0 + abs(observed_speed - candidate_speed))
            smoothness = 1.0 / (1.0 + candidate_smoothness)

            lookup[index] = {
                "continuity": float(continuity),
                "extrapolation": float(extrapolation),
                "direction": float(direction),
                "speed": float(speed),
                "smoothness": float(smoothness),
            }
        return lookup

    @staticmethod
    def _combine_scores(weights: Mapping[str, float], components: Mapping[str, float]) -> float:
        return float(sum(float(weights.get(name, 0.0)) * float(components.get(name, 0.0)) for name in components))

    @staticmethod
    def _build_heuristic_rationale(components: Mapping[str, float], generation_type: str) -> str:
        dominant = max(components.items(), key=lambda item: item[1])[0]
        return (
            f"{generation_type} candidate; strongest heuristic was {dominant} "
            f"({components[dominant]:.3f})."
        )


class CriticAgent:
    """Produce human-readable critiques for ranked candidates."""

    def critique(
        self,
        ranked_candidates: Sequence[CandidateEvaluation],
        plan: Optional[TaskPlan] = None,
    ) -> List[CritiqueMessage]:
        critiques: List[CritiqueMessage] = []
        if not ranked_candidates:
            return critiques

        best = ranked_candidates[0]
        evidence_label = "latent compatibility" if best.evaluator_name != "heuristic" else "motion consistency"
        critiques.append(
            CritiqueMessage(
                candidate_index=best.candidate_index,
                severity="info",
                message=(
                    f"Candidate {best.candidate_index} is the strongest continuation candidate "
                    f"with score {best.score:.3f} under {evidence_label}."
                ),
                generation_type=best.generation_type,
            )
        )

        for item in ranked_candidates[1:]:
            gap = best.score - item.score
            if gap < 0.05:
                severity = "warning"
                message = (
                    f"Candidate {item.candidate_index} is close to the leader but still loses on "
                    f"{evidence_label}; inspect the strongest scoring components."
                )
            elif item.generation_type not in {"true", "true_continuation", "ground_truth"}:
                severity = "warning"
                message = (
                    f"Candidate {item.candidate_index} looks plausible but its {item.generation_type} "
                    f"construction is less consistent under {item.evaluator_name}."
                )
            else:
                severity = "info"
                message = (
                    f"Candidate {item.candidate_index} is lower ranked because its active evaluator "
                    f"signals are weaker."
                )

            if plan is not None and plan.notes:
                message = f"{message} Plan note: {plan.notes[0]}"

            critiques.append(
                CritiqueMessage(
                    candidate_index=item.candidate_index,
                    severity=severity,
                    message=message,
                    generation_type=item.generation_type,
                )
            )

        return critiques


class ExecutiveAgent:
    """Coordinate planning, evaluation, critique, and selection."""

    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        evaluator: Optional[EvaluatorAgent] = None,
        critic: Optional[CriticAgent] = None,
    ) -> None:
        self.planner = planner or PlannerAgent()
        self.evaluator = evaluator or EvaluatorAgent()
        self.critic = critic or CriticAgent()

    def select(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        representation_model: Any = None,
        evaluation_mode: Optional[str] = None,
    ) -> FutureSelectionResult:
        trace: List[PipelineTraceEvent] = []

        active_mode = evaluation_mode or self.evaluator.mode
        plan = self.planner.plan(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
            representation_model=representation_model,
            evaluation_mode=active_mode,
        )
        trace.append(
            PipelineTraceEvent(
                stage="plan",
                message="Built future-selection plan.",
                payload=plan.as_dict(),
            )
        )

        if representation_model is not None:
            self.evaluator.representation_model = representation_model
        self.evaluator.mode = active_mode

        ranked = self.evaluator.evaluate(
            observed,
            candidates,
            plan=plan,
            candidate_metadata=candidate_metadata,
        )
        trace.append(
            PipelineTraceEvent(
                stage="evaluate",
                message="Ranked candidate futures.",
                payload={
                    "evaluation_mode": active_mode,
                    "ranked_candidates": [item.as_dict() for item in ranked],
                },
            )
        )

        critiques = self.critic.critique(ranked, plan=plan)
        trace.append(
            PipelineTraceEvent(
                stage="critique",
                message="Generated critique messages.",
                payload={"critique_messages": [item.as_dict() for item in critiques]},
            )
        )

        selected_index = ranked[0].candidate_index if ranked else -1
        trace.append(
            PipelineTraceEvent(
                stage="select",
                message="Selected final candidate.",
                candidate_index=selected_index,
                payload={
                    "selected_index": selected_index,
                    "selected_score": ranked[0].score if ranked else None,
                    "evaluation_mode": active_mode,
                },
            )
        )

        return FutureSelectionResult(
            selected_index=selected_index,
            ranked_candidates=list(ranked),
            critique_messages=list(critiques),
            trace=trace,
            plan=plan,
        )

    def run(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        representation_model: Any = None,
        evaluation_mode: Optional[str] = None,
    ) -> FutureSelectionResult:
        """Alias for select(), useful for notebook and pipeline calls."""

        return self.select(
            observed=observed,
            candidates=candidates,
            candidate_metadata=candidate_metadata,
            representation_model=representation_model,
            evaluation_mode=evaluation_mode,
        )

    def execute(
        self,
        observed: ArrayLike,
        candidates: ArrayLike,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        representation_model: Any = None,
        evaluation_mode: Optional[str] = None,
    ) -> FutureSelectionResult:
        """Alias for select(), mirroring common pipeline terminology."""

        return self.select(
            observed=observed,
            candidates=candidates,
            candidate_metadata=candidate_metadata,
            representation_model=representation_model,
            evaluation_mode=evaluation_mode,
        )


def run_future_selection_pipeline(
    observed: ArrayLike,
    candidates: ArrayLike,
    candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    representation_model: Any = None,
    evaluation_mode: str = "heuristic",
) -> FutureSelectionResult:
    """Convenience wrapper for notebook and script use."""

    executive = ExecutiveAgent(
        evaluator=EvaluatorAgent(
            mode=evaluation_mode,
            representation_model=representation_model,
        )
    )
    return executive.select(
        observed=observed,
        candidates=candidates,
        candidate_metadata=candidate_metadata,
        representation_model=representation_model,
        evaluation_mode=evaluation_mode,
    )
