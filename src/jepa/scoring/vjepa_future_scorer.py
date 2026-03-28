"""V-JEPA-backed scoring for future-selection candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .compatibility_metrics import (
    confidence_tier,
    cosine_similarity,
    latent_transition_smoothness,
    rank_indices,
    softmax_normalize,
    top1_confidence_margin,
)


def _as_tensor(value: Tensor | Any) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if value.requires_grad else value.detach()
    return torch.as_tensor(value, dtype=torch.float32)


def _candidate_generation_type(metadata: Optional[Sequence[Mapping[str, Any]]], index: int) -> str:
    if metadata is None or index >= len(metadata):
        return "unknown"
    item = metadata[index]
    generation_type = item.get("generation_type")
    if generation_type is None:
        generation_type = item.get("strategy")
    return str(generation_type) if generation_type is not None else "unknown"


def _candidate_details(metadata: Optional[Sequence[Mapping[str, Any]]], index: int) -> Dict[str, Any]:
    if metadata is None or index >= len(metadata):
        return {}
    return dict(metadata[index])


@dataclass(slots=True)
class VJEPAFutureCandidateScore:
    candidate_index: int
    score: float
    probability: float
    rank: int
    components: Dict[str, float]
    generation_type: str = "unknown"
    rationale: str = ""
    source_index: Optional[int] = None
    is_true: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "score": self.score,
            "probability": self.probability,
            "rank": self.rank,
            "components": dict(self.components),
            "generation_type": self.generation_type,
            "rationale": self.rationale,
            "source_index": self.source_index,
            "is_true": self.is_true,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class VJEPAFutureScoreBundle:
    evaluator_name: str
    scoring_variant: str
    model_id: str
    backend_used: str
    selected_index: int
    confidence: float
    confidence_tier: str
    candidate_scores: List[VJEPAFutureCandidateScore]
    observed_shape: Tuple[int, ...]
    candidates_shape: Tuple[int, ...]
    embedding_dim: int
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "scoring_variant": self.scoring_variant,
            "model_id": self.model_id,
            "backend_used": self.backend_used,
            "selected_index": self.selected_index,
            "confidence": self.confidence,
            "confidence_tier": self.confidence_tier,
            "candidate_scores": [item.as_dict() for item in self.candidate_scores],
            "observed_shape": list(self.observed_shape),
            "candidates_shape": list(self.candidates_shape),
            "embedding_dim": self.embedding_dim,
            "notes": list(self.notes),
        }

    @property
    def uncertainty(self) -> str:
        """Backward-compatible alias for older callers."""

        return self.confidence_tier


class VJEPAFutureScorer:
    """Score candidate futures with a V-JEPA 2 encoder backbone."""

    def __init__(self, adapter: Optional[Any] = None, *, scoring_variant: str = "overlap_transition") -> None:
        if adapter is None:
            from jepa.models import VJEPA2Adapter

            adapter = VJEPA2Adapter()
        self.adapter = adapter
        self.scoring_variant = scoring_variant

    def score_context_future(
        self,
        observed: Tensor | Any,
        candidate: Tensor | Any,
        _plan: Optional[Mapping[str, Any]] = None,
    ) -> float:
        candidate_batch = _as_tensor(candidate)
        if candidate_batch.ndim == 4:
            candidate_batch = candidate_batch.unsqueeze(0)
        bundle = self.score_example(observed, candidate_batch, candidate_metadata=None, scoring_variant=self.scoring_variant)
        return float(bundle.candidate_scores[0].score)

    def score_future_candidates(
        self,
        observed: Tensor | Any,
        candidates: Tensor | Any,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        scoring_variant: Optional[str] = None,
    ) -> VJEPAFutureScoreBundle:
        return self.score_example(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
            scoring_variant=scoring_variant or self.scoring_variant,
        )

    def score_example(
        self,
        observed: Tensor | Any,
        candidates: Tensor | Any,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        scoring_variant: Optional[str] = None,
    ) -> VJEPAFutureScoreBundle:
        observed_tensor = _as_tensor(observed).to(dtype=torch.float32)
        candidates_tensor = _as_tensor(candidates).to(dtype=torch.float32)

        if observed_tensor.ndim != 4:
            raise ValueError(f"Observed clip must have shape (T, C, H, W); got {tuple(observed_tensor.shape)}.")
        if candidates_tensor.ndim != 5:
            raise ValueError(
                f"Candidate futures must have shape (K, T_future, C, H, W); got {tuple(candidates_tensor.shape)}."
            )
        if candidates_tensor.shape[0] < 1:
            raise ValueError("At least one candidate future is required.")

        variant = scoring_variant or self.scoring_variant
        if variant not in {"overlap_transition", "prefix_future_cosine"}:
            raise ValueError(f"Unsupported V-JEPA scoring variant: {variant}")

        if variant == "prefix_future_cosine":
            candidate_scores, embedding_dim = self._score_prefix_future_cosine(
                observed_tensor,
                candidates_tensor,
                candidate_metadata=candidate_metadata,
            )
        else:
            candidate_scores, embedding_dim = self._score_overlap_transition(
                observed_tensor,
                candidates_tensor,
                candidate_metadata=candidate_metadata,
            )

        scores_tensor = torch.tensor([item.score for item in candidate_scores], dtype=torch.float32)
        probabilities = softmax_normalize(scores_tensor)
        ordering = rank_indices(scores_tensor, descending=True)
        confidence = top1_confidence_margin(probabilities)
        tier = confidence_tier(confidence)

        rank_lookup = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ordering)}
        for item in candidate_scores:
            item.rank = rank_lookup[item.candidate_index]
            item.probability = float(probabilities[item.candidate_index].item())

        candidate_scores.sort(key=lambda item: item.score, reverse=True)
        selected_index = candidate_scores[0].candidate_index if candidate_scores else -1

        runtime = self.adapter.describe_runtime() if hasattr(self.adapter, "describe_runtime") else {}
        notes = [
            "Combined 16-frame task clips are internally resampled to the V-JEPA frame count.",
            "Single-channel clips are converted to RGB inside preprocessing when needed.",
        ]

        return VJEPAFutureScoreBundle(
            evaluator_name=f"vjepa2_{variant}",
            scoring_variant=variant,
            model_id=str(runtime.get("model_id") or getattr(self.adapter.config, "model_id", "unknown")),
            backend_used=str(runtime.get("backend_used") or getattr(self.adapter, "backend_used", "unknown")),
            selected_index=selected_index,
            confidence=confidence,
            confidence_tier=tier,
            candidate_scores=candidate_scores,
            observed_shape=tuple(int(dim) for dim in observed_tensor.shape),
            candidates_shape=tuple(int(dim) for dim in candidates_tensor.shape),
            embedding_dim=embedding_dim,
            notes=notes,
        )

    def _score_prefix_future_cosine(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[List[VJEPAFutureCandidateScore], int]:
        observed_embedding = self.adapter.encode_video(observed)
        candidate_embeddings = self.adapter.encode_batch(candidates, batch_size=4)
        cosine_scores = cosine_similarity(candidate_embeddings, observed_embedding.unsqueeze(0))

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(candidates.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            details = _candidate_details(candidate_metadata, index)
            component_value = float(cosine_scores[index].item())
            components = {"prefix_future_cosine": component_value}
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=component_value,
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(components, generation_type, "prefix_future_cosine"),
                    source_index=details.get("source_index"),
                    is_true=details.get("is_true"),
                    details=details.get("details", {}),
                )
            )

        return candidate_scores, int(candidate_embeddings.shape[-1])

    def _score_overlap_transition(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[List[VJEPAFutureCandidateScore], int]:
        combined_clips = torch.stack([torch.cat([observed, candidate], dim=0) for candidate in candidates], dim=0)
        segment_length = int(observed.shape[0])
        starts = (0, max(0, segment_length // 2), segment_length)

        segments: List[Tensor] = []
        for clip in combined_clips:
            for start in starts:
                stop = start + segment_length
                segments.append(clip[start:stop])

        embeddings = self.adapter.encode_batch(torch.stack(segments, dim=0), batch_size=4)
        embedding_dim = int(embeddings.shape[-1])
        embeddings = embeddings.view(combined_clips.shape[0], len(starts), embedding_dim)

        z_a = embeddings[:, 0]
        z_b = embeddings[:, 1]
        z_c = embeddings[:, 2]

        cosine_ab = cosine_similarity(z_a, z_b)
        cosine_bc = cosine_similarity(z_b, z_c)
        smoothness = latent_transition_smoothness(z_a, z_b, z_c)
        total_scores = (0.4 * cosine_ab) + (0.4 * cosine_bc) + (0.2 * smoothness)

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(candidates.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            metadata = _candidate_details(candidate_metadata, index)
            components = {
                "cosine_a_b": float(cosine_ab[index].item()),
                "cosine_b_c": float(cosine_bc[index].item()),
                "transition_smoothness": float(smoothness[index].item()),
            }
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(total_scores[index].item()),
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(components, generation_type, "overlap_transition"),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=metadata.get("details", {}),
                )
            )

        return candidate_scores, embedding_dim

    @staticmethod
    def _build_rationale(components: Mapping[str, float], generation_type: str, variant: str) -> str:
        dominant = max(components.items(), key=lambda item: item[1])[0]
        return (
            f"{generation_type} candidate under {variant}; strongest latent component was "
            f"{dominant} ({components[dominant]:.3f})."
        )
