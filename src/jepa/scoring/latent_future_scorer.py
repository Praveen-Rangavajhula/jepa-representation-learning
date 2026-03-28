"""Scoring wrapper for the learnable latent future predictor."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

from jepa.latent import LatentFuturePredictor

from .compatibility_metrics import rank_indices, softmax_normalize, top1_confidence_margin, uncertainty_bucket
from .vjepa_future_scorer import (
    VJEPAFutureCandidateScore,
    VJEPAFutureScoreBundle,
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


class LatentFuturePredictorScorer:
    """Rank candidate futures using a learned latent future predictor."""

    def __init__(
        self,
        predictor: LatentFuturePredictor,
        *,
        adapter: Any,
    ) -> None:
        self.predictor = predictor
        self.adapter = adapter

    def fit(self, dataset: Sequence[Any], *, max_examples: Optional[int] = None) -> Any:
        return self.predictor.fit(dataset, adapter=self.adapter, max_examples=max_examples)

    def score_future_candidates(
        self,
        observed: Tensor | Any,
        candidates: Tensor | Any,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> VJEPAFutureScoreBundle:
        return self.score_example(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )

    def score_example(
        self,
        observed: Tensor | Any,
        candidates: Tensor | Any,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> VJEPAFutureScoreBundle:
        observed_tensor = _as_tensor(observed).to(dtype=torch.float32)
        candidates_tensor = _as_tensor(candidates).to(dtype=torch.float32)
        if observed_tensor.ndim != 4:
            raise ValueError(f"Observed clip must have shape (T, C, H, W); got {tuple(observed_tensor.shape)}.")
        if candidates_tensor.ndim != 5:
            raise ValueError(
                f"Candidate futures must have shape (K, T_future, C, H, W); got {tuple(candidates_tensor.shape)}."
            )

        score_payload = self.predictor.score_candidates(
            observed_tensor,
            candidates_tensor,
            adapter=self.adapter,
        )
        scores = torch.as_tensor(score_payload["scores"], dtype=torch.float32)
        cosine_scores = torch.as_tensor(score_payload["predicted_future_cosine"], dtype=torch.float32)
        distance_scores = torch.as_tensor(score_payload["predicted_future_distance_score"], dtype=torch.float32)

        probabilities = softmax_normalize(scores)
        ordering = rank_indices(scores, descending=True)
        confidence = top1_confidence_margin(probabilities)
        uncertainty = uncertainty_bucket(confidence)
        rank_lookup = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ordering)}

        candidate_scores = []
        for index in range(candidates_tensor.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            metadata = _candidate_details(candidate_metadata, index)
            components = {
                "predicted_future_cosine": float(cosine_scores[index].item()),
                "predicted_future_distance_score": float(distance_scores[index].item()),
            }
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(scores[index].item()),
                    probability=float(probabilities[index].item()),
                    rank=rank_lookup[index],
                    components=components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(components, generation_type),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=metadata.get("details", {}),
                )
            )

        candidate_scores.sort(key=lambda item: item.score, reverse=True)
        runtime = self.adapter.describe_runtime() if hasattr(self.adapter, "describe_runtime") else {}
        selected_index = candidate_scores[0].candidate_index if candidate_scores else -1
        embedding_dim = int(score_payload["candidate_latents"].shape[-1])
        notes = [
            "Candidate futures are ranked against a predicted future latent derived from the observed prefix.",
            "The predictor head is trained on frozen V-JEPA embeddings.",
        ]
        training_summary = self.predictor.training_summary
        if training_summary is not None:
            notes.append(
                f"Predictor training used {training_summary.train_examples} train and "
                f"{training_summary.validation_examples} validation examples."
            )

        return VJEPAFutureScoreBundle(
            evaluator_name="vjepa2_latent_predictor",
            scoring_variant="predicted_future_alignment",
            model_id=str(runtime.get("model_id") or getattr(self.adapter.config, "model_id", "unknown")),
            backend_used=str(runtime.get("backend_used") or getattr(self.adapter, "backend_used", "unknown")),
            selected_index=selected_index,
            confidence=confidence,
            uncertainty=uncertainty,
            candidate_scores=candidate_scores,
            observed_shape=tuple(int(dim) for dim in observed_tensor.shape),
            candidates_shape=tuple(int(dim) for dim in candidates_tensor.shape),
            embedding_dim=embedding_dim,
            notes=notes,
        )

    @staticmethod
    def _build_rationale(components: Mapping[str, float], generation_type: str) -> str:
        dominant = max(components.items(), key=lambda item: item[1])[0]
        return (
            f"{generation_type} candidate under predicted_future_alignment; strongest learned latent "
            f"signal was {dominant} ({components[dominant]:.3f})."
        )
