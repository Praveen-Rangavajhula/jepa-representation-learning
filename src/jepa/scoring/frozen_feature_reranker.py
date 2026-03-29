"""Tiny learned reranker on top of frozen heuristic and V-JEPA evaluator features."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from jepa.agents import run_future_selection_pipeline

from .compatibility_metrics import confidence_tier, rank_indices, softmax_normalize, top1_confidence_margin
from .vjepa_future_scorer import VJEPAFutureCandidateScore, VJEPAFutureScoreBundle


def _as_tensor(value: Tensor | Any) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if value.requires_grad else value.detach()
    return torch.as_tensor(value, dtype=torch.float32)


def _safe_key(name: str) -> str:
    cleaned = []
    for character in str(name).strip().lower():
        cleaned.append(character if character.isalnum() else "_")
    collapsed = "".join(cleaned)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "unnamed"


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
class FrozenFeatureRerankerConfig:
    """Configuration for the tiny frozen-feature reranker."""

    model_kind: str = "linear"
    include_heuristic_features: bool = True
    include_candidate_type_indicators: bool = False
    strict_split: bool = True
    allow_provisional_overlap_split: bool = False
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 16
    validation_fraction: float = 0.2
    max_training_examples: int = 128
    mlp_hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 23
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    cache_version: str = "frozen_feature_reranker_v2"

    def validate(self) -> "FrozenFeatureRerankerConfig":
        if self.model_kind not in {"linear", "mlp"}:
            raise ValueError("model_kind must be 'linear' or 'mlp'.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must satisfy 0 <= value < 1.")
        if self.max_training_examples < 1:
            raise ValueError("max_training_examples must be at least 1.")
        if self.mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= value < 1.")
        return self

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "include_heuristic_features": self.include_heuristic_features,
            "include_candidate_type_indicators": self.include_candidate_type_indicators,
            "strict_split": self.strict_split,
            "allow_provisional_overlap_split": self.allow_provisional_overlap_split,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
            "max_training_examples": self.max_training_examples,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout": self.dropout,
            "seed": self.seed,
            "device": str(self.resolved_device()),
            "cache_dir": self.cache_dir,
            "cache_version": self.cache_version,
        }


@dataclass(slots=True)
class FrozenFeatureCache:
    """Cached candidate-level feature matrix for reranker training or evaluation."""

    split_name: str
    feature_names: List[str]
    features: Tensor
    labels: Tensor
    source_video_ids: List[str]
    example_payloads: List[Dict[str, Any]]
    cache_hit: bool = False
    provisional: bool = False
    cache_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "feature_names": list(self.feature_names),
            "features_shape": list(self.features.shape),
            "labels_shape": list(self.labels.shape),
            "source_video_ids": list(self.source_video_ids),
            "example_payloads": list(self.example_payloads),
            "cache_hit": self.cache_hit,
            "provisional": self.provisional,
            "cache_path": self.cache_path,
        }


@dataclass(slots=True)
class FrozenFeatureRerankerTrainingSummary:
    """Compact training summary for the tiny reranker."""

    model_kind: str
    feature_dim: int
    train_examples: int
    validation_examples: int
    epochs: int
    best_epoch: int
    final_train_loss: float
    best_validation_loss: float
    provisional: bool = False
    feature_cache_hit: bool = False
    feature_extraction_seconds: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    feature_names: List[str] = field(default_factory=list)
    linear_weight_preview: Dict[str, float] = field(default_factory=dict)
    loss_history: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "feature_dim": self.feature_dim,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.final_train_loss,
            "best_validation_loss": self.best_validation_loss,
            "provisional": self.provisional,
            "feature_cache_hit": self.feature_cache_hit,
            "feature_extraction_seconds": self.feature_extraction_seconds,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
            "feature_names": list(self.feature_names),
            "linear_weight_preview": dict(self.linear_weight_preview),
            "loss_history": list(self.loss_history),
        }


class _LinearRerankerNetwork(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.layer(features).squeeze(-1)


class _TinyMLPRerankerNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)


class FrozenFeatureRerankerScorer:
    """Tiny learned reranker built on top of frozen heuristic and V-JEPA features."""

    def __init__(
        self,
        config: Optional[FrozenFeatureRerankerConfig] = None,
        *,
        masked_only_scorer: Any,
        boundary_hybrid_scorer: Any,
    ) -> None:
        self.config = (config or FrozenFeatureRerankerConfig()).validate()
        self.masked_only_scorer = masked_only_scorer
        self.boundary_hybrid_scorer = boundary_hybrid_scorer
        self.device = self.config.resolved_device()
        self.model: Optional[nn.Module] = None
        self.feature_names: List[str] = []
        self.feature_mean: Optional[Tensor] = None
        self.feature_std: Optional[Tensor] = None
        self.training_summary: Optional[FrozenFeatureRerankerTrainingSummary] = None
        self._feature_cache_memory: Dict[str, FrozenFeatureCache] = {}

    def feature_spec(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "config": self.config.as_dict(),
            "trained": self.model is not None and self.feature_mean is not None and self.feature_std is not None,
        }

    def fit(
        self,
        dataset: Sequence[Any],
        *,
        max_examples: Optional[int] = None,
        split_name: str = "train",
        provisional: bool = False,
        force_rebuild_cache: bool = False,
    ) -> FrozenFeatureRerankerTrainingSummary:
        total_start = time.perf_counter()
        feature_start = time.perf_counter()
        feature_cache = self.build_feature_cache(
            dataset,
            split_name=split_name,
            max_examples=max_examples,
            provisional=provisional,
            force_rebuild=force_rebuild_cache,
        )
        feature_extraction_seconds = float(time.perf_counter() - feature_start)
        features = feature_cache.features
        labels = feature_cache.labels
        if features.shape[0] < 2:
            raise ValueError("Need at least two examples to train the frozen-feature reranker.")

        feature_dim = int(features.shape[-1])
        self.feature_names = list(feature_cache.feature_names)
        self._ensure_model(feature_dim)

        generator = torch.Generator().manual_seed(self.config.seed)
        permutation = torch.randperm(features.shape[0], generator=generator)
        validation_examples = int(round(features.shape[0] * self.config.validation_fraction))
        validation_examples = min(max(validation_examples, 1), int(features.shape[0] - 1))
        train_examples = int(features.shape[0] - validation_examples)

        train_indices = permutation[:train_examples]
        validation_indices = permutation[train_examples:]
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        validation_features = features[validation_indices]
        validation_labels = labels[validation_indices]

        self.feature_mean = train_features.reshape(-1, feature_dim).mean(dim=0)
        self.feature_std = train_features.reshape(-1, feature_dim).std(dim=0, unbiased=False).clamp_min(1e-6)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_state: Optional[Dict[str, Tensor]] = None
        best_validation_loss = float("inf")
        best_epoch = 0
        history: List[Dict[str, float]] = []

        training_start = time.perf_counter()
        self.model.train()
        for epoch in range(self.config.epochs):
            order = torch.randperm(train_examples, generator=generator)
            running_loss = 0.0
            batch_count = 0
            for start in range(0, train_examples, self.config.batch_size):
                batch_indices = order[start : start + self.config.batch_size]
                batch_features = train_features[batch_indices]
                batch_labels = train_labels[batch_indices]
                normalized = self._normalize_features(batch_features).to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(normalized)
                loss = F.cross_entropy(logits, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batch_count += 1

            train_loss = running_loss / max(batch_count, 1)
            validation_loss = self._evaluate_loss(validation_features, validation_labels)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(train_loss),
                    "validation_loss": float(validation_loss),
                }
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_epoch = int(epoch + 1)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        training_seconds = float(time.perf_counter() - training_start)
        total_seconds = float(time.perf_counter() - total_start)

        self.training_summary = FrozenFeatureRerankerTrainingSummary(
            model_kind=self.config.model_kind,
            feature_dim=feature_dim,
            train_examples=train_examples,
            validation_examples=validation_examples,
            epochs=self.config.epochs,
            best_epoch=best_epoch,
            final_train_loss=float(history[-1]["train_loss"]),
            best_validation_loss=best_validation_loss,
            provisional=provisional,
            feature_cache_hit=feature_cache.cache_hit,
            feature_extraction_seconds=feature_extraction_seconds,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            feature_names=list(self.feature_names),
            linear_weight_preview=self._linear_weight_preview(),
            loss_history=history,
        )
        return self.training_summary

    def build_feature_cache(
        self,
        dataset: Sequence[Any],
        *,
        split_name: str,
        max_examples: Optional[int] = None,
        provisional: bool = False,
        force_rebuild: bool = False,
    ) -> FrozenFeatureCache:
        count = min(len(dataset), int(max_examples or self.config.max_training_examples or len(dataset)))
        source_video_ids = self._source_video_ids(dataset, count)

        if not force_rebuild:
            cached = self._load_cached_feature_matrix(split_name, source_video_ids, provisional=provisional)
            if cached is not None:
                self.feature_names = list(cached.feature_names)
                self._feature_cache_memory[split_name] = cached
                return cached

        example_feature_maps: List[List[Dict[str, float]]] = []
        example_payloads: List[Dict[str, Any]] = []
        labels: List[int] = []
        feature_names: set[str] = set()

        for index in range(count):
            example = dataset[index]
            payload = self._build_feature_payload_for_example(example, example_index=index)
            example_feature_maps.append(payload["candidate_feature_maps"])
            example_payloads.append(payload["cache_payload"])
            labels.append(int(example.correct_index))
            for candidate_map in payload["candidate_feature_maps"]:
                feature_names.update(candidate_map.keys())

        ordered_feature_names = sorted(feature_names)
        candidate_count = len(example_feature_maps[0]) if example_feature_maps else 0
        feature_tensor = torch.zeros((count, candidate_count, len(ordered_feature_names)), dtype=torch.float32)
        for example_index, candidate_maps in enumerate(example_feature_maps):
            for candidate_index, candidate_map in enumerate(candidate_maps):
                for feature_index, feature_name in enumerate(ordered_feature_names):
                    feature_tensor[example_index, candidate_index, feature_index] = float(
                        candidate_map.get(feature_name, 0.0)
                    )

        cache = FrozenFeatureCache(
            split_name=split_name,
            feature_names=ordered_feature_names,
            features=feature_tensor,
            labels=torch.tensor(labels, dtype=torch.long),
            source_video_ids=source_video_ids,
            example_payloads=example_payloads,
            cache_hit=False,
            provisional=provisional,
            cache_path=str(self._cache_path(split_name)) if self._cache_path(split_name) is not None else None,
        )
        self.feature_names = list(ordered_feature_names)
        self._feature_cache_memory[split_name] = cache
        self._save_feature_cache(cache)
        return cache

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
        self._require_trained_model()
        observed_tensor = _as_tensor(observed).to(dtype=torch.float32)
        candidates_tensor = _as_tensor(candidates).to(dtype=torch.float32)
        if observed_tensor.ndim != 4:
            raise ValueError(f"Observed clip must have shape (T, C, H, W); got {tuple(observed_tensor.shape)}.")
        if candidates_tensor.ndim != 5:
            raise ValueError(
                f"Candidate futures must have shape (K, T_future, C, H, W); got {tuple(candidates_tensor.shape)}."
            )

        payload = self._build_feature_payload(
            observed_tensor,
            candidates_tensor,
            candidate_metadata=candidate_metadata,
        )
        feature_maps = payload["candidate_feature_maps"]
        features = self._feature_tensor_from_maps(feature_maps)
        normalized = self._normalize_features(features.unsqueeze(0)).to(self.device)
        with torch.no_grad():
            logits = self.model(normalized).squeeze(0).detach().cpu()

        probabilities = softmax_normalize(logits)
        ordering = rank_indices(logits, descending=True)
        confidence = top1_confidence_margin(probabilities)
        tier = confidence_tier(confidence)
        rank_lookup = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ordering)}

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(candidates_tensor.shape[0]):
            metadata = _candidate_details(candidate_metadata, index)
            generation_type = _candidate_generation_type(candidate_metadata, index)
            feature_components = dict(feature_maps[index])
            feature_components.update(
                {
                    "reranker_logit": float(logits[index].item()),
                    "heuristic_score": float(payload["rows_by_evaluator"]["heuristic"][index]["score"]),
                    "masked_only_score": float(payload["rows_by_evaluator"]["masked_only"][index]["score"]),
                    "boundary_hybrid_score": float(payload["rows_by_evaluator"]["boundary_hybrid"][index]["score"]),
                }
            )
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(logits[index].item()),
                    probability=float(probabilities[index].item()),
                    rank=rank_lookup[index],
                    components=feature_components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(feature_components, generation_type),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=metadata.get("details", {}),
                )
            )

        candidate_scores.sort(key=lambda item: item.score, reverse=True)
        selected_index = candidate_scores[0].candidate_index if candidate_scores else -1

        notes = [
            "Candidates are reranked using frozen heuristic, masked-only, and boundary-hybrid scorer features.",
            "The base V-JEPA scorers remain frozen; only the tiny reranker head is learned.",
            f"Reranker model kind: {self.config.model_kind}.",
        ]
        if self.config.include_heuristic_features:
            notes.append("Heuristic-derived candidate features are included.")
        if self.training_summary is not None and self.training_summary.provisional:
            notes.append("This reranker was trained under a provisional overlap-split policy.")

        return VJEPAFutureScoreBundle(
            evaluator_name=f"vjepa2_tiny_reranker_{self.config.model_kind}",
            scoring_variant=f"frozen_feature_{self.config.model_kind}_reranker",
            model_id=self._base_model_id(),
            backend_used=self._base_backend_used(),
            selected_index=selected_index,
            confidence=confidence,
            confidence_tier=tier,
            candidate_scores=candidate_scores,
            observed_shape=tuple(int(dim) for dim in observed_tensor.shape),
            candidates_shape=tuple(int(dim) for dim in candidates_tensor.shape),
            embedding_dim=int(len(self.feature_names)),
            notes=notes,
        )

    def _build_feature_payload_for_example(self, example: Any, *, example_index: int) -> Dict[str, Any]:
        payload = self._build_feature_payload(
            example.observed,
            example.candidates,
            candidate_metadata=example.metadata["candidate_strategies"],
        )
        cache_payload = {
            "example_index": int(example_index),
            "source_video_id": example.metadata.get("source_video_id"),
            "correct_index": int(example.correct_index),
            "candidate_metadata": list(example.metadata["candidate_strategies"]),
            "evaluator_rows": payload["rows_by_evaluator"],
        }
        return {
            "candidate_feature_maps": payload["candidate_feature_maps"],
            "cache_payload": cache_payload,
        }

    def _build_feature_payload(
        self,
        observed: Tensor | Any,
        candidates: Tensor | Any,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        heuristic_bundle = run_future_selection_pipeline(
            observed=observed,
            candidates=candidates,
            candidate_metadata=candidate_metadata,
            evaluation_mode="heuristic",
        )
        masked_bundle = self.masked_only_scorer.score_example(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )
        boundary_bundle = self.boundary_hybrid_scorer.score_example(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )

        rows_by_evaluator = {
            "heuristic": self._rows_by_candidate_index(heuristic_bundle),
            "masked_only": self._rows_by_candidate_index(masked_bundle),
            "boundary_hybrid": self._rows_by_candidate_index(boundary_bundle),
        }
        feature_maps = self._feature_maps_from_rows(rows_by_evaluator, candidate_metadata=candidate_metadata)
        return {
            "candidate_feature_maps": feature_maps,
            "rows_by_evaluator": rows_by_evaluator,
        }

    def _feature_maps_from_rows(
        self,
        rows_by_evaluator: Mapping[str, Dict[int, Dict[str, Any]]],
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, float]]:
        candidate_indices = sorted(rows_by_evaluator["boundary_hybrid"].keys())
        active_evaluator_names = [
            evaluator_name
            for evaluator_name in rows_by_evaluator
            if evaluator_name != "heuristic" or self.config.include_heuristic_features
        ]
        top_rows = {
            evaluator_name: max(rows_by_evaluator[evaluator_name].values(), key=lambda item: float(item["score"]))
            for evaluator_name in active_evaluator_names
        }
        evaluator_stats: Dict[str, Dict[str, float | None]] = {}
        for evaluator_name in active_evaluator_names:
            ordered_rows = [rows_by_evaluator[evaluator_name][candidate_index] for candidate_index in candidate_indices]
            score_values = torch.tensor([float(item["score"]) for item in ordered_rows], dtype=torch.float32)
            sorted_scores, _ = torch.sort(score_values, descending=True)
            probability_values = [item.get("probability") for item in ordered_rows]
            probability_stats_available = all(value is not None for value in probability_values)
            probability_tensor = (
                torch.tensor([float(value) for value in probability_values], dtype=torch.float32)
                if probability_stats_available
                else None
            )
            evaluator_stats[evaluator_name] = {
                "score_mean": float(score_values.mean().item()),
                "score_std": float(score_values.std(unbiased=False).item()) if score_values.numel() > 1 else 0.0,
                "top_score": float(sorted_scores[0].item()),
                "runner_up_score": float(sorted_scores[1].item()) if sorted_scores.numel() > 1 else float(sorted_scores[0].item()),
                "score_range": float((sorted_scores[0] - sorted_scores[-1]).item()) if sorted_scores.numel() > 1 else 0.0,
                "probability_mean": (
                    float(probability_tensor.mean().item()) if probability_tensor is not None else None
                ),
                "probability_std": (
                    float(probability_tensor.std(unbiased=False).item()) if probability_tensor is not None and probability_tensor.numel() > 1 else 0.0
                ),
                "top_probability": (
                    float(torch.max(probability_tensor).item()) if probability_tensor is not None else None
                ),
                "runner_up_probability": (
                    float(torch.sort(probability_tensor, descending=True).values[1].item())
                    if probability_tensor is not None and probability_tensor.numel() > 1
                    else (float(torch.max(probability_tensor).item()) if probability_tensor is not None else None)
                ),
            }
        feature_maps: List[Dict[str, float]] = []

        for candidate_index in candidate_indices:
            feature_map: Dict[str, float] = {}
            ranks: Dict[str, float] = {}
            scores: Dict[str, float] = {}
            probabilities: Dict[str, float] = {}
            top_indicators: Dict[str, float] = {}
            for evaluator_name in active_evaluator_names:
                row_map = rows_by_evaluator[evaluator_name]
                row = row_map[candidate_index]
                ranks[evaluator_name] = float(row["rank"])
                scores[evaluator_name] = float(row["score"])
                top_row = top_rows[evaluator_name]
                top_indicators[evaluator_name] = 1.0 if int(row["rank"]) == 1 else 0.0
                stats = evaluator_stats[evaluator_name]
                candidate_count = max(len(candidate_indices), 1)

                feature_map[f"{evaluator_name}_score"] = float(row["score"])
                feature_map[f"{evaluator_name}_rank"] = float(row["rank"])
                feature_map[f"{evaluator_name}_inverse_rank"] = 1.0 / max(float(row["rank"]), 1.0)
                feature_map[f"{evaluator_name}_normalized_rank"] = (
                    1.0 - ((float(row["rank"]) - 1.0) / max(float(candidate_count - 1), 1.0))
                )
                feature_map[f"{evaluator_name}_is_top"] = top_indicators[evaluator_name]
                feature_map[f"{evaluator_name}_score_gap_from_top"] = float(top_row["score"]) - float(row["score"])
                feature_map[f"{evaluator_name}_score_centered"] = float(row["score"]) - float(stats["score_mean"])
                feature_map[f"{evaluator_name}_score_z"] = (
                    (float(row["score"]) - float(stats["score_mean"])) / max(float(stats["score_std"] or 0.0), 1e-6)
                )
                feature_map[f"{evaluator_name}_score_gap_from_runner_up"] = float(row["score"]) - float(
                    stats["runner_up_score"]
                )
                feature_map[f"{evaluator_name}_score_fraction_of_range"] = (
                    (float(row["score"]) - (float(stats["top_score"]) - float(stats["score_range"])))
                    / max(float(stats["score_range"] or 0.0), 1e-6)
                )
                if row.get("probability") is not None and top_row.get("probability") is not None:
                    probability_value = float(row["probability"])
                    probabilities[evaluator_name] = probability_value
                    feature_map[f"{evaluator_name}_probability"] = float(row["probability"])
                    feature_map[f"{evaluator_name}_prob_gap_from_top"] = float(top_row["probability"]) - float(
                        row["probability"]
                    )
                    probability_mean = float(stats["probability_mean"] or 0.0)
                    probability_std = float(stats["probability_std"] or 0.0)
                    runner_up_probability = float(stats["runner_up_probability"] or 0.0)
                    feature_map[f"{evaluator_name}_prob_centered"] = probability_value - probability_mean
                    feature_map[f"{evaluator_name}_prob_z"] = (
                        (probability_value - probability_mean) / max(probability_std, 1e-6)
                    )
                    feature_map[f"{evaluator_name}_prob_gap_from_runner_up"] = probability_value - runner_up_probability
                for component_name, component_value in sorted((row.get("components") or {}).items()):
                    feature_map[f"{evaluator_name}_component_{_safe_key(component_name)}"] = float(component_value)

            pair_names = [
                ("heuristic", "masked_only"),
                ("heuristic", "boundary_hybrid"),
                ("masked_only", "boundary_hybrid"),
            ]
            for left_name, right_name in pair_names:
                if left_name not in ranks or right_name not in ranks:
                    continue
                feature_map[f"rank_gap_{left_name}_{right_name}"] = abs(ranks[left_name] - ranks[right_name])
                feature_map[f"rank_match_{left_name}_{right_name}"] = 1.0 if ranks[left_name] == ranks[right_name] else 0.0
                feature_map[f"score_diff_{left_name}_{right_name}"] = scores[left_name] - scores[right_name]
                feature_map[f"score_abs_diff_{left_name}_{right_name}"] = abs(scores[left_name] - scores[right_name])
                feature_map[f"score_product_{left_name}_{right_name}"] = scores[left_name] * scores[right_name]
                feature_map[f"top_pair_agree_{left_name}_{right_name}"] = (
                    1.0 if top_indicators[left_name] == 1.0 and top_indicators[right_name] == 1.0 else 0.0
                )
                if left_name in probabilities and right_name in probabilities:
                    feature_map[f"prob_diff_{left_name}_{right_name}"] = (
                        probabilities[left_name] - probabilities[right_name]
                    )
                    feature_map[f"prob_abs_diff_{left_name}_{right_name}"] = abs(
                        probabilities[left_name] - probabilities[right_name]
                    )
                    feature_map[f"prob_product_{left_name}_{right_name}"] = (
                        probabilities[left_name] * probabilities[right_name]
                    )

            score_values = list(scores.values())
            rank_values = list(ranks.values())
            top_votes = sum(top_indicators.values())
            feature_map["score_mean_across_evaluators"] = sum(score_values) / max(len(score_values), 1)
            feature_map["score_max_across_evaluators"] = max(score_values)
            feature_map["score_min_across_evaluators"] = min(score_values)
            feature_map["score_range_across_evaluators"] = max(score_values) - min(score_values)
            feature_map["rank_mean_across_evaluators"] = sum(rank_values) / max(len(rank_values), 1)
            feature_map["best_rank_across_evaluators"] = min(rank_values)
            feature_map["worst_rank_across_evaluators"] = max(rank_values)
            feature_map["top_vote_count"] = top_votes
            feature_map["top_vote_fraction"] = top_votes / max(float(len(active_evaluator_names)), 1.0)
            feature_map["all_evaluators_top_agree"] = 1.0 if top_votes == float(len(active_evaluator_names)) else 0.0
            if probabilities:
                probability_values = list(probabilities.values())
                feature_map["probability_mean_across_evaluators"] = sum(probability_values) / max(
                    len(probability_values),
                    1,
                )
                feature_map["probability_max_across_evaluators"] = max(probability_values)
                feature_map["probability_min_across_evaluators"] = min(probability_values)
                feature_map["probability_range_across_evaluators"] = max(probability_values) - min(probability_values)

            if self.config.include_candidate_type_indicators:
                feature_map.update(self._optional_candidate_source_features(candidate_metadata, candidate_index))

            feature_maps.append(feature_map)

        return feature_maps

    @staticmethod
    def _optional_candidate_source_features(
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
        candidate_index: int,
    ) -> Dict[str, float]:
        metadata = _candidate_details(candidate_metadata, candidate_index)
        details = dict(metadata.get("details") or {})
        candidate_role = str(details.get("candidate_role") or "")
        strategy = str(metadata.get("strategy") or metadata.get("generation_type") or "")
        same_source = 1.0 if candidate_role in {"observed_clip_true_future", "same_clip_temporal_negative"} else 0.0
        external_source = 1.0 - same_source
        return {
            "candidate_source_same_clip": same_source,
            "candidate_source_external_clip": external_source,
            "candidate_source_other_sample": 1.0 if strategy == "future_segment_from_other_sample" else 0.0,
            "candidate_source_paired_counterfactual": 1.0 if strategy == "paired_template_counterfactual" else 0.0,
        }

    @staticmethod
    def _candidate_rows(bundle: Any) -> List[Dict[str, Any]]:
        if hasattr(bundle, "candidate_scores"):
            rows: List[Dict[str, Any]] = []
            for item in bundle.candidate_scores:
                rows.append(
                    {
                        "candidate_index": int(item.candidate_index),
                        "rank": int(item.rank),
                        "score": float(item.score),
                        "probability": float(item.probability) if getattr(item, "probability", None) is not None else None,
                        "generation_type": str(item.generation_type),
                        "components": dict(item.components),
                    }
                )
            return rows

        if hasattr(bundle, "ranked_candidates"):
            rows = []
            for rank, item in enumerate(bundle.ranked_candidates, start=1):
                rows.append(
                    {
                        "candidate_index": int(item.candidate_index),
                        "rank": int(rank),
                        "score": float(item.score),
                        "probability": None,
                        "generation_type": str(item.generation_type),
                        "components": dict(item.components),
                    }
                )
            return rows

        raise TypeError(f"Unsupported bundle type for reranker feature extraction: {type(bundle)!r}")

    def _rows_by_candidate_index(self, bundle: Any) -> Dict[int, Dict[str, Any]]:
        return {row["candidate_index"]: row for row in self._candidate_rows(bundle)}

    def _feature_tensor_from_maps(self, feature_maps: Sequence[Mapping[str, float]]) -> Tensor:
        if not self.feature_names:
            raise RuntimeError("Feature names are unavailable. Fit the reranker before scoring examples.")
        tensor = torch.zeros((len(feature_maps), len(self.feature_names)), dtype=torch.float32)
        for candidate_index, feature_map in enumerate(feature_maps):
            for feature_index, feature_name in enumerate(self.feature_names):
                tensor[candidate_index, feature_index] = float(feature_map.get(feature_name, 0.0))
        return tensor

    def _normalize_features(self, features: Tensor) -> Tensor:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Feature normalization statistics are unavailable. Fit the reranker first.")
        return (features.to(dtype=torch.float32) - self.feature_mean) / self.feature_std

    def _evaluate_loss(self, features: Tensor, labels: Tensor) -> float:
        self._require_trained_model()
        normalized = self._normalize_features(features).to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            logits = self.model(normalized)
            loss = F.cross_entropy(logits, labels)
        return float(loss.item())

    def _ensure_model(self, feature_dim: int) -> None:
        if self.model is not None:
            current_dim = None
            if self.config.model_kind == "linear" and hasattr(self.model, "layer"):
                current_dim = int(self.model.layer.in_features)
            elif self.config.model_kind == "mlp" and hasattr(self.model, "layers"):
                current_dim = int(self.model.layers[0].in_features)
            if current_dim == feature_dim:
                return
            self.model = None
        if self.config.model_kind == "linear":
            self.model = _LinearRerankerNetwork(feature_dim)
        else:
            self.model = _TinyMLPRerankerNetwork(
                feature_dim,
                hidden_dim=self.config.mlp_hidden_dim,
                dropout=self.config.dropout,
            )
        self.model.to(self.device)

    def _require_trained_model(self) -> None:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("The frozen-feature reranker must be fit before it can score examples.")

    def _cache_path(self, split_name: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"reranker_features_{split_name}.pt"

    def _source_video_ids(self, dataset: Sequence[Any], count: int) -> List[str]:
        identifiers: List[str] = []
        for index in range(count):
            example = dataset[index]
            identifiers.append(str(example.metadata.get("source_video_id", f"example_{index}")))
        return identifiers

    def _load_cached_feature_matrix(
        self,
        split_name: str,
        source_video_ids: Sequence[str],
        *,
        provisional: bool,
    ) -> Optional[FrozenFeatureCache]:
        in_memory = self._feature_cache_memory.get(split_name)
        if in_memory is not None and list(in_memory.source_video_ids) == list(source_video_ids):
            return FrozenFeatureCache(
                split_name=in_memory.split_name,
                feature_names=list(in_memory.feature_names),
                features=in_memory.features.clone(),
                labels=in_memory.labels.clone(),
                source_video_ids=list(in_memory.source_video_ids),
                example_payloads=list(in_memory.example_payloads),
                cache_hit=True,
                provisional=in_memory.provisional,
                cache_path=in_memory.cache_path,
            )

        cache_path = self._cache_path(split_name)
        if cache_path is None or not cache_path.exists():
            return None

        payload = torch.load(cache_path, map_location="cpu")
        expected_policy = self._cache_policy_payload(provisional=provisional)
        if payload.get("cache_version") != self.config.cache_version:
            return None
        if payload.get("feature_policy") != expected_policy:
            return None
        if payload.get("source_video_ids") != list(source_video_ids):
            return None

        return FrozenFeatureCache(
            split_name=str(payload["split_name"]),
            feature_names=list(payload["feature_names"]),
            features=torch.as_tensor(payload["features"], dtype=torch.float32),
            labels=torch.as_tensor(payload["labels"], dtype=torch.long),
            source_video_ids=list(payload["source_video_ids"]),
            example_payloads=list(payload.get("example_payloads") or []),
            cache_hit=True,
            provisional=bool(payload.get("provisional", False)),
            cache_path=str(cache_path),
        )

    def _save_feature_cache(self, cache: FrozenFeatureCache) -> None:
        cache_path = self._cache_path(cache.split_name)
        if cache_path is None:
            return
        payload = {
            "cache_version": self.config.cache_version,
            "feature_policy": self._cache_policy_payload(provisional=cache.provisional),
            "split_name": cache.split_name,
            "feature_names": list(cache.feature_names),
            "features": cache.features.cpu(),
            "labels": cache.labels.cpu(),
            "source_video_ids": list(cache.source_video_ids),
            "example_payloads": list(cache.example_payloads),
            "provisional": cache.provisional,
        }
        torch.save(payload, cache_path)

    def _cache_policy_payload(self, *, provisional: bool) -> Dict[str, Any]:
        return {
            "include_heuristic_features": self.config.include_heuristic_features,
            "include_candidate_type_indicators": self.config.include_candidate_type_indicators,
            "model_kind": self.config.model_kind,
            "provisional": provisional,
        }

    def _linear_weight_preview(self) -> Dict[str, float]:
        if self.model is None or self.config.model_kind != "linear" or not self.feature_names:
            return {}
        weight = self.model.layer.weight.detach().cpu().reshape(-1)
        preview = sorted(
            zip(self.feature_names, [float(value) for value in weight.tolist()]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[: min(8, len(self.feature_names))]
        return {name: value for name, value in preview}

    def _base_model_id(self) -> str:
        runtime = self.boundary_hybrid_scorer.adapter.describe_runtime()
        return str(runtime.get("model_id") or getattr(self.boundary_hybrid_scorer.adapter.config, "model_id", "unknown"))

    def _base_backend_used(self) -> str:
        runtime = self.boundary_hybrid_scorer.adapter.describe_runtime()
        return str(runtime.get("backend_used") or getattr(self.boundary_hybrid_scorer.adapter, "backend_used", "unknown"))

    @staticmethod
    def _build_rationale(components: Mapping[str, float], generation_type: str) -> str:
        dominant = max(components.items(), key=lambda item: item[1])[0]
        return (
            f"{generation_type} candidate under frozen_feature_reranker; strongest reranker-visible signal was "
            f"{dominant} ({components[dominant]:.3f})."
        )
