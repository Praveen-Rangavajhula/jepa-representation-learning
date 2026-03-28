"""Learnable latent future prediction on top of frozen V-JEPA embeddings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _as_tensor(value: Tensor | Any) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if value.requires_grad else value.detach()
    return torch.as_tensor(value, dtype=torch.float32)


@dataclass(slots=True)
class LatentFuturePredictorConfig:
    """Training and inference settings for the latent future predictor."""

    hidden_dim: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 25
    batch_size: int = 16
    adapter_batch_size: int = 8
    validation_fraction: float = 0.2
    max_training_examples: int = 128
    mse_weight: float = 0.5
    cosine_weight: float = 0.5
    seed: int = 23
    device: Optional[str] = None

    def validate(self) -> "LatentFuturePredictorConfig":
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.adapter_batch_size < 1:
            raise ValueError("adapter_batch_size must be at least 1.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must satisfy 0 <= value < 1.")
        if self.max_training_examples < 1:
            raise ValueError("max_training_examples must be at least 1.")
        if self.mse_weight < 0 or self.cosine_weight < 0:
            raise ValueError("loss weights must be non-negative.")
        return self

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "adapter_batch_size": self.adapter_batch_size,
            "validation_fraction": self.validation_fraction,
            "max_training_examples": self.max_training_examples,
            "mse_weight": self.mse_weight,
            "cosine_weight": self.cosine_weight,
            "seed": self.seed,
            "device": str(self.resolved_device()),
        }


@dataclass(slots=True)
class LatentFuturePredictorTrainingSummary:
    """Compact training summary for notebook reporting."""

    embedding_dim: int
    train_examples: int
    validation_examples: int
    epochs: int
    best_epoch: int
    final_train_loss: float
    best_validation_loss: float
    embedding_cache_hit: bool = False
    embedding_extraction_seconds: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    loss_history: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.final_train_loss,
            "best_validation_loss": self.best_validation_loss,
            "embedding_cache_hit": self.embedding_cache_hit,
            "embedding_extraction_seconds": self.embedding_extraction_seconds,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
            "loss_history": list(self.loss_history),
        }


class _LatentPredictorNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class LatentFuturePredictor:
    """Predict the next latent state from the observed latent state."""

    def __init__(self, config: Optional[LatentFuturePredictorConfig] = None) -> None:
        self.config = (config or LatentFuturePredictorConfig()).validate()
        self.device = self.config.resolved_device()
        self.model: Optional[_LatentPredictorNetwork] = None
        self.embedding_dim: Optional[int] = None
        self.training_summary: Optional[LatentFuturePredictorTrainingSummary] = None
        self._training_embedding_cache: Dict[tuple[Any, ...], tuple[Tensor, Tensor]] = {}

    def fit(
        self,
        dataset: Sequence[Any],
        *,
        adapter: Any,
        max_examples: Optional[int] = None,
    ) -> LatentFuturePredictorTrainingSummary:
        total_start = time.perf_counter()
        observed_embeddings, future_embeddings, cache_hit, embedding_extraction_seconds = self._build_training_embeddings(
            dataset,
            adapter=adapter,
            max_examples=max_examples,
        )
        if observed_embeddings.shape[0] < 2:
            raise ValueError("Need at least two examples to train the latent future predictor.")

        embedding_dim = int(observed_embeddings.shape[-1])
        self._ensure_model(embedding_dim)
        generator = torch.Generator().manual_seed(self.config.seed)
        indices = torch.randperm(observed_embeddings.shape[0], generator=generator)

        validation_examples = int(round(observed_embeddings.shape[0] * self.config.validation_fraction))
        validation_examples = min(max(validation_examples, 1), observed_embeddings.shape[0] - 1)
        train_examples = int(observed_embeddings.shape[0] - validation_examples)

        train_indices = indices[:train_examples]
        validation_indices = indices[train_examples:]

        train_observed = observed_embeddings[train_indices]
        train_future = future_embeddings[train_indices]
        validation_observed = observed_embeddings[validation_indices]
        validation_future = future_embeddings[validation_indices]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_state: Optional[Dict[str, Tensor]] = None
        best_validation_loss = float("inf")
        best_epoch = 0
        history: List[Dict[str, float]] = []

        self.model.train()
        training_start = time.perf_counter()
        for epoch in range(self.config.epochs):
            permutation = torch.randperm(train_observed.shape[0], generator=generator)
            running_loss = 0.0
            batch_count = 0

            for start in range(0, train_observed.shape[0], self.config.batch_size):
                batch_indices = permutation[start : start + self.config.batch_size]
                batch_observed = train_observed[batch_indices].to(device=self.device, dtype=torch.float32)
                batch_future = train_future[batch_indices].to(device=self.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                predicted = self.model(batch_observed)
                loss = self._loss(predicted, batch_future)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batch_count += 1

            train_loss = running_loss / max(batch_count, 1)
            validation_loss = self._evaluate_loss(validation_observed, validation_future)
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

        self.training_summary = LatentFuturePredictorTrainingSummary(
            embedding_dim=embedding_dim,
            train_examples=train_examples,
            validation_examples=validation_examples,
            epochs=self.config.epochs,
            best_epoch=best_epoch,
            final_train_loss=float(history[-1]["train_loss"]),
            best_validation_loss=best_validation_loss,
            embedding_cache_hit=cache_hit,
            embedding_extraction_seconds=float(embedding_extraction_seconds),
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            loss_history=history,
        )
        return self.training_summary

    def predict_future_latent(
        self,
        observed_clip: Tensor | Any,
        *,
        adapter: Any,
    ) -> Tensor:
        self._require_trained_model()
        observed_tensor = _as_tensor(observed_clip).to(dtype=torch.float32)

        if observed_tensor.ndim in {1, 2} and observed_tensor.shape[-1] == self.embedding_dim:
            observed_embeddings = observed_tensor
            if observed_embeddings.ndim == 1:
                observed_embeddings = observed_embeddings.unsqueeze(0)
        else:
            if observed_tensor.ndim == 4:
                observed_embeddings = adapter.encode_video(observed_tensor).unsqueeze(0)
            elif observed_tensor.ndim == 5:
                observed_embeddings = adapter.encode_batch(
                    observed_tensor,
                    batch_size=self.config.adapter_batch_size,
                )
            else:
                raise ValueError(
                    "observed_clip must be an embedding or a clip tensor shaped (T, C, H, W) "
                    "or (B, T, C, H, W)."
                )

        observed_embeddings = F.normalize(
            observed_embeddings.to(device=self.device, dtype=torch.float32),
            dim=-1,
        )
        with torch.no_grad():
            predicted = self.model(observed_embeddings)
        return F.normalize(predicted.to(dtype=torch.float32), dim=-1).detach().cpu()

    def score_candidates(
        self,
        observed_clip: Tensor | Any,
        candidate_futures: Tensor | Any,
        *,
        adapter: Any,
    ) -> Dict[str, Tensor]:
        self._require_trained_model()
        candidate_tensor = _as_tensor(candidate_futures).to(dtype=torch.float32)
        if candidate_tensor.ndim != 5:
            raise ValueError(
                f"candidate_futures must have shape (K, T, C, H, W); got {tuple(candidate_tensor.shape)}."
            )

        predicted_future = self.predict_future_latent(observed_clip, adapter=adapter).squeeze(0)
        candidate_embeddings = adapter.encode_batch(
            candidate_tensor,
            batch_size=self.config.adapter_batch_size,
        ).detach().to(dtype=torch.float32).cpu()
        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)

        cosine_scores = (candidate_embeddings * predicted_future.unsqueeze(0)).sum(dim=-1)
        distance_scores = 1.0 / (1.0 + torch.linalg.norm(candidate_embeddings - predicted_future.unsqueeze(0), dim=-1))

        return {
            "scores": cosine_scores,
            "predicted_future_cosine": cosine_scores,
            "predicted_future_distance_score": distance_scores,
            "predicted_future_latent": predicted_future,
            "candidate_latents": candidate_embeddings,
        }

    def _build_training_embeddings(
        self,
        dataset: Sequence[Any],
        *,
        adapter: Any,
        max_examples: Optional[int],
    ) -> tuple[Tensor, Tensor, bool, float]:
        example_count = min(len(dataset), int(max_examples or self.config.max_training_examples))
        cache_key = self._training_embedding_cache_key(
            dataset,
            adapter=adapter,
            example_count=example_count,
        )
        cached = self._training_embedding_cache.get(cache_key)
        if cached is not None:
            observed_cached, future_cached = cached
            return observed_cached.clone(), future_cached.clone(), True, 0.0

        extraction_start = time.perf_counter()
        observed_clips: List[Tensor] = []
        future_clips: List[Tensor] = []
        for index in range(example_count):
            example = dataset[index]
            observed = _as_tensor(getattr(example, "observed")).to(dtype=torch.float32)
            candidates = _as_tensor(getattr(example, "candidates")).to(dtype=torch.float32)
            correct_index = int(getattr(example, "correct_index"))
            observed_clips.append(observed)
            future_clips.append(candidates[correct_index])

        observed_batch = torch.stack(observed_clips, dim=0)
        future_batch = torch.stack(future_clips, dim=0)

        observed_embeddings = adapter.encode_batch(
            observed_batch,
            batch_size=self.config.adapter_batch_size,
        ).detach().to(dtype=torch.float32).cpu()
        future_embeddings = adapter.encode_batch(
            future_batch,
            batch_size=self.config.adapter_batch_size,
        ).detach().to(dtype=torch.float32).cpu()

        normalized = (
            F.normalize(observed_embeddings, dim=-1),
            F.normalize(future_embeddings, dim=-1),
        )
        self._training_embedding_cache[cache_key] = (
            normalized[0].clone(),
            normalized[1].clone(),
        )
        extraction_seconds = float(time.perf_counter() - extraction_start)
        return normalized[0], normalized[1], False, extraction_seconds

    def _training_embedding_cache_key(
        self,
        dataset: Sequence[Any],
        *,
        adapter: Any,
        example_count: int,
    ) -> tuple[Any, ...]:
        runtime: Mapping[str, Any] = {}
        if hasattr(adapter, "describe_runtime"):
            try:
                runtime = adapter.describe_runtime()
            except Exception:
                runtime = {}

        return (
            id(dataset),
            example_count,
            self.config.adapter_batch_size,
            str(runtime.get("model_id", "")),
            str(runtime.get("backend_used", "")),
            str(runtime.get("dtype", "")),
        )

    def _ensure_model(self, embedding_dim: int) -> None:
        if self.model is not None and self.embedding_dim == embedding_dim:
            return
        self.embedding_dim = embedding_dim
        self.model = _LatentPredictorNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(device=self.device, dtype=torch.float32)

    def _evaluate_loss(self, observed_embeddings: Tensor, future_embeddings: Tensor) -> float:
        self._require_trained_model()
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(observed_embeddings.to(device=self.device, dtype=torch.float32))
            loss = self._loss(predicted, future_embeddings.to(device=self.device, dtype=torch.float32))
        self.model.train()
        return float(loss.item())

    def _loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)
        mse_loss = F.mse_loss(predicted, target)
        cosine_loss = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
        return (self.config.mse_weight * mse_loss) + (self.config.cosine_weight * cosine_loss)

    def _require_trained_model(self) -> None:
        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("LatentFuturePredictor must be fitted before inference.")
