"""Compatibility metrics for V-JEPA-backed future selection."""

from __future__ import annotations

from statistics import fmean, pvariance
from typing import List, Sequence

import torch
from torch import Tensor


def cosine_similarity(left: Tensor, right: Tensor) -> Tensor:
    left = torch.as_tensor(left, dtype=torch.float32)
    right = torch.as_tensor(right, dtype=torch.float32)
    left = torch.nn.functional.normalize(left, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)
    return (left * right).sum(dim=-1)


def latent_transition_smoothness(z_a: Tensor, z_b: Tensor, z_c: Tensor) -> Tensor:
    delta_ab = z_b - z_a
    delta_bc = z_c - z_b
    return 1.0 / (1.0 + torch.linalg.norm(delta_ab - delta_bc, dim=-1))


def latent_acceleration_penalty(z_a: Tensor, z_b: Tensor, z_c: Tensor) -> Tensor:
    delta_ab = z_b - z_a
    delta_bc = z_c - z_b
    return torch.linalg.norm(delta_ab - delta_bc, dim=-1)


def softmax_normalize(scores: Tensor, temperature: float = 1.0) -> Tensor:
    values = torch.as_tensor(scores, dtype=torch.float32)
    return torch.softmax(values / max(temperature, 1e-6), dim=-1)


def top1_confidence_margin(probabilities: Tensor) -> float:
    probs = torch.as_tensor(probabilities, dtype=torch.float32)
    if probs.numel() == 0:
        return 0.0
    if probs.numel() == 1:
        return 1.0
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.25:
        return "high"
    if confidence >= 0.10:
        return "medium"
    return "low"


def uncertainty_bucket(confidence: float) -> str:
    """Backward-compatible alias for legacy readers.

    The canonical categorical term for the pipeline is now ``confidence_tier``.
    """

    return confidence_tier(confidence)


def rank_indices(scores: Tensor, *, descending: bool = True) -> List[int]:
    tensor = torch.as_tensor(scores, dtype=torch.float32)
    order = torch.argsort(tensor, descending=descending)
    return [int(index.item()) for index in order]


def reciprocal_rank(correct_rank: int) -> float:
    if correct_rank < 1:
        raise ValueError("correct_rank must be 1-indexed and positive.")
    return 1.0 / float(correct_rank)


def mean_reciprocal_rank(correct_ranks: Sequence[int]) -> float:
    if not correct_ranks:
        return 0.0
    values = [reciprocal_rank(rank) for rank in correct_ranks]
    return float(sum(values) / len(values))


def average_correct_rank(correct_ranks: Sequence[int]) -> float:
    if not correct_ranks:
        return 0.0
    return float(sum(correct_ranks) / len(correct_ranks))


def mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    """Return population mean and variance for a numeric sequence."""

    if not values:
        return 0.0, 0.0
    floats = [float(value) for value in values]
    if len(floats) == 1:
        return floats[0], 0.0
    return float(fmean(floats)), float(pvariance(floats))
