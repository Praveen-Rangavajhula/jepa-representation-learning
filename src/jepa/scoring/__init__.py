"""Scoring utilities for JEPA future-selection experiments."""

from .compatibility_metrics import (
    average_correct_rank,
    confidence_tier,
    cosine_similarity,
    latent_acceleration_penalty,
    latent_transition_smoothness,
    mean_reciprocal_rank,
    rank_indices,
    reciprocal_rank,
    softmax_normalize,
    top1_confidence_margin,
    uncertainty_bucket,
)
from .frozen_feature_reranker import (
    FrozenFeatureCache,
    FrozenFeatureRerankerConfig,
    FrozenFeatureRerankerScorer,
    FrozenFeatureRerankerTrainingSummary,
)
from .latent_future_scorer import LatentFuturePredictorScorer
from .semantic_future_scorer import HybridFutureScorer, SSV2SemanticFutureScorer
from .vjepa_future_scorer import (
    VJEPAFutureCandidateScore,
    VJEPAFutureScoreBundle,
    VJEPAFutureScorer,
)

__all__ = [
    "LatentFuturePredictorScorer",
    "FrozenFeatureCache",
    "FrozenFeatureRerankerConfig",
    "FrozenFeatureRerankerScorer",
    "FrozenFeatureRerankerTrainingSummary",
    "HybridFutureScorer",
    "VJEPAFutureCandidateScore",
    "VJEPAFutureScoreBundle",
    "VJEPAFutureScorer",
    "SSV2SemanticFutureScorer",
    "average_correct_rank",
    "confidence_tier",
    "cosine_similarity",
    "latent_acceleration_penalty",
    "latent_transition_smoothness",
    "mean_reciprocal_rank",
    "rank_indices",
    "reciprocal_rank",
    "softmax_normalize",
    "top1_confidence_margin",
    "uncertainty_bucket",
]
