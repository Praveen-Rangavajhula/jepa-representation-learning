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

    MASKED_RUNTIME_SIGNATURE = "single_mask_boundary_blocks_v2"
    BOUNDARY_HYBRID_SIGNATURE = "masked_boundary_hybrid_v1"

    def __init__(
        self,
        adapter: Optional[Any] = None,
        *,
        scoring_variant: str = "masked_future_prediction",
        auto_route_real_video: bool = True,
    ) -> None:
        if adapter is None:
            from jepa.models import VJEPA2Adapter

            adapter = VJEPA2Adapter()
        self.adapter = adapter
        self.scoring_variant = scoring_variant
        self.auto_route_real_video = auto_route_real_video
        self.masked_runtime_signature = self.MASKED_RUNTIME_SIGNATURE
        self.boundary_hybrid_signature = self.BOUNDARY_HYBRID_SIGNATURE
        self._predictor_mask_cache: Dict[Tuple[int, int, int, int], tuple[List[Tensor], List[Tensor], int]] = {}

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
        if (
            self.auto_route_real_video
            and variant == "masked_future_prediction"
            and self._looks_like_real_video_metadata(candidate_metadata)
        ):
            variant = "masked_boundary_hybrid"
        effective_variant = variant
        if variant not in {
            "masked_future_prediction",
            "masked_boundary_hybrid",
            "overlap_transition",
            "prefix_future_cosine",
        }:
            raise ValueError(f"Unsupported V-JEPA scoring variant: {variant}")

        if variant == "masked_boundary_hybrid":
            try:
                candidate_scores, embedding_dim, extra_notes = self._score_masked_boundary_hybrid(
                    observed_tensor,
                    candidates_tensor,
                    candidate_metadata=candidate_metadata,
                )
            except Exception as exc:
                masked_failure = f"{type(exc).__name__}: {exc}"
                if getattr(self.adapter, "device", None) is not None and getattr(self.adapter.device, "type", "") == "cuda":
                    raise RuntimeError(
                        "The boundary-focused masked hybrid scorer failed on CUDA. We do not try an in-place "
                        "fallback in the same runtime because a failed predictor call can leave the CUDA context in "
                        "an unstable state and waste Colab time. Restart the Colab runtime, make sure the latest repo "
                        "state is loaded, and rerun from the top. "
                        f"Masked scorer error: {masked_failure}"
                    ) from exc
                effective_variant = "overlap_transition"
                try:
                    candidate_scores, embedding_dim = self._score_overlap_transition(
                        observed_tensor,
                        candidates_tensor,
                        candidate_metadata=candidate_metadata,
                    )
                    extra_notes = [
                        "Boundary-focused masked hybrid was requested but unavailable in this runtime; "
                        "the scorer fell back to overlap_transition.",
                        f"Masked scorer error: {masked_failure}",
                    ]
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "The boundary-focused masked hybrid scorer failed, and the encoder-only overlap fallback also "
                        "failed in the same runtime. Restart the Colab runtime and rerun from the top."
                    ) from fallback_exc
        elif variant == "masked_future_prediction":
            try:
                candidate_scores, embedding_dim, extra_notes = self._score_masked_future_prediction(
                    observed_tensor,
                    candidates_tensor,
                    candidate_metadata=candidate_metadata,
                )
            except Exception as exc:
                masked_failure = f"{type(exc).__name__}: {exc}"
                if getattr(self.adapter, "device", None) is not None and getattr(self.adapter.device, "type", "") == "cuda":
                    raise RuntimeError(
                        "The masked future prediction scorer failed on CUDA. We do not try the encoder-only overlap "
                        "fallback in the same runtime because a failed predictor call can leave the CUDA context in an "
                        "unstable state and waste Colab time. Restart the Colab runtime, make sure the latest repo "
                        "state is loaded, and rerun from the top. "
                        f"Masked scorer error: {masked_failure}"
                    ) from exc
                effective_variant = "overlap_transition"
                try:
                    candidate_scores, embedding_dim = self._score_overlap_transition(
                        observed_tensor,
                        candidates_tensor,
                        candidate_metadata=candidate_metadata,
                    )
                    extra_notes = [
                        "Masked future prediction was requested but unavailable in this runtime; "
                        "the scorer fell back to overlap_transition.",
                        f"Masked scorer error: {masked_failure}",
                    ]
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "The masked future prediction scorer failed, and the encoder-only overlap fallback also failed "
                        "in the same runtime. This usually means the first masked attempt left the CUDA context in a bad "
                        "state. Restart the Colab runtime and rerun from the top."
                    ) from fallback_exc
        elif variant == "prefix_future_cosine":
            candidate_scores, embedding_dim = self._score_prefix_future_cosine(
                observed_tensor,
                candidates_tensor,
                candidate_metadata=candidate_metadata,
            )
            extra_notes = []
        else:
            candidate_scores, embedding_dim = self._score_overlap_transition(
                observed_tensor,
                candidates_tensor,
                candidate_metadata=candidate_metadata,
            )
            extra_notes = []

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
        notes.extend(extra_notes)

        return VJEPAFutureScoreBundle(
            evaluator_name=f"vjepa2_{effective_variant}",
            scoring_variant=effective_variant,
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

    @staticmethod
    def _looks_like_real_video_metadata(metadata: Optional[Sequence[Mapping[str, Any]]]) -> bool:
        if not metadata:
            return False
        for item in metadata:
            if any(key in item for key in ("pair_group", "label_template", "paired_template", "source_video_id")):
                return True
        return False

    @staticmethod
    def _swap_future_blocks(future: Tensor) -> Tensor:
        future = torch.as_tensor(future)
        if future.ndim != 4:
            raise ValueError(f"Future clip must have shape (T, C, H, W); got {tuple(future.shape)}.")
        if future.shape[0] < 2:
            return future
        half = int(future.shape[0] // 2)
        if future.shape[0] % 2 == 0 and half >= 1:
            return torch.cat([future[half:], future[:half]], dim=0)
        return torch.flip(future, dims=[0])

    @staticmethod
    def _temporal_range_to_token_indices(
        *,
        start: int,
        stop: int,
        spatial_tokens_per_step: int,
    ) -> Tensor:
        temporal = torch.arange(start, stop, dtype=torch.long)
        if temporal.numel() == 0:
            raise ValueError("Temporal token range must contain at least one step.")
        base = temporal.unsqueeze(1) * int(spatial_tokens_per_step)
        spatial = torch.arange(int(spatial_tokens_per_step), dtype=torch.long).unsqueeze(0)
        return (base + spatial).reshape(-1)

    def _build_predictor_masks(
        self,
        *,
        batch_size: int,
        observed_length: int,
        future_length: int,
    ) -> tuple[list[Tensor], list[Tensor], int]:
        layout = self.adapter.describe_token_layout()
        total_temporal_tokens = int(layout["temporal_tokens"])
        spatial_tokens = int(layout["spatial_tokens_per_temporal_step"])
        cache_key = (
            int(total_temporal_tokens),
            int(spatial_tokens),
            int(observed_length),
            int(future_length),
        )
        cached = self._predictor_mask_cache.get(cache_key)
        if cached is None:
            observed_fraction = float(observed_length) / float(max(observed_length + future_length, 1))
            context_stop = int(round(total_temporal_tokens * observed_fraction))
            context_stop = max(1, min(total_temporal_tokens - 1, context_stop))
            future_temporal_tokens = total_temporal_tokens - context_stop

            usable_temporal_tokens = min(context_stop, future_temporal_tokens)
            if usable_temporal_tokens >= 2 and usable_temporal_tokens % 2 == 0:
                block_count = 2
                block_size = usable_temporal_tokens // 2
            else:
                block_count = 1
                block_size = usable_temporal_tokens

            if block_size < 1:
                raise ValueError("Predictor mask block size must be positive.")

            # Keep context and target blocks equal-sized and anchored near the observed/future boundary.
            context_start = max(0, context_stop - (block_count * block_size))
            context_ranges = [
                (context_start + block_index * block_size, context_start + (block_index + 1) * block_size)
                for block_index in range(block_count)
            ]
            target_ranges = [
                (context_stop + block_index * block_size, context_stop + (block_index + 1) * block_size)
                for block_index in range(block_count)
            ]
            context_index_groups = [
                self._temporal_range_to_token_indices(
                    start=start,
                    stop=stop,
                    spatial_tokens_per_step=spatial_tokens,
                )
                for start, stop in context_ranges
            ]
            target_index_groups = [
                self._temporal_range_to_token_indices(
                    start=start,
                    stop=stop,
                    spatial_tokens_per_step=spatial_tokens,
                )
                for start, stop in target_ranges
            ]
            cached = (context_index_groups, target_index_groups, block_count)
            self._predictor_mask_cache[cache_key] = cached

        context_index_groups, target_index_groups, block_count = cached
        context_masks = [
            context_indices.unsqueeze(0).repeat(batch_size, 1)
            for context_indices in context_index_groups
        ]
        target_masks = [
            target_indices.unsqueeze(0).repeat(batch_size, 1)
            for target_indices in target_index_groups
        ]
        return context_masks, target_masks, block_count

    def _validate_predictor_mask_groups(
        self,
        *,
        context_masks: Sequence[Tensor],
        target_masks: Sequence[Tensor],
        batch_size: int,
    ) -> int:
        if len(context_masks) != len(target_masks):
            raise ValueError(
                "Predictor context and target masks must have the same number of groups. "
                f"Got {len(context_masks)} context groups and {len(target_masks)} target groups."
            )
        if not context_masks:
            raise ValueError("At least one predictor mask group is required.")

        sequence_length = int(self.adapter.describe_token_layout()["sequence_length"])
        expected_batch_size = int(batch_size)
        expected_token_count: int | None = None

        for group_index, (context_mask, target_mask) in enumerate(zip(context_masks, target_masks)):
            context_tensor = torch.as_tensor(context_mask, dtype=torch.long)
            target_tensor = torch.as_tensor(target_mask, dtype=torch.long)
            if context_tensor.ndim != 2 or target_tensor.ndim != 2:
                raise ValueError(
                    "Predictor masks must be 2D tensors of shape (batch_size, num_token_indices). "
                    f"Group {group_index} got context {tuple(context_tensor.shape)} and target {tuple(target_tensor.shape)}."
                )
            if int(context_tensor.shape[0]) != expected_batch_size or int(target_tensor.shape[0]) != expected_batch_size:
                raise ValueError(
                    "Predictor mask batch dimensions must match the candidate batch size. "
                    f"Expected {expected_batch_size}, got context {tuple(context_tensor.shape)} and target "
                    f"{tuple(target_tensor.shape)} in group {group_index}."
                )
            if int(context_tensor.shape[1]) < 1 or int(target_tensor.shape[1]) < 1:
                raise ValueError(f"Predictor mask group {group_index} must include at least one token.")
            if int(context_tensor.shape[1]) != int(target_tensor.shape[1]):
                raise ValueError(
                    "Context and target mask groups must be equal-sized for masked future scoring. "
                    f"Group {group_index} got {int(context_tensor.shape[1])} context tokens and "
                    f"{int(target_tensor.shape[1])} target tokens."
                )
            if expected_token_count is None:
                expected_token_count = int(context_tensor.shape[1])
            elif int(context_tensor.shape[1]) != expected_token_count:
                raise ValueError(
                    "All predictor mask groups must use the same token count. "
                    f"Expected {expected_token_count}, got {int(context_tensor.shape[1])} in group {group_index}."
                )
            if int(context_tensor.min().item()) < 0 or int(target_tensor.min().item()) < 0:
                raise ValueError(f"Predictor mask group {group_index} contains negative token indices.")
            if int(context_tensor.max().item()) >= sequence_length or int(target_tensor.max().item()) >= sequence_length:
                raise ValueError(
                    "Predictor mask group references token positions outside the V-JEPA sequence length. "
                    f"Sequence length is {sequence_length}, group {group_index} has context max "
                    f"{int(context_tensor.max().item())} and target max {int(target_tensor.max().item())}."
                )
        return int(expected_token_count or 0)

    def _score_masked_future_prediction(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[List[VJEPAFutureCandidateScore], int, list[str]]:
        combined_clips = torch.stack([torch.cat([observed, candidate], dim=0) for candidate in candidates], dim=0)
        context_masks, target_masks, block_count = self._build_predictor_masks(
            batch_size=int(combined_clips.shape[0]),
            observed_length=int(observed.shape[0]),
            future_length=int(candidates.shape[1]),
        )
        mask_token_count = self._validate_predictor_mask_groups(
            context_masks=context_masks,
            target_masks=target_masks,
            batch_size=int(combined_clips.shape[0]),
        )
        masked_blocks = []
        for context_mask, target_mask in zip(context_masks, target_masks):
            masked_blocks.append(
                self.adapter.predict_masked_tokens(
                    combined_clips,
                    context_masks=[context_mask],
                    target_masks=[target_mask],
                    batch_size=2,
                )
            )

        predicted_blocks = torch.cat([block.predicted_tokens for block in masked_blocks], dim=1)
        target_blocks = torch.cat([block.target_tokens for block in masked_blocks], dim=1)

        token_aligned = cosine_similarity(predicted_blocks, target_blocks)
        block_token_alignment = token_aligned.mean(dim=2)
        predicted = predicted_blocks.mean(dim=2)
        target = target_blocks.mean(dim=2)
        aligned = cosine_similarity(predicted, target)

        if block_count >= 2:
            reversed_target = torch.flip(target, dims=[1])
            reversed_alignment = cosine_similarity(predicted, reversed_target)
            order_margin_raw = block_token_alignment.mean(dim=1) - reversed_alignment.mean(dim=1)
            order_score = (torch.clamp(order_margin_raw, min=-1.0, max=1.0) + 1.0) / 2.0
            boundary_alignment = block_token_alignment[:, 0]
            future_alignment = block_token_alignment.mean(dim=1)
            worst_block_alignment = block_token_alignment.min(dim=1).values
            predicted_delta = predicted[:, 1:] - predicted[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            transition_consistency = (cosine_similarity(predicted_delta, target_delta).mean(dim=1) + 1.0) / 2.0
            total_scores = (
                0.30 * boundary_alignment
                + 0.25 * future_alignment
                + 0.25 * order_score
                + 0.10 * worst_block_alignment
                + 0.10 * transition_consistency
            )
        else:
            order_margin_raw = torch.zeros_like(aligned[:, 0])
            order_score = torch.full_like(aligned[:, 0], 0.5)
            boundary_alignment = block_token_alignment[:, 0]
            future_alignment = block_token_alignment[:, 0]
            worst_block_alignment = block_token_alignment[:, 0]
            transition_consistency = aligned[:, 0]
            total_scores = aligned[:, 0]

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(candidates.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            metadata = _candidate_details(candidate_metadata, index)
            components = {
                "predictor_boundary_alignment": float(boundary_alignment[index].item()),
                "predictor_future_alignment": float(future_alignment[index].item()),
                "predictor_worst_block_alignment": float(worst_block_alignment[index].item()),
                "predictor_order_score": float(order_score[index].item()),
                "predictor_transition_consistency": float(transition_consistency[index].item()),
                "predictor_order_margin_raw": float(order_margin_raw[index].item()),
                "predictor_block_token_alignment_mean": float(block_token_alignment[index].mean().item()),
                "predictor_aligned_mean": float(aligned[index].mean().item()),
            }
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(total_scores[index].item()),
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(components, generation_type, "masked_future_prediction"),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=metadata.get("details", {}),
                )
            )

        notes = [
            "This scorer uses V-JEPA predictor outputs with context and target masks rather than encoder-only pooling.",
            f"Future prediction is scored over {block_count} equal temporal block(s) to retain order sensitivity.",
            "Each temporal block is predicted in a separate masked forward pass to avoid unstable multi-mask batching in the current HF runtime.",
            "Context masks use equal-sized near-boundary observed blocks so predictor context and target groups stay shape-compatible.",
            f"Each predictor block uses {mask_token_count} masked tokens per candidate.",
            f"Masked scorer implementation signature: {self.masked_runtime_signature}.",
            "Predictor masks are cached by token layout and observed/future split so repeated notebook runs avoid rebuilding them.",
        ]
        return candidate_scores, int(masked_blocks[0].hidden_size), notes

    def _score_masked_boundary_hybrid(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[List[VJEPAFutureCandidateScore], int, list[str]]:
        base_scores, embedding_dim, base_notes = self._score_masked_boundary_hybrid_base(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        block_swap_scores: List[float] = [0.0 for _ in range(candidates.shape[0])]
        order_margin_raw = torch.zeros(candidates.shape[0], dtype=torch.float32)
        block_swapped_available = False

        if candidates.shape[1] >= 2:
            swapped_candidates = torch.stack(
                [self._swap_future_blocks(candidate) for candidate in candidates],
                dim=0,
            )
            swapped_scores, _, _ = self._score_masked_boundary_hybrid_base(
                observed,
                swapped_candidates,
                candidate_metadata=candidate_metadata,
            )
            block_swapped_available = True
            for index, swapped in enumerate(swapped_scores):
                block_swap_scores[index] = float(swapped.score)

        for index, base in enumerate(base_scores):
            swapped_score = block_swap_scores[index] if block_swapped_available else float(base.score)
            order_margin_raw[index] = float(base.score) - float(swapped_score)
            final_score = float(base.score) + 0.10 * float(order_margin_raw[index].item())
            components = dict(base.components)
            components.update(
                {
                    "boundary_hybrid_base_score": float(base.score),
                    "boundary_hybrid_block_swapped_score": float(swapped_score),
                    "boundary_hybrid_order_margin": float(order_margin_raw[index].item()),
                }
            )
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=base.candidate_index,
                    score=final_score,
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=base.generation_type,
                    rationale=self._build_rationale(components, base.generation_type, "masked_boundary_hybrid"),
                    source_index=base.source_index,
                    is_true=base.is_true,
                    details=base.details,
                )
            )

        notes = [
            "This scorer uses V-JEPA predictor outputs plus a boundary overlap encoder check and a direct block-swap order margin.",
            "The real-video path is auto-routed to this variant when candidate metadata looks like a clip-based real-video example.",
            "Masked predictor calls stay single-mask and fail fast on CUDA to avoid poisoned-runtime fallbacks.",
            f"Boundary hybrid implementation signature: {self.boundary_hybrid_signature}.",
        ]
        notes.extend(base_notes)
        return candidate_scores, embedding_dim, notes

    def _score_masked_boundary_hybrid_base(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[List[VJEPAFutureCandidateScore], int, list[str]]:
        combined_clips = torch.stack([torch.cat([observed, candidate], dim=0) for candidate in candidates], dim=0)
        context_masks, target_masks, block_count = self._build_predictor_masks(
            batch_size=int(combined_clips.shape[0]),
            observed_length=int(observed.shape[0]),
            future_length=int(candidates.shape[1]),
        )
        mask_token_count = self._validate_predictor_mask_groups(
            context_masks=context_masks,
            target_masks=target_masks,
            batch_size=int(combined_clips.shape[0]),
        )

        masked_blocks: list[Any] = []
        for context_mask, target_mask in zip(context_masks, target_masks):
            masked_blocks.append(
                self.adapter.predict_masked_tokens(
                    combined_clips,
                    context_masks=[context_mask],
                    target_masks=[target_mask],
                    batch_size=2,
                )
            )

        predicted_blocks = torch.cat([block.predicted_tokens for block in masked_blocks], dim=1)
        target_blocks = torch.cat([block.target_tokens for block in masked_blocks], dim=1)

        token_aligned = cosine_similarity(predicted_blocks, target_blocks)
        block_token_alignment = token_aligned.mean(dim=2)
        predicted = predicted_blocks.mean(dim=2)
        target = target_blocks.mean(dim=2)
        aligned = cosine_similarity(predicted, target)

        if block_count >= 2:
            boundary_alignment = block_token_alignment[:, 0]
            future_alignment = block_token_alignment[:, -1]
            predicted_delta = predicted[:, 1:] - predicted[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            transition_consistency = (cosine_similarity(predicted_delta, target_delta).mean(dim=1) + 1.0) / 2.0
        else:
            boundary_alignment = block_token_alignment[:, 0]
            future_alignment = block_token_alignment[:, 0]
            transition_consistency = aligned[:, 0]

        overlap_scores, _ = self._score_overlap_transition(
            observed,
            candidates,
            candidate_metadata=candidate_metadata,
        )
        overlap_score_tensor = torch.tensor(
            [item.score for item in overlap_scores],
            dtype=boundary_alignment.dtype,
            device=boundary_alignment.device,
        )
        total_scores = (
            0.45 * boundary_alignment
            + 0.15 * future_alignment
            + 0.20 * overlap_score_tensor
            + 0.10 * transition_consistency
        )

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(candidates.shape[0]):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            metadata = _candidate_details(candidate_metadata, index)
            components = {
                "predictor_boundary_alignment": float(boundary_alignment[index].item()),
                "predictor_future_alignment": float(future_alignment[index].item()),
                "encoder_boundary_overlap": float(overlap_score_tensor[index].item()),
                "transition_delta_consistency": float(transition_consistency[index].item()),
                "predictor_aligned_mean": float(aligned[index].mean().item()),
                "predictor_block_token_alignment_mean": float(block_token_alignment[index].mean().item()),
                "predictor_mask_token_count": float(mask_token_count),
            }
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(total_scores[index].item()),
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=generation_type,
                    rationale=self._build_rationale(components, generation_type, "masked_boundary_hybrid"),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=metadata.get("details", {}),
                )
            )

        notes = [
            "Boundary hybrid scores predictor alignment near the observed/future boundary, overlap-transition compatibility, and transition delta consistency.",
            "The overlap component is measured on short windows straddling the observed/future boundary.",
            "The predictor component is kept frozen and evaluated with single-mask calls only.",
        ]
        return candidate_scores, int(masked_blocks[0].hidden_size), notes

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
