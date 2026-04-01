"""Semantic and hybrid future scorers for SSV2-style candidate ranking."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from jepa.agents import run_future_selection_pipeline
from jepa.models import VJEPA2AdapterConfig, clip_to_processor_videos, ensure_video_batch, make_ssv2_vjepa_config

from .compatibility_metrics import confidence_tier, rank_indices, softmax_normalize, top1_confidence_margin
from .vjepa_future_scorer import VJEPAFutureCandidateScore, VJEPAFutureScoreBundle


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


def _candidate_metadata(metadata: Optional[Sequence[Mapping[str, Any]]], index: int) -> Dict[str, Any]:
    if metadata is None or index >= len(metadata):
        return {}
    return dict(metadata[index])


def _candidate_detail_payload(metadata: Optional[Sequence[Mapping[str, Any]]], index: int) -> Dict[str, Any]:
    payload = _candidate_metadata(metadata, index)
    details = payload.get("details", {})
    return dict(details) if isinstance(details, Mapping) else {}


def _candidate_lookup(bundle: Any) -> Dict[int, Any]:
    if hasattr(bundle, "candidate_scores"):
        return {int(item.candidate_index): item for item in bundle.candidate_scores}
    if hasattr(bundle, "ranked_candidates"):
        return {int(item.candidate_index): item for item in bundle.ranked_candidates}
    raise TypeError(f"Unsupported bundle type for hybrid scoring: {type(bundle)!r}")


class SSV2SemanticFutureScorer:
    """Score candidate futures with the SSV2-tuned V-JEPA 2 classification head."""

    VALID_SCORE_MODES = {"max_logit", "max_probability", "top2_margin"}

    def __init__(
        self,
        config: Optional[VJEPA2AdapterConfig] = None,
        *,
        score_mode: str = "max_probability",
        batch_size: int = 2,
    ) -> None:
        base_config = config or make_ssv2_vjepa_config()
        self.config = base_config.validate()
        self.score_mode = str(score_mode).strip().lower()
        if self.score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(
                f"Unsupported semantic score_mode {score_mode!r}. Expected one of {sorted(self.VALID_SCORE_MODES)}."
            )
        self.batch_size = max(int(batch_size), 1)
        self.model: Any = None
        self.processor: Any = None
        self.backend_used: str | None = None
        self.device = self.config.resolved_device()
        self.model_dtype = self.config.resolved_torch_dtype()
        self.last_load_errors: Dict[str, str] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self, *, force: bool = False) -> Dict[str, Any]:
        if self.loaded and not force:
            return self.describe_runtime()

        self.model = None
        self.processor = None
        self.backend_used = None
        self.last_load_errors = {}

        try:
            from transformers import AutoModelForVideoClassification, AutoVideoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Semantic future scoring requires `transformers`, `accelerate`, `huggingface_hub`, and `safetensors`."
            ) from exc

        self.processor = AutoVideoProcessor.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )

        attempted_load_kwargs: List[Dict[str, Any]] = [
            {
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "low_cpu_mem_usage": True,
                "dtype": self.model_dtype,
                "attn_implementation": self.config.attn_implementation,
            },
            {
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "low_cpu_mem_usage": True,
                "torch_dtype": self.model_dtype,
                "attn_implementation": self.config.attn_implementation,
            },
        ]

        model = None
        last_error: Exception | None = None
        for load_kwargs in attempted_load_kwargs:
            try:
                model = AutoModelForVideoClassification.from_pretrained(self.config.model_id, **load_kwargs)
                break
            except TypeError:
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs.pop("attn_implementation", None)
                try:
                    model = AutoModelForVideoClassification.from_pretrained(self.config.model_id, **fallback_kwargs)
                    break
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc

        if model is None:
            message = f"Failed to load semantic classification head {self.config.model_id}."
            self.last_load_errors["huggingface"] = f"{type(last_error).__name__}: {last_error}" if last_error else message
            raise RuntimeError(message) from last_error

        self.model = model.to(self.device)
        self.model.eval()
        self.backend_used = "huggingface"
        return self.describe_runtime()

    def describe_runtime(self) -> Dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        num_labels = getattr(model_config, "num_labels", None)
        return {
            "loaded": self.loaded,
            "backend_used": self.backend_used,
            "model_id": self.config.model_id,
            "device": str(self.device),
            "dtype": str(self.model_dtype).replace("torch.", ""),
            "target_frames": self.config.target_frames,
            "image_size": self.config.image_size,
            "score_mode": self.score_mode,
            "batch_size": self.batch_size,
            "model_class": type(self.model).__name__ if self.model is not None else None,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "num_labels": int(num_labels) if num_labels is not None else None,
            "last_load_errors": dict(self.last_load_errors),
        }

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
            scoring_variant=scoring_variant,
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
        if int(candidates_tensor.shape[0]) < 1:
            raise ValueError("At least one candidate future is required.")

        combined_clips = torch.stack([torch.cat([observed_tensor, candidate], dim=0) for candidate in candidates_tensor], dim=0)
        logits = self._forward_logits_batched(combined_clips)
        probabilities = torch.softmax(logits, dim=-1)
        max_probability, predicted_ids = probabilities.max(dim=-1)
        max_logit = logits.max(dim=-1).values
        if logits.shape[-1] >= 2:
            top2_probabilities = torch.topk(probabilities, k=2, dim=-1).values
            top2_logits = torch.topk(logits, k=2, dim=-1).values
            top2_margin = top2_probabilities[:, 0] - top2_probabilities[:, 1]
            top2_logit_margin = top2_logits[:, 0] - top2_logits[:, 1]
        else:
            top2_margin = torch.ones_like(max_probability)
            top2_logit_margin = torch.ones_like(max_logit)

        if self.score_mode == "max_logit":
            raw_scores = max_logit
        elif self.score_mode == "top2_margin":
            raw_scores = top2_margin
        else:
            raw_scores = max_probability

        ranking_probabilities = softmax_normalize(raw_scores)
        ordering = rank_indices(raw_scores, descending=True)
        confidence = top1_confidence_margin(ranking_probabilities)
        tier = confidence_tier(confidence)
        rank_lookup = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ordering)}

        model_config = getattr(self.model, "config", None)
        id2label = getattr(model_config, "id2label", {}) or {}
        num_labels = int(getattr(model_config, "num_labels", int(logits.shape[-1])))

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        for index in range(int(candidates_tensor.shape[0])):
            generation_type = _candidate_generation_type(candidate_metadata, index)
            metadata = _candidate_metadata(candidate_metadata, index)
            details = _candidate_detail_payload(candidate_metadata, index)
            predicted_label_id = int(predicted_ids[index].item())
            predicted_label = str(id2label.get(predicted_label_id, predicted_label_id))
            components = {
                "semantic_max_logit": float(max_logit[index].item()),
                "semantic_max_probability": float(max_probability[index].item()),
                "semantic_top2_margin": float(top2_margin[index].item()),
                "semantic_top2_logit_margin": float(top2_logit_margin[index].item()),
            }
            details.update(
                {
                    "semantic_predicted_label": predicted_label,
                    "semantic_predicted_label_id": predicted_label_id,
                    "semantic_score_mode": self.score_mode,
                }
            )
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=float(raw_scores[index].item()),
                    probability=float(ranking_probabilities[index].item()),
                    rank=rank_lookup[index],
                    components=components,
                    generation_type=generation_type,
                    rationale=(
                        f"{generation_type} candidate under semantic_only; classifier predicted "
                        f"{predicted_label!r} with {self.score_mode}={float(raw_scores[index].item()):.3f}."
                    ),
                    source_index=metadata.get("source_index"),
                    is_true=metadata.get("is_true"),
                    details=details,
                )
            )

        candidate_scores.sort(key=lambda item: item.score, reverse=True)
        selected_index = candidate_scores[0].candidate_index if candidate_scores else -1
        notes = [
            "Semantic scorer uses AutoModelForVideoClassification on the official SSV2-tuned V-JEPA 2 checkpoint.",
            f"Each candidate is scored from the concatenated observed+future clip and converted with score_mode={self.score_mode}.",
            "This scorer is intended as a semantic compatibility prior and does not replace temporal scoring on its own.",
        ]

        return VJEPAFutureScoreBundle(
            evaluator_name="semantic_only",
            scoring_variant=scoring_variant or f"ssv2_classification_{self.score_mode}",
            model_id=self.config.model_id,
            backend_used=str(self.backend_used or "huggingface"),
            selected_index=selected_index,
            confidence=confidence,
            confidence_tier=tier,
            candidate_scores=candidate_scores,
            observed_shape=tuple(int(dim) for dim in observed_tensor.shape),
            candidates_shape=tuple(int(dim) for dim in candidates_tensor.shape),
            embedding_dim=num_labels,
            notes=notes,
        )

    def _forward_logits_batched(self, clips: Tensor | Any) -> Tensor:
        batch = ensure_video_batch(clips)
        if not self.loaded:
            self.load()

        outputs: List[Tensor] = []
        start = 0
        current_batch_size = min(self.batch_size, int(batch.shape[0]))
        while start < int(batch.shape[0]):
            end = min(start + current_batch_size, int(batch.shape[0]))
            try:
                outputs.append(self._forward_logits(batch[start:end]))
                start = end
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

        return torch.cat(outputs, dim=0)

    def _forward_logits(self, clips: Tensor) -> Tensor:
        if not self.loaded:
            self.load()

        videos = clip_to_processor_videos(
            clips,
            target_frames=self.config.target_frames,
            image_size=self.config.image_size,
        )
        inputs = self._processor_call(videos)
        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Semantic classification model did not expose logits.")
        return torch.as_tensor(logits, dtype=torch.float32)

    def _processor_call(self, videos: Sequence[Tensor]) -> Mapping[str, Tensor]:
        if self.processor is None:
            raise RuntimeError("Semantic video processor is not loaded.")
        try:
            outputs = self.processor(
                videos,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_rescale=False,
            )
        except TypeError:
            outputs = self.processor(
                videos=videos,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_rescale=False,
            )

        if isinstance(outputs, Mapping):
            return outputs
        if torch.is_tensor(outputs):
            return {"pixel_values_videos": outputs}
        raise TypeError(f"Unsupported processor output type: {type(outputs)!r}")

    def _move_inputs_to_device(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(device=self.device, dtype=self.model_dtype)
                else:
                    moved[key] = value.to(device=self.device)
            else:
                moved[key] = value
        return moved


class HybridFutureScorer:
    """Blend a temporal scorer with the SSV2 semantic scorer."""

    def __init__(
        self,
        semantic_scorer: SSV2SemanticFutureScorer,
        *,
        temporal_scorer: Optional[Any] = None,
        temporal_mode: str = "heuristic",
        temporal_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        self.semantic_scorer = semantic_scorer
        self.temporal_scorer = temporal_scorer
        self.temporal_mode = str(temporal_mode).strip().lower()
        self.temporal_weight = float(temporal_weight)
        self.semantic_weight = float(semantic_weight)
        if self.temporal_mode not in {"heuristic", "scorer"}:
            raise ValueError("HybridFutureScorer temporal_mode must be 'heuristic' or 'scorer'.")
        if self.temporal_mode == "scorer" and self.temporal_scorer is None:
            raise ValueError("HybridFutureScorer temporal_mode='scorer' requires a temporal_scorer instance.")

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
            scoring_variant=scoring_variant,
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

        semantic_bundle = self.semantic_scorer.score_example(
            observed_tensor,
            candidates_tensor,
            candidate_metadata=candidate_metadata,
        )
        temporal_bundle = self._score_temporal(
            observed_tensor,
            candidates_tensor,
            candidate_metadata=candidate_metadata,
        )

        semantic_lookup = _candidate_lookup(semantic_bundle)
        temporal_lookup = _candidate_lookup(temporal_bundle)

        candidate_scores: List[VJEPAFutureCandidateScore] = []
        raw_scores: List[float] = []
        for index in range(int(candidates_tensor.shape[0])):
            semantic_item = semantic_lookup[index]
            temporal_item = temporal_lookup[index]
            generation_type = str(getattr(semantic_item, "generation_type", getattr(temporal_item, "generation_type", "unknown")))
            details = dict(getattr(semantic_item, "details", {}) or {})
            temporal_score = float(getattr(temporal_item, "score"))
            semantic_score = float(getattr(semantic_item, "score"))
            final_score = (self.temporal_weight * temporal_score) + (self.semantic_weight * semantic_score)
            raw_scores.append(final_score)

            components = {
                **{f"temporal_{key}": float(value) for key, value in dict(getattr(temporal_item, "components", {})).items()},
                **{f"semantic_{key}": float(value) for key, value in dict(getattr(semantic_item, "components", {})).items()},
                "temporal_total": temporal_score,
                "semantic_total": semantic_score,
                "temporal_weight": self.temporal_weight,
                "semantic_weight": self.semantic_weight,
            }
            candidate_scores.append(
                VJEPAFutureCandidateScore(
                    candidate_index=index,
                    score=final_score,
                    probability=0.0,
                    rank=-1,
                    components=components,
                    generation_type=generation_type,
                    rationale=(
                        f"{generation_type} candidate under hybrid scoring; "
                        f"temporal={temporal_score:.3f}, semantic={semantic_score:.3f}, final={final_score:.3f}."
                    ),
                    source_index=getattr(semantic_item, "source_index", getattr(temporal_item, "source_index", None)),
                    is_true=getattr(semantic_item, "is_true", getattr(temporal_item, "is_true", None)),
                    details=details,
                )
            )

        raw_scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        probabilities = softmax_normalize(raw_scores_tensor)
        ordering = rank_indices(raw_scores_tensor, descending=True)
        confidence = top1_confidence_margin(probabilities)
        tier = confidence_tier(confidence)
        rank_lookup = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ordering)}
        for item in candidate_scores:
            item.rank = rank_lookup[item.candidate_index]
            item.probability = float(probabilities[item.candidate_index].item())

        candidate_scores.sort(key=lambda item: item.score, reverse=True)
        selected_index = candidate_scores[0].candidate_index if candidate_scores else -1

        temporal_label = (
            "heuristic"
            if self.temporal_mode == "heuristic"
            else str(getattr(temporal_bundle, "evaluator_name", "temporal_scorer"))
        )
        notes = [
            f"Hybrid scorer blends temporal source={temporal_label} with semantic_only using weights "
            f"temporal={self.temporal_weight:.3f}, semantic={self.semantic_weight:.3f}.",
            f"Semantic score mode is {self.semantic_scorer.score_mode}.",
            "This scorer is intended as a lightweight fusion layer without any additional training.",
        ]
        notes.extend(list(getattr(semantic_bundle, "notes", [])))

        return VJEPAFutureScoreBundle(
            evaluator_name="hybrid",
            scoring_variant=scoring_variant or f"{temporal_label}_plus_semantic_{self.semantic_scorer.score_mode}",
            model_id=str(getattr(semantic_bundle, "model_id", self.semantic_scorer.config.model_id)),
            backend_used=str(getattr(semantic_bundle, "backend_used", "huggingface")),
            selected_index=selected_index,
            confidence=confidence,
            confidence_tier=tier,
            candidate_scores=candidate_scores,
            observed_shape=tuple(int(dim) for dim in observed_tensor.shape),
            candidates_shape=tuple(int(dim) for dim in candidates_tensor.shape),
            embedding_dim=int(getattr(semantic_bundle, "embedding_dim", 0)),
            notes=notes,
        )

    def _score_temporal(
        self,
        observed: Tensor,
        candidates: Tensor,
        *,
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> Any:
        if self.temporal_mode == "heuristic":
            return run_future_selection_pipeline(
                observed=observed,
                candidates=candidates,
                candidate_metadata=candidate_metadata,
                evaluation_mode="heuristic",
            )
        if hasattr(self.temporal_scorer, "score_example"):
            return self.temporal_scorer.score_example(
                observed,
                candidates,
                candidate_metadata=candidate_metadata,
            )
        if hasattr(self.temporal_scorer, "score_future_candidates"):
            return self.temporal_scorer.score_future_candidates(
                observed,
                candidates,
                candidate_metadata=candidate_metadata,
            )
        raise TypeError("Temporal scorer must expose score_example() or score_future_candidates().")
