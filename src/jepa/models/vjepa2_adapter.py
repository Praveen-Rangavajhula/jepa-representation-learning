"""Notebook-first adapter for loading and encoding clips with V-JEPA 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .model_config import VJEPA2AdapterConfig
from .video_preprocessing import (
    ensure_video_batch,
    preprocess_for_vjepa,
    summarize_preprocessed_clip,
)


@dataclass(slots=True)
class VJEPA2MaskedPredictionResult:
    """Structured masked-prediction outputs for future scoring."""

    predicted_tokens: Tensor
    target_tokens: Tensor
    num_masks: int
    mask_token_count: int
    hidden_size: int
    preprocessed_shape: tuple[int, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicted_tokens_shape": list(self.predicted_tokens.shape),
            "target_tokens_shape": list(self.target_tokens.shape),
            "num_masks": self.num_masks,
            "mask_token_count": self.mask_token_count,
            "hidden_size": self.hidden_size,
            "preprocessed_shape": list(self.preprocessed_shape),
        }


class VJEPA2Adapter:
    """Wrap official Hugging Face and torch.hub loading paths for V-JEPA 2."""

    def __init__(self, config: Optional[VJEPA2AdapterConfig] = None) -> None:
        self.config = (config or VJEPA2AdapterConfig()).validate()
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

        backends: List[str] = [self.config.backend]
        if self.config.fallback_backend not in backends:
            backends.append(self.config.fallback_backend)

        failures: List[str] = []
        for backend in backends:
            try:
                if backend == "huggingface":
                    self._load_huggingface()
                elif backend == "torch_hub":
                    self._load_torch_hub()
                else:
                    raise ValueError(f"Unsupported backend: {backend}")
                self.backend_used = backend
                return self.describe_runtime()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.last_load_errors[backend] = message
                failures.append(f"{backend}: {message}")

        joined = " | ".join(failures) if failures else "unknown error"
        raise RuntimeError(
            "Unable to load an official V-JEPA 2 model. "
            f"Tried backends {backends}. Failures: {joined}"
        )

    def _load_huggingface(self) -> None:
        try:
            from transformers import AutoModel, AutoVideoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face V-JEPA 2 loading requires `transformers`, `accelerate`, "
                "`huggingface_hub`, and `safetensors`."
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
                model = AutoModel.from_pretrained(self.config.model_id, **load_kwargs)
                break
            except TypeError:
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs.pop("attn_implementation", None)
                try:
                    model = AutoModel.from_pretrained(self.config.model_id, **fallback_kwargs)
                    break
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc

        if model is None:
            raise RuntimeError(f"Failed to load Hugging Face checkpoint {self.config.model_id}.") from last_error

        self.model = model.to(self.device)
        self.model.eval()

    def _load_torch_hub(self) -> None:
        try:
            processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor", trust_repo=True)
            model = torch.hub.load(
                "facebookresearch/vjepa2",
                self.config.fallback_model_name,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Official torch.hub fallback requires access to facebookresearch/vjepa2 and its extra "
                "dependencies such as `timm` and `einops`."
            ) from exc

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self.processor = processor
        self.model = model

    def preprocess(self, clips: Tensor | Any) -> Tensor:
        return preprocess_for_vjepa(
            clips,
            target_frames=self.config.target_frames,
            image_size=self.config.image_size,
        )

    def summarize_preprocessing(self, raw_clip: Tensor | Any) -> Dict[str, Any]:
        processed = self.preprocess(raw_clip)
        return summarize_preprocessed_clip(raw_clip, processed)

    def _processor_call(self, videos: Sequence[Tensor]) -> Mapping[str, Tensor]:
        if self.processor is None:
            raise RuntimeError("Processor is not loaded.")

        if self.backend_used == "huggingface":
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
        elif self.backend_used == "torch_hub":
            try:
                outputs = self.processor(videos)
            except TypeError:
                outputs = self.processor(videos=videos)
        else:
            raise RuntimeError("No V-JEPA backend has been selected.")

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

    def _move_masks_to_device(self, masks: Sequence[Tensor] | None) -> list[Tensor] | None:
        if masks is None:
            return None
        moved: list[Tensor] = []
        for mask in masks:
            tensor = torch.as_tensor(mask, dtype=torch.long, device=self.device)
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected mask tensor with shape (batch_size, num_token_indices); got {tuple(tensor.shape)}."
                )
            moved.append(tensor)
        return moved

    @staticmethod
    def _validate_mask_groups(
        context_masks: Sequence[Tensor],
        target_masks: Sequence[Tensor],
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if len(context_masks) != len(target_masks):
            raise ValueError(
                "context_masks and target_masks must have the same number of groups. "
                f"Got {len(context_masks)} and {len(target_masks)}."
            )
        for group_index, (context_mask, target_mask) in enumerate(zip(context_masks, target_masks)):
            context_tensor = torch.as_tensor(context_mask, dtype=torch.long)
            target_tensor = torch.as_tensor(target_mask, dtype=torch.long)
            if context_tensor.ndim != 2 or target_tensor.ndim != 2:
                raise ValueError(
                    "Predictor masks must be 2D tensors of shape (batch_size, num_token_indices). "
                    f"Group {group_index} got context {tuple(context_tensor.shape)} and target {tuple(target_tensor.shape)}."
                )
            if int(context_tensor.shape[0]) != batch_size or int(target_tensor.shape[0]) != batch_size:
                raise ValueError(
                    "Predictor mask batch dimensions must match the clip batch size. "
                    f"Expected {batch_size}, got context {tuple(context_tensor.shape)} and target "
                    f"{tuple(target_tensor.shape)} in group {group_index}."
                )
            if int(context_tensor.shape[1]) < 1 or int(target_tensor.shape[1]) < 1:
                raise ValueError(f"Predictor mask group {group_index} must contain at least one token.")
            if int(context_tensor.min().item()) < 0 or int(target_tensor.min().item()) < 0:
                raise ValueError(f"Predictor mask group {group_index} contains negative token indices.")
            if int(context_tensor.max().item()) >= sequence_length or int(target_tensor.max().item()) >= sequence_length:
                raise ValueError(
                    "Predictor mask group references token positions outside the V-JEPA sequence length. "
                    f"Sequence length is {sequence_length}, group {group_index} has context max "
                    f"{int(context_tensor.max().item())} and target max {int(target_tensor.max().item())}."
                )

    def _model_forward(
        self,
        preprocessed: Tensor,
        *,
        context_masks: Sequence[Tensor] | None = None,
        target_masks: Sequence[Tensor] | None = None,
        skip_predictor: bool = True,
    ) -> Any:
        if not self.loaded:
            self.load()

        videos = [video.detach().cpu() for video in ensure_video_batch(preprocessed)]
        inputs = self._processor_call(videos)
        inputs = self._move_inputs_to_device(inputs)

        model_kwargs: Dict[str, Any] = dict(inputs)
        context_masks = self._move_masks_to_device(context_masks)
        target_masks = self._move_masks_to_device(target_masks)
        if context_masks is not None:
            model_kwargs["context_mask"] = context_masks
        if target_masks is not None:
            model_kwargs["target_mask"] = target_masks

        with torch.no_grad():
            try:
                outputs = self.model(**model_kwargs, skip_predictor=skip_predictor)
            except TypeError:
                if skip_predictor:
                    outputs = self.model(**model_kwargs)
                else:
                    outputs = self.model(**model_kwargs)
        return outputs

    def _forward_embeddings(self, preprocessed: Tensor) -> Tensor:
        outputs = self._model_forward(preprocessed, skip_predictor=True)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Model output does not expose `last_hidden_state` for feature extraction.")

        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return F.normalize(pooled, dim=-1)

    def _reshape_masked_sequence(
        self,
        tensor: Tensor,
        *,
        batch_size: int,
        num_masks: int,
    ) -> Tensor:
        if tensor.ndim != 3:
            raise RuntimeError(f"Expected masked sequence tensor with 3 dims; got {tuple(tensor.shape)}.")
        if int(tensor.shape[0]) != int(batch_size * num_masks):
            raise RuntimeError(
                "Masked sequence batch dimension does not match batch_size * num_masks. "
                f"Got {tuple(tensor.shape)}, batch_size={batch_size}, num_masks={num_masks}."
            )
        return tensor.view(num_masks, batch_size, tensor.shape[1], tensor.shape[2]).transpose(0, 1).contiguous()

    def encode_batch(self, clips: Tensor | Any, *, batch_size: int = 4) -> Tensor:
        batch = ensure_video_batch(clips)
        preprocessed = self.preprocess(batch)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        outputs: List[Tensor] = []
        start = 0
        current_batch_size = min(batch_size, int(preprocessed.shape[0]))
        while start < int(preprocessed.shape[0]):
            end = min(start + current_batch_size, int(preprocessed.shape[0]))
            try:
                embeddings = self._forward_embeddings(preprocessed[start:end])
                outputs.append(embeddings)
                start = end
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

        return torch.cat(outputs, dim=0)

    def encode_video(self, clip: Tensor | Any) -> Tensor:
        return self.encode_batch(clip, batch_size=1)[0]

    def describe_token_layout(self) -> Dict[str, int]:
        if not self.loaded:
            self.load()
        model_config = getattr(self.model, "config", None)
        patch_size = int(getattr(model_config, "patch_size", 16))
        tubelet_size = int(getattr(model_config, "tubelet_size", 2))
        frames = int(self.config.target_frames)
        image_size = int(self.config.image_size)
        temporal_tokens = max(1, frames // max(tubelet_size, 1))
        spatial_tokens = max(1, (image_size // max(patch_size, 1)) ** 2)
        return {
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "frames": frames,
            "image_size": image_size,
            "temporal_tokens": temporal_tokens,
            "spatial_tokens_per_temporal_step": spatial_tokens,
            "sequence_length": temporal_tokens * spatial_tokens,
        }

    def predict_masked_tokens(
        self,
        clips: Tensor | Any,
        *,
        context_masks: Sequence[Tensor],
        target_masks: Sequence[Tensor],
        batch_size: int = 4,
    ) -> VJEPA2MaskedPredictionResult:
        if not context_masks or not target_masks:
            raise ValueError("context_masks and target_masks must both be non-empty.")
        if len(context_masks) != len(target_masks):
            raise ValueError("context_masks and target_masks must have the same number of mask groups.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        batch = ensure_video_batch(clips)
        preprocessed = self.preprocess(batch)
        num_masks = len(target_masks)
        sequence_length = int(self.describe_token_layout()["sequence_length"])
        self._validate_mask_groups(
            context_masks,
            target_masks,
            batch_size=int(preprocessed.shape[0]),
            sequence_length=sequence_length,
        )

        predicted_chunks: list[Tensor] = []
        target_chunks: list[Tensor] = []
        start = 0
        current_batch_size = min(batch_size, int(preprocessed.shape[0]))
        while start < int(preprocessed.shape[0]):
            end = min(start + current_batch_size, int(preprocessed.shape[0]))
            mini = preprocessed[start:end]
            mini_context_masks = [torch.as_tensor(mask[start:end], dtype=torch.long) for mask in context_masks]
            mini_target_masks = [torch.as_tensor(mask[start:end], dtype=torch.long) for mask in target_masks]

            try:
                outputs = self._model_forward(
                    mini,
                    context_masks=mini_context_masks,
                    target_masks=mini_target_masks,
                    skip_predictor=False,
                )
                predictor_output = getattr(outputs, "predictor_output", None)
                if predictor_output is None:
                    raise RuntimeError("Model output did not include predictor_output.")
                predicted = getattr(predictor_output, "last_hidden_state", None)
                target = getattr(predictor_output, "target_hidden_state", None)
                if predicted is None or target is None:
                    raise RuntimeError(
                        "Predictor output must expose both last_hidden_state and target_hidden_state."
                    )
                mini_batch_size = int(end - start)
                predicted = self._reshape_masked_sequence(predicted, batch_size=mini_batch_size, num_masks=num_masks)
                target = self._reshape_masked_sequence(target, batch_size=mini_batch_size, num_masks=num_masks)
                predicted_chunks.append(F.normalize(predicted, dim=-1))
                target_chunks.append(F.normalize(target, dim=-1))
                start = end
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

        predicted_tokens = torch.cat(predicted_chunks, dim=0)
        target_tokens = torch.cat(target_chunks, dim=0)
        return VJEPA2MaskedPredictionResult(
            predicted_tokens=predicted_tokens,
            target_tokens=target_tokens,
            num_masks=num_masks,
            mask_token_count=int(predicted_tokens.shape[2]),
            hidden_size=int(predicted_tokens.shape[-1]),
            preprocessed_shape=tuple(int(dim) for dim in preprocessed.shape),
        )

    def score_future_candidates(
        self,
        observed_clip: Tensor | Any,
        candidate_futures: Tensor | Any,
        *,
        scoring_variant: str = "masked_future_prediction",
        candidate_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Any:
        from jepa.scoring.vjepa_future_scorer import VJEPAFutureScorer

        scorer = VJEPAFutureScorer(adapter=self, scoring_variant=scoring_variant)
        return scorer.score_example(
            observed_clip,
            candidate_futures,
            candidate_metadata=candidate_metadata,
            scoring_variant=scoring_variant,
        )

    def describe_runtime(self) -> Dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        runtime = {
            "loaded": self.loaded,
            "backend_used": self.backend_used,
            "model_id": self.config.model_id,
            "fallback_model_name": self.config.fallback_model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype).replace("torch.", ""),
            "target_frames": self.config.target_frames,
            "image_size": self.config.image_size,
            "model_class": type(self.model).__name__ if self.model is not None else None,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "frames_per_clip": getattr(model_config, "frames_per_clip", None),
            "crop_size": getattr(model_config, "crop_size", None),
            "patch_size": getattr(model_config, "patch_size", None),
            "tubelet_size": getattr(model_config, "tubelet_size", None),
            "last_load_errors": dict(self.last_load_errors),
        }
        if self.loaded:
            runtime["token_layout"] = self.describe_token_layout()
        return runtime
