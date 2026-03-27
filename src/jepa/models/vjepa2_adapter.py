"""Notebook-first adapter for loading and encoding clips with V-JEPA 2."""

from __future__ import annotations

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

    def _forward_embeddings(self, preprocessed: Tensor) -> Tensor:
        if not self.loaded:
            self.load()

        videos = [video.detach().cpu() for video in ensure_video_batch(preprocessed)]
        inputs = self._processor_call(videos)
        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            try:
                outputs = self.model(**inputs, skip_predictor=True)
            except TypeError:
                outputs = self.model(**inputs)

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Model output does not expose `last_hidden_state` for feature extraction.")

        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return F.normalize(pooled, dim=-1)

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

    def score_future_candidates(
        self,
        observed_clip: Tensor | Any,
        candidate_futures: Tensor | Any,
        *,
        scoring_variant: str = "overlap_transition",
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
        return {
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
            "last_load_errors": dict(self.last_load_errors),
        }
