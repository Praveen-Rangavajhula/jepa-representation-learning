"""Configuration helpers for notebook-first V-JEPA 2 loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

import torch


BackendName = Literal["huggingface", "torch_hub"]


def is_colab_runtime() -> bool:
    """Best-effort detection for a Colab-backed runtime."""

    return bool(
        os.environ.get("COLAB_GPU")
        or os.environ.get("COLAB_RELEASE_TAG")
        or Path("/content").exists()
    )


def _default_cache_dir() -> str:
    if is_colab_runtime():
        return "/content/.cache/huggingface"
    return str((Path.cwd() / ".cache" / "huggingface").resolve())


def _default_dtype_name() -> str:
    return "float16" if torch.cuda.is_available() else "float32"


@dataclass(slots=True)
class VJEPA2AdapterConfig:
    """Runtime configuration for the V-JEPA 2 adapter."""

    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    backend: BackendName = "huggingface"
    fallback_backend: BackendName = "torch_hub"
    fallback_model_name: str = "vjepa2_vit_large"
    target_frames: int = 64
    image_size: int = 256
    dtype: str = field(default_factory=_default_dtype_name)
    attn_implementation: str = "sdpa"
    cache_dir: str = field(default_factory=_default_cache_dir)
    device: str | None = None
    local_files_only: bool = False

    def validate(self) -> "VJEPA2AdapterConfig":
        if self.backend not in {"huggingface", "torch_hub"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.fallback_backend not in {"huggingface", "torch_hub"}:
            raise ValueError(f"Unsupported fallback backend: {self.fallback_backend}")
        if self.target_frames < 2:
            raise ValueError("target_frames must be at least 2.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32.")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("dtype must be one of: float16, float32, bfloat16.")
        return self

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolved_dtype_name(self) -> str:
        device = self.resolved_device()
        if device.type == "cpu" and self.dtype == "float16":
            return "float32"
        return self.dtype

    def resolved_torch_dtype(self) -> torch.dtype:
        dtype_name = self.resolved_dtype_name()
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping[dtype_name]

    def as_dict(self) -> Dict[str, Any]:
        device = self.resolved_device()
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "fallback_backend": self.fallback_backend,
            "fallback_model_name": self.fallback_model_name,
            "target_frames": self.target_frames,
            "image_size": self.image_size,
            "dtype": self.resolved_dtype_name(),
            "attn_implementation": self.attn_implementation,
            "cache_dir": self.cache_dir,
            "device": str(device),
            "local_files_only": self.local_files_only,
            "is_colab_runtime": is_colab_runtime(),
        }
