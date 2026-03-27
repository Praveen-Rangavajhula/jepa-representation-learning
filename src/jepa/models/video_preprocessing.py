"""Reusable clip preprocessing for V-JEPA 2 style video encoders."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


def ensure_video_batch(clips: Tensor | Any) -> Tensor:
    """Normalize common inputs to ``(B, T, C, H, W)``."""

    tensor = clips if isinstance(clips, torch.Tensor) else torch.as_tensor(clips)
    tensor = tensor.detach().clone() if tensor.requires_grad else tensor
    tensor = tensor.to(dtype=torch.float32)

    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(
            f"Expected clip tensor with shape (T, C, H, W) or (B, T, C, H, W); got {tuple(tensor.shape)}."
        )

    if not torch.isfinite(tensor).all():
        raise ValueError("Video tensor contains non-finite values.")
    if float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
        raise ValueError("Expected video tensor values in [0, 1].")
    return tensor.contiguous()


def temporal_resample_indices(num_frames: int, target_frames: int) -> Tensor:
    """Build deterministic frame indices for repeat/interleave or downsampling."""

    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    if target_frames < 1:
        raise ValueError("target_frames must be positive.")
    if num_frames == target_frames:
        return torch.arange(num_frames, dtype=torch.long)

    base = torch.arange(target_frames, dtype=torch.long)
    indices = torch.div(base * num_frames, target_frames, rounding_mode="floor")
    return indices.clamp_(0, num_frames - 1)


def temporal_resample_batch(batch: Tensor, target_frames: int) -> Tensor:
    batch = ensure_video_batch(batch)
    indices = temporal_resample_indices(batch.shape[1], target_frames).to(batch.device)
    return batch.index_select(dim=1, index=indices)


def convert_grayscale_to_rgb(batch: Tensor) -> Tensor:
    batch = ensure_video_batch(batch)
    channels = int(batch.shape[2])
    if channels == 3:
        return batch
    if channels == 1:
        return batch.repeat(1, 1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, received {channels}.")


def resize_video_batch(batch: Tensor, image_size: int) -> Tensor:
    batch = ensure_video_batch(batch)
    batch = convert_grayscale_to_rgb(batch)
    bsz, frames, channels, _, _ = batch.shape
    flattened = batch.view(bsz * frames, channels, batch.shape[-2], batch.shape[-1])
    resized = F.interpolate(
        flattened,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.view(bsz, frames, channels, image_size, image_size)


def preprocess_for_vjepa(
    clips: Tensor | Any,
    *,
    target_frames: int = 64,
    image_size: int = 256,
) -> Tensor:
    """Apply the notebook-friendly V-JEPA 2 preprocessing contract."""

    batch = ensure_video_batch(clips)
    batch = temporal_resample_batch(batch, target_frames=target_frames)
    batch = resize_video_batch(batch, image_size=image_size)
    return batch.clamp(0.0, 1.0).contiguous()


def clip_to_processor_videos(
    clips: Tensor | Any,
    *,
    target_frames: int = 64,
    image_size: int = 256,
) -> List[Tensor]:
    """Return a list of per-video tensors ready for a video processor call."""

    batch = preprocess_for_vjepa(clips, target_frames=target_frames, image_size=image_size)
    return [video.detach().cpu() for video in batch]


def summarize_preprocessed_clip(raw_clip: Tensor | Any, processed_clip: Tensor | Any) -> Dict[str, Any]:
    """Compact shape/range summary for notebook sanity checks."""

    raw = ensure_video_batch(raw_clip)
    processed = ensure_video_batch(processed_clip)
    return {
        "raw_shape": list(raw.shape),
        "processed_shape": list(processed.shape),
        "raw_min": float(raw.min().item()),
        "raw_max": float(raw.max().item()),
        "processed_min": float(processed.min().item()),
        "processed_max": float(processed.max().item()),
    }
