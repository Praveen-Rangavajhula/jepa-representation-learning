"""Model adapters and preprocessing helpers for JEPA experiments."""

from .model_config import (
    DEFAULT_GENERIC_VJEPA_MODEL_ID,
    DEFAULT_SSV2_VJEPA_MODEL_ID,
    VJEPA2AdapterConfig,
    is_colab_runtime,
    make_generic_vjepa_config,
    make_ssv2_vjepa_config,
)
from .video_preprocessing import (
    clip_to_processor_videos,
    convert_grayscale_to_rgb,
    ensure_video_batch,
    preprocess_for_vjepa,
    resize_video_batch,
    summarize_preprocessed_clip,
    temporal_resample_batch,
    temporal_resample_indices,
)
from .vjepa2_adapter import VJEPA2Adapter, VJEPA2MaskedPredictionResult

__all__ = [
    "VJEPA2Adapter",
    "VJEPA2AdapterConfig",
    "VJEPA2MaskedPredictionResult",
    "DEFAULT_GENERIC_VJEPA_MODEL_ID",
    "DEFAULT_SSV2_VJEPA_MODEL_ID",
    "clip_to_processor_videos",
    "convert_grayscale_to_rgb",
    "ensure_video_batch",
    "is_colab_runtime",
    "make_generic_vjepa_config",
    "make_ssv2_vjepa_config",
    "preprocess_for_vjepa",
    "resize_video_batch",
    "summarize_preprocessed_clip",
    "temporal_resample_batch",
    "temporal_resample_indices",
]
