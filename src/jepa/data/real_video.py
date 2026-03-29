"""Real-world video dataset adapters for future-selection experiments.

This module keeps the current future-selection contract intact:

- observed clip: ``(T_obs, C, H, W)``
- candidate futures: ``(K, T_future, C, H, W)``
- one correct future and several plausible negatives

The default intended source is Something-Something V2 hosted in standard
Hugging Face video/parquet format, but the adapter layer is generic
enough to support UCF101 fallback or other local video manifests later
without changing the downstream task, scoring, or evaluation code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_template(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _as_tuple(values: Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(str(item) for item in values)
    return (str(values),)


def _ensure_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "clip"


def _resolve_path(base: Path, value: Any) -> Path:
    path = _ensure_path(value)
    if path.is_absolute():
        return path
    return base / path


def _format_description(template: str, placeholders: Sequence[str], *, suffix: str = "") -> str:
    base = template.strip()
    if placeholders:
        base = f"{base} ({'; '.join(str(item) for item in placeholders)})"
    if suffix:
        base = f"{base} {suffix}".strip()
    return base


def _neutral_observed_description(*, frame_count: int, pair_group: str) -> str:
    _ = pair_group
    return (
        f"Observed prefix: a short {frame_count}-frame clip showing a human-object interaction in "
        "progress. The commentary layer only receives a neutral motion summary rather than the exact "
        "action label."
    )


def _metadata_description(candidate_metadata: Mapping[str, Any]) -> str | None:
    details = candidate_metadata.get("details")
    if isinstance(details, Mapping):
        description = details.get("description")
        if description:
            return str(description)
    description = candidate_metadata.get("description")
    if description:
        return str(description)
    return None


def _to_tchw(frames: Tensor) -> Tensor:
    frames = torch.as_tensor(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(frames.shape)}.")

    if frames.shape[1] in {1, 3} and frames.shape[-1] not in {1, 3}:
        return frames
    if frames.shape[-1] in {1, 3}:
        return frames.permute(0, 3, 1, 2).contiguous()
    if frames.shape[0] in {1, 3}:
        return frames.permute(1, 0, 2, 3).contiguous()

    raise ValueError(
        "Could not infer channel layout for video tensor. Expected one of "
        "(T, C, H, W), (T, H, W, C), or (C, T, H, W)."
    )


def _normalize_video_tensor(frames: Tensor) -> Tensor:
    frames = torch.as_tensor(frames)
    if frames.dtype.is_floating_point:
        frames = frames.to(dtype=torch.float32)
    else:
        frames = frames.to(dtype=torch.float32).div(255.0)
    return frames.clamp(0.0, 1.0)


def _resize_video_tensor(frames: Tensor, image_size: int) -> Tensor:
    if frames.shape[-2:] == (image_size, image_size):
        return frames
    return F.interpolate(
        frames.to(dtype=torch.float32),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def _uniform_sample_indices(num_frames: int, target_frames: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("Cannot sample frames from an empty video.")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")
    if num_frames == target_frames:
        return np.arange(target_frames, dtype=np.int64)
    indices = np.linspace(0, max(num_frames - 1, 0), num=target_frames)
    return np.clip(np.round(indices).astype(np.int64), 0, max(num_frames - 1, 0))


def _sample_video_to_length(frames: Tensor, target_frames: int) -> Tensor:
    frames = torch.as_tensor(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) tensor, got {tuple(frames.shape)}.")
    indices = _uniform_sample_indices(int(frames.shape[0]), target_frames)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=frames.device)
    return frames.index_select(0, index_tensor)


def _center_local_window(frames: Tensor, window_length: int) -> Tensor:
    frames = torch.as_tensor(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) tensor, got {tuple(frames.shape)}.")
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    num_frames = int(frames.shape[0])
    if num_frames <= window_length:
        return frames
    start = max(0, (num_frames - window_length) // 2)
    stop = start + window_length
    return frames[start:stop]


def _decode_torchcodec_video(video: Any, *, max_chunk_size: int = 64) -> Tensor:
    if not hasattr(video, "get_frames_in_range"):
        raise TypeError("Video object does not expose get_frames_in_range().")

    chunks: List[Tensor] = []
    start = 0
    while True:
        try:
            batch = video.get_frames_in_range(start, start + max_chunk_size, 1)
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            message = str(exc).lower()
            if chunks and "no more frames left to decode" in message:
                break
            raise RealVideoDataError(
                "Failed to decode a video via the torchcodec/VideoDecoder path. "
                "Verify that the dataset exposes a readable video object."
            ) from exc

        frames = getattr(batch, "data", batch)
        frames = torch.as_tensor(frames)
        if frames.numel() == 0:
            break
        chunks.append(frames)
        if int(frames.shape[0]) < max_chunk_size:
            break
        start += max_chunk_size

    if not chunks:
        raise RealVideoDataError("Could not decode any frames from the video object.")

    return torch.cat(chunks, dim=0)


def _decode_video_path(video_path: Path) -> Tensor:
    try:
        from torchvision.io import read_video
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RealVideoDependencyError(
            "torchvision video decoding is not available. Install torchvision with video support "
            "or provide a dataset that exposes decoded torchcodec videos."
        ) from exc

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise RealVideoDataError(f"Video file {video_path} decoded to zero frames.")
    _ = info
    return frames


def _decode_video_value(video_value: Any, *, video_path: Optional[Path] = None) -> Tensor:
    if isinstance(video_value, Tensor):
        return video_value
    if hasattr(video_value, "get_frames_in_range"):
        return _decode_torchcodec_video(video_value)
    if isinstance(video_value, (str, Path)) or hasattr(video_value, "__fspath__"):
        return _decode_video_path(_ensure_path(video_value))
    if isinstance(video_value, Mapping) and video_value.get("path"):
        return _decode_video_path(_ensure_path(video_value["path"]))
    if video_path is not None:
        return _decode_video_path(video_path)

    raise RealVideoDataError(
        "Unsupported video value. Expected a torchcodec video object, a filesystem path, "
        "or a mapping with a 'path' field."
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False))
            handle.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RealVideoDataError(RuntimeError):
    """Raised when a real-video dataset cannot be loaded or decoded."""


class RealVideoDependencyError(RealVideoDataError):
    """Raised when an optional dependency is missing."""


@dataclass(frozen=True)
class RealVideoTemplateSpec:
    label_template: str
    pair_group: str
    paired_template: str
    description_hint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label_template": self.label_template,
            "pair_group": self.pair_group,
            "paired_template": self.paired_template,
            "description_hint": self.description_hint,
        }


DEFAULT_SOMETHING_SOMETHING_TEMPLATE_SPECS: Tuple[RealVideoTemplateSpec, ...] = (
    RealVideoTemplateSpec(
        label_template="Opening something",
        pair_group="open_close",
        paired_template="Closing something",
        description_hint="The visible object begins to open or reveal itself.",
    ),
    RealVideoTemplateSpec(
        label_template="Closing something",
        pair_group="open_close",
        paired_template="Opening something",
        description_hint="The visible object begins to close or cover itself.",
    ),
    RealVideoTemplateSpec(
        label_template="Moving something towards the camera",
        pair_group="camera_motion",
        paired_template="Moving something away from the camera",
        description_hint="An object moves closer to the viewpoint.",
    ),
    RealVideoTemplateSpec(
        label_template="Moving something away from the camera",
        pair_group="camera_motion",
        paired_template="Moving something towards the camera",
        description_hint="An object moves away from the viewpoint.",
    ),
    RealVideoTemplateSpec(
        label_template="Pushing something from left to right",
        pair_group="horizontal_push",
        paired_template="Pushing something from right to left",
        description_hint="Motion proceeds laterally from left to right.",
    ),
    RealVideoTemplateSpec(
        label_template="Pushing something from right to left",
        pair_group="horizontal_push",
        paired_template="Pushing something from left to right",
        description_hint="Motion proceeds laterally from right to left.",
    ),
    RealVideoTemplateSpec(
        label_template="Putting something into something",
        pair_group="container_motion",
        paired_template="Taking something out of something",
        description_hint="An object moves into a target container or space.",
    ),
    RealVideoTemplateSpec(
        label_template="Taking something out of something",
        pair_group="container_motion",
        paired_template="Putting something into something",
        description_hint="An object is removed from a container or space.",
    ),
)


DEFAULT_UCF101_TEMPLATE_SPECS: Tuple[RealVideoTemplateSpec, ...] = (
    RealVideoTemplateSpec(
        label_template="ApplyEyeMakeup",
        pair_group="face_grooming",
        paired_template="ApplyLipstick",
        description_hint="Applying eye makeup to the face.",
    ),
    RealVideoTemplateSpec(
        label_template="ApplyLipstick",
        pair_group="face_grooming",
        paired_template="ApplyEyeMakeup",
        description_hint="Applying lipstick to the face.",
    ),
    RealVideoTemplateSpec(
        label_template="BoxingPunchingBag",
        pair_group="boxing_motion",
        paired_template="BoxingSpeedBag",
        description_hint="Repeated punching against a punching bag.",
    ),
    RealVideoTemplateSpec(
        label_template="BoxingSpeedBag",
        pair_group="boxing_motion",
        paired_template="BoxingPunchingBag",
        description_hint="Repeated striking of a speed bag.",
    ),
    RealVideoTemplateSpec(
        label_template="BreastStroke",
        pair_group="swimming_style",
        paired_template="FrontCrawl",
        description_hint="Swimming with breaststroke-style arm and body motion.",
    ),
    RealVideoTemplateSpec(
        label_template="FrontCrawl",
        pair_group="swimming_style",
        paired_template="BreastStroke",
        description_hint="Swimming with front-crawl-style arm and body motion.",
    ),
    RealVideoTemplateSpec(
        label_template="PullUps",
        pair_group="upper_body_exercise",
        paired_template="PushUps",
        description_hint="Up-and-down pull-up exercise motion.",
    ),
    RealVideoTemplateSpec(
        label_template="PushUps",
        pair_group="upper_body_exercise",
        paired_template="PullUps",
        description_hint="Up-and-down push-up exercise motion.",
    ),
)


@dataclass(slots=True)
class RealVideoSubsetConfig:
    """Configuration for a lightweight real-video subset."""

    dataset_name: str = "mteb/SomethingSomethingV2"
    cache_root: str = "data/real_video_cache"
    source_window_length: int = 32
    sequence_length: int = 16
    observed_length: int = 8
    future_length: int = 8
    image_size: int = 256
    train_examples_per_template: int = 24
    eval_examples_per_template: int = 16
    confident_eval_examples_per_template: int = 40
    max_source_scan: int = 20_000
    seed: int = 13
    split_column: str = "split"
    text_column: str = "label"
    video_id_column: str = "video_id"
    placeholders_column: str = "placeholders"
    video_column: str = "video"
    template_specs: Tuple[RealVideoTemplateSpec, ...] = DEFAULT_SOMETHING_SOMETHING_TEMPLATE_SPECS
    streaming: bool = False
    local_manifest_path: Optional[str] = None

    def validate(self) -> "RealVideoSubsetConfig":
        if self.source_window_length < 1:
            raise ValueError("source_window_length must be positive.")
        if self.sequence_length != self.observed_length + self.future_length:
            raise ValueError("sequence_length must equal observed_length + future_length.")
        if self.observed_length < 1 or self.future_length < 1:
            raise ValueError("observed_length and future_length must both be positive.")
        if self.image_size < 28:
            raise ValueError("image_size must be at least 28.")
        if self.train_examples_per_template < 1 or self.eval_examples_per_template < 1:
            raise ValueError("examples per template must be positive.")
        if self.confident_eval_examples_per_template < self.eval_examples_per_template:
            raise ValueError("confident eval size must be at least the default eval size.")
        if not self.template_specs:
            raise ValueError("template_specs must not be empty.")
        return self

    @property
    def template_lookup(self) -> Dict[str, RealVideoTemplateSpec]:
        return {_normalize_template(spec.label_template): spec for spec in self.template_specs}

    @property
    def pair_lookup(self) -> Dict[str, Tuple[str, ...]]:
        groups: Dict[str, List[str]] = {}
        for spec in self.template_specs:
            groups.setdefault(spec.pair_group, []).append(spec.label_template)
        return {group: tuple(values) for group, values in groups.items()}

    def target_examples_for_split(self, split: str) -> int:
        if split in {"train", "training"}:
            return self.train_examples_per_template * len(self.template_specs)
        if split in {"validation", "val", "test", "evaluation", "eval"}:
            return self.eval_examples_per_template * len(self.template_specs)
        if split in {"confident_eval", "large_eval"}:
            return self.confident_eval_examples_per_template * len(self.template_specs)
        return self.eval_examples_per_template * len(self.template_specs)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "cache_root": self.cache_root,
            "source_window_length": self.source_window_length,
            "sequence_length": self.sequence_length,
            "observed_length": self.observed_length,
            "future_length": self.future_length,
            "image_size": self.image_size,
            "train_examples_per_template": self.train_examples_per_template,
            "eval_examples_per_template": self.eval_examples_per_template,
            "confident_eval_examples_per_template": self.confident_eval_examples_per_template,
            "max_source_scan": self.max_source_scan,
            "seed": self.seed,
            "split_column": self.split_column,
            "text_column": self.text_column,
            "video_id_column": self.video_id_column,
            "placeholders_column": self.placeholders_column,
            "video_column": self.video_column,
            "streaming": self.streaming,
            "local_manifest_path": self.local_manifest_path,
            "template_specs": [spec.as_dict() for spec in self.template_specs],
        }


@dataclass(slots=True)
class RealVideoClipRecord:
    """A cached or source-side real-video clip plus descriptive metadata."""

    video_id: str
    label_template: str
    pair_group: str
    paired_template: str
    placeholders: Tuple[str, ...]
    description: str
    split: str
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    clip: Optional[Tensor] = None
    cache_path: Optional[str] = None
    source_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "label_template": self.label_template,
            "pair_group": self.pair_group,
            "paired_template": self.paired_template,
            "placeholders": list(self.placeholders),
            "description": self.description,
            "split": self.split,
            "raw_metadata": _jsonable(self.raw_metadata),
            "cache_path": self.cache_path,
            "source_path": self.source_path,
        }


class RealVideoFutureSelectionDataset(Dataset):
    """Dataset of cached real-video clips converted into future-selection examples."""

    def __init__(
        self,
        records: Sequence[RealVideoClipRecord],
        config: RealVideoSubsetConfig,
    ) -> None:
        self.records = list(records)
        self.config = config.validate()
        self._template_groups: Dict[str, List[int]] = {}
        self._pair_groups: Dict[str, List[int]] = {}
        for index, record in enumerate(self.records):
            self._template_groups.setdefault(_normalize_template(record.label_template), []).append(index)
            self._pair_groups.setdefault(record.pair_group, []).append(index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self.records):
            raise IndexError(index)

        source = self.records[index]
        if source.clip is None:
            raise RealVideoDataError(
                f"Cached record {source.video_id!r} does not contain a clip tensor. "
                "Rebuild the cache for the selected split."
            )

        rng = np.random.default_rng(int(self.config.seed + index))
        observed, future = self._split_clip(source.clip)

        candidates: List[Tensor] = []
        candidate_metadata: List[Dict[str, Any]] = []

        true_candidate = future
        candidates.append(true_candidate)
        candidate_metadata.append(
            self._candidate_metadata(
                source=source,
                strategy="true_continuation",
                generation_type="true_continuation",
                is_true=True,
                description_suffix="true continuation",
                source_video_id=source.video_id,
                details={"candidate_role": "observed_clip_true_future"},
            )
        )

        temporal_negative, temporal_negative_mode, temporal_negative_details = self._build_temporal_negative(future, rng)
        candidates.append(temporal_negative)
        candidate_metadata.append(
            self._candidate_metadata(
                source=source,
                strategy=temporal_negative_mode,
                generation_type=temporal_negative_mode,
                is_true=False,
                description_suffix=self._strategy_description(temporal_negative_mode),
                source_video_id=source.video_id,
                details={
                    "candidate_role": "same_clip_temporal_negative",
                    **temporal_negative_details,
                },
            )
        )

        same_template_other = self._pick_same_template_other(index=index, rng=rng)
        same_template_clip = self._future_from_record(same_template_other)
        candidates.append(same_template_clip)
        candidate_metadata.append(
            self._candidate_metadata(
                source=same_template_other,
                strategy="future_segment_from_other_sample",
                generation_type="future_segment_from_other_sample",
                is_true=False,
                description_suffix="same-template other-sample future",
                source_video_id=same_template_other.video_id,
                details={"fallback": False, "source_rank": "same_template"},
            )
        )

        paired_source, paired_strategy = self._pick_paired_or_fallback(index=index, rng=rng)
        paired_clip = self._future_from_record(paired_source)
        candidates.append(paired_clip)
        candidate_metadata.append(
            self._candidate_metadata(
                source=paired_source,
                strategy=paired_strategy,
                generation_type=paired_strategy,
                is_true=False,
                description_suffix=self._strategy_description(paired_strategy),
                source_video_id=paired_source.video_id,
                details={"fallback": paired_strategy != "paired_template_counterfactual"},
            )
        )

        permutation = list(range(len(candidates)))
        rng.shuffle(permutation)
        shuffled_candidates = [candidates[i] for i in permutation]
        shuffled_metadata: List[Dict[str, Any]] = []
        correct_index = 0
        for new_index, original_index in enumerate(permutation):
            item = dict(candidate_metadata[original_index])
            details = dict(item.get("details", {}) or {})
            details["construction_index"] = int(original_index)
            item["details"] = details
            item["candidate_index"] = int(new_index)
            item["construction_index"] = int(original_index)
            shuffled_metadata.append(item)
            if original_index == 0:
                correct_index = int(new_index)

        candidates_tensor = torch.stack(shuffled_candidates, dim=0)
        observed_description = _neutral_observed_description(
            frame_count=int(observed.shape[0]),
            pair_group=source.pair_group,
        )

        metadata = {
            "dataset_name": self.config.dataset_name,
            "split": source.split,
            "source_video_id": source.video_id,
            "label_template": source.label_template,
            "paired_template": source.paired_template,
            "placeholders": list(source.placeholders),
            "pair_group": source.pair_group,
            "observed_description": observed_description,
            "source_description": source.description,
            "candidate_strategies": shuffled_metadata,
            "cache_path": source.cache_path,
            "raw_metadata": _jsonable(source.raw_metadata),
            "sequence_length": self.config.sequence_length,
            "observed_length": self.config.observed_length,
            "future_length": self.config.future_length,
            "observed_shape_convention": "(T_obs, C, H, W)",
            "candidates_shape_convention": "(K, T_future, C, H, W)",
            "candidate_construction_order": [
                item.get("strategy", "unknown") for item in candidate_metadata
            ],
        }

        from jepa.tasks.future_selection import FutureSelectionExample

        return FutureSelectionExample(
            observed=observed,
            candidates=candidates_tensor,
            correct_index=correct_index,
            metadata=metadata,
        )

    def _split_clip(self, clip: Tensor) -> Tuple[Tensor, Tensor]:
        clip = torch.as_tensor(clip)
        if clip.ndim != 4:
            raise RealVideoDataError(
                f"Expected cached clip to have shape (T, C, H, W); got {tuple(clip.shape)}."
            )
        if clip.shape[0] < self.config.sequence_length:
            raise RealVideoDataError(
                f"Cached clip for future selection requires at least {self.config.sequence_length} frames, "
                f"but got {clip.shape[0]}."
            )
        clip = clip[: self.config.sequence_length]
        return clip[: self.config.observed_length], clip[self.config.observed_length : self.config.sequence_length]

    def _future_from_record(self, record: RealVideoClipRecord) -> Tensor:
        if record.clip is None:
            raise RealVideoDataError(f"Cached record {record.video_id!r} is missing its clip tensor.")
        _, future = self._split_clip(record.clip)
        return future

    def _build_temporal_negative(
        self,
        future: Tensor,
        rng: np.random.Generator,
    ) -> Tuple[Tensor, str, Dict[str, Any]]:
        future = torch.as_tensor(future)
        if future.ndim != 4:
            raise RealVideoDataError(f"Expected future clip to have shape (T, C, H, W); got {tuple(future.shape)}.")

        frame_count = int(future.shape[0])
        if frame_count >= 4 and frame_count % 2 == 0:
            block_size = frame_count // 2
            first_block = future[:block_size]
            second_block = future[block_size:]
            mode = "temporal_order_two_block_swap" if int(rng.integers(0, 2)) == 0 else "temporal_order_block_reverse"
            if mode == "temporal_order_two_block_swap":
                negative = torch.cat([second_block, first_block], dim=0)
            else:
                negative = torch.cat([first_block.flip(0), second_block.flip(0)], dim=0)
            return (
                negative.contiguous(),
                mode,
                {
                    "temporal_negative_mode": mode,
                    "block_count": 2,
                    "block_size": block_size,
                    "source_frame_count": frame_count,
                },
            )

        negative = future.flip(0)
        mode = "temporal_order_reverse"
        return (
            negative.contiguous(),
            mode,
            {
                "temporal_negative_mode": mode,
                "block_count": 1,
                "block_size": frame_count,
                "source_frame_count": frame_count,
            },
        )

    def _pick_same_template_other(self, *, index: int, rng: np.random.Generator) -> RealVideoClipRecord:
        source = self.records[index]
        same_template = [
            candidate_index
            for candidate_index in self._template_groups.get(_normalize_template(source.label_template), [])
            if candidate_index != index
        ]
        if same_template:
            chosen_index = int(rng.choice(same_template))
            return self.records[chosen_index]

        fallback_pool = [
            candidate_index
            for candidate_index in self._pair_groups.get(source.pair_group, [])
            if candidate_index != index
        ]
        if fallback_pool:
            chosen_index = int(rng.choice(fallback_pool))
            return self.records[chosen_index]

        other_indices = [candidate_index for candidate_index in range(len(self.records)) if candidate_index != index]
        if not other_indices:
            raise RealVideoDataError("Could not find an alternative record for negative candidate generation.")
        chosen_index = int(rng.choice(other_indices))
        return self.records[chosen_index]

    def _pick_paired_or_fallback(
        self,
        *,
        index: int,
        rng: np.random.Generator,
    ) -> Tuple[RealVideoClipRecord, str]:
        source = self.records[index]
        paired_template = _normalize_template(source.paired_template)
        paired_pool = [
            candidate_index
            for candidate_index in self._template_groups.get(paired_template, [])
            if candidate_index != index
        ]
        if paired_pool:
            return self.records[int(rng.choice(paired_pool))], "paired_template_counterfactual"

        same_pair_pool = [
            candidate_index
            for candidate_index in self._pair_groups.get(source.pair_group, [])
            if candidate_index != index
        ]
        if same_pair_pool:
            return self.records[int(rng.choice(same_pair_pool))], "same_pair_fallback"

        other_indices = [candidate_index for candidate_index in range(len(self.records)) if candidate_index != index]
        if other_indices:
            return self.records[int(rng.choice(other_indices))], "random_future_fallback"

        raise RealVideoDataError("Could not build a paired or fallback negative candidate.")

    def _strategy_description(self, strategy: str) -> str:
        descriptions = {
            "paired_template_counterfactual": "paired counterfactual",
            "same_pair_fallback": "same-pair fallback future",
            "random_future_fallback": "generic random future",
            "temporal_order_two_block_swap": "two-block temporal swap",
            "temporal_order_block_reverse": "reverse-block temporal negative",
            "temporal_order_reverse": "reversed temporal order",
        }
        return descriptions.get(strategy, strategy)

    def _candidate_metadata(
        self,
        *,
        source: RealVideoClipRecord,
        strategy: str,
        generation_type: str,
        is_true: bool,
        description_suffix: str,
        source_video_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        description = _format_description(source.label_template, source.placeholders, suffix=f"({description_suffix})")
        return {
            "strategy": strategy,
            "generation_type": generation_type,
            "source_video_id": source_video_id,
            "label_template": source.label_template,
            "placeholders": list(source.placeholders),
            "pair_group": source.pair_group,
            "description": description,
            "is_true": is_true,
            "details": {
                "description": description,
                "paired_template": source.paired_template,
                "source_description": source.description,
                **(details or {}),
            },
        }


class RealVideoDatasetAdapter:
    """Base adapter for clip-based real video sources."""

    def __init__(self, config: RealVideoSubsetConfig) -> None:
        self.config = config.validate()
        self._cache_root = _repo_root() / self.config.cache_root

    def cache_root_for_split(self, split: str) -> Path:
        window_tag = f"srcwin{self.config.source_window_length}"
        return self._cache_root / f"{self.dataset_slug}_{window_tag}" / split

    def source_split_for_request(self, split: str) -> str:
        normalized = split.strip().lower()
        dataset_name = _normalize_template(self.config.dataset_name)
        if "somethingsomethingv2" in dataset_name or "something_something_v2" in dataset_name:
            return "test"
        if "ucf-101" in dataset_name or "ucf101" in dataset_name:
            return "train"
        if normalized in {"validation", "val", "test", "evaluation", "eval", "confident_eval", "large_eval"}:
            return "validation"
        if normalized in {"train", "training"}:
            return "train"
        return split

    @property
    def dataset_slug(self) -> str:
        return _normalize_template(self.config.dataset_name).replace("/", "_").replace(" ", "_")

    def prepare_cache(self, split: str, *, force_rebuild: bool = False) -> Path:
        target_dir = self.cache_root_for_split(split)
        manifest_path = target_dir / "manifest.jsonl"
        if manifest_path.exists() and not force_rebuild:
            return manifest_path

        source_records = self._load_source_records(split)
        if not source_records:
            raise RealVideoDataError(
                f"No source records were loaded for split {split!r}. "
                "Check the dataset name, split, and template filters."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        clip_dir = target_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: List[Dict[str, Any]] = []
        for record in source_records:
            if record.clip is None:
                record = self._materialize_record(record)
            if record.clip is None:
                raise RealVideoDataError(
                    f"Record {record.video_id!r} did not materialize into a clip tensor."
                )
            clip_path = clip_dir / f"{_safe_filename(record.video_id)}.pt"
            torch.save(record.clip, clip_path)
            manifest_rows.append(
                {
                    **record.as_dict(),
                    "cache_path": str(clip_path),
                    "clip_path": str(clip_path.relative_to(target_dir)),
                }
            )

        _write_jsonl(manifest_path, manifest_rows)
        return manifest_path

    def build_dataset(self, split: str) -> RealVideoFutureSelectionDataset:
        cached_records = self.load_cached_records(split)
        return RealVideoFutureSelectionDataset(cached_records, self.config)

    def load_cached_records(self, split: str) -> List[RealVideoClipRecord]:
        manifest_path = self.prepare_cache(split)
        target_dir = manifest_path.parent
        rows = _read_jsonl(manifest_path)
        records: List[RealVideoClipRecord] = []
        for row in rows:
            cache_path = _resolve_path(target_dir, row.get("clip_path", row.get("cache_path", "")))
            if not cache_path.exists():
                raise RealVideoDataError(
                    f"Cached clip file is missing: {cache_path}. Rebuild the cache for split {split!r}."
                )
            clip = torch.load(cache_path, map_location="cpu")
            records.append(
                RealVideoClipRecord(
                    video_id=str(row["video_id"]),
                    label_template=str(row["label_template"]),
                    pair_group=str(row["pair_group"]),
                    paired_template=str(row["paired_template"]),
                    placeholders=_as_tuple(row.get("placeholders")),
                    description=str(row["description"]),
                    split=str(row.get("split", split)),
                    raw_metadata=dict(row.get("raw_metadata") or {}),
                    clip=clip,
                    cache_path=str(cache_path),
                    source_path=row.get("source_path"),
                )
            )
        return records

    def _materialize_record(self, record: RealVideoClipRecord) -> RealVideoClipRecord:
        if record.clip is not None:
            return record
        if not record.source_path:
            raise RealVideoDataError(
                f"Record {record.video_id!r} is missing both a clip tensor and a source path."
            )
        clip = self._decode_clip(Path(record.source_path), record.raw_metadata)
        clip = self._prepare_clip(clip)
        return RealVideoClipRecord(
            video_id=record.video_id,
            label_template=record.label_template,
            pair_group=record.pair_group,
            paired_template=record.paired_template,
            placeholders=record.placeholders,
            description=record.description,
            split=record.split,
            raw_metadata=dict(record.raw_metadata),
            clip=clip,
            cache_path=record.cache_path,
            source_path=record.source_path,
        )

    def _prepare_clip(self, clip: Tensor) -> Tensor:
        clip = _to_tchw(clip)
        clip = _normalize_video_tensor(clip)
        clip = _center_local_window(clip, self.config.source_window_length)
        clip = _sample_video_to_length(clip, self.config.sequence_length)
        clip = _resize_video_tensor(clip, self.config.image_size)
        return clip.contiguous()

    def _decode_clip(self, path: Path, raw_metadata: Mapping[str, Any]) -> Tensor:
        if path.exists():
            return _decode_video_path(path)
        raise RealVideoDataError(
            f"Could not decode clip for record with path {path}. "
            "Ensure the file exists or provide a cached clip tensor."
        )

    def _load_source_records(self, split: str) -> List[RealVideoClipRecord]:
        raise NotImplementedError

    def _template_targets_for_split(self, split: str) -> Dict[str, int]:
        count = self.config.target_examples_for_split(split) // len(self.config.template_specs)
        return {_normalize_template(spec.label_template): count for spec in self.config.template_specs}

    def _record_from_row(self, row: Mapping[str, Any], *, split: str) -> RealVideoClipRecord:
        label_template = str(
            row.get(self.config.text_column)
            or row.get("text")
            or row.get("template")
            or row.get("label")
            or row.get("label_template")
            or ""
        )
        if not label_template:
            raise RealVideoDataError("Each source row must contain a text/label_template field.")
        norm_template = _normalize_template(label_template)
        spec = self.config.template_lookup.get(norm_template)
        if spec is None:
            raise RealVideoDataError(
                f"Row does not match one of the configured real-video templates: {row!r}"
            )

        placeholders = _as_tuple(row.get(self.config.placeholders_column) or row.get("placeholders"))
        video_id = str(
            row.get(self.config.video_id_column)
            or row.get("video_id")
            or row.get("id")
            or row.get("name")
            or ""
        )
        if not video_id:
            video_id = f"{split}_{abs(hash(json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False))) % (10**12)}"

        description = str(row.get("description") or _format_description(spec.label_template, placeholders))
        return RealVideoClipRecord(
            video_id=video_id,
            label_template=spec.label_template,
            pair_group=spec.pair_group,
            paired_template=spec.paired_template,
            placeholders=placeholders,
            description=description,
            split=split,
            raw_metadata=dict(row),
            source_path=str(
                row.get("video_path")
                or row.get("path")
                or row.get("file")
                or row.get("source_path")
                or ""
            ),
        )

    def _materialize_source_record(
        self,
        record: RealVideoClipRecord,
        row: Mapping[str, Any],
    ) -> RealVideoClipRecord:
        video_value = (
            row.get(self.config.video_column)
            or row.get("video")
            or row.get("path")
            or row.get("video_path")
        )
        if video_value is None and not record.source_path:
            raise RealVideoDataError(
                f"Row for video_id {record.video_id!r} did not provide a decodable video object or path."
            )
        clip = _decode_video_value(
            video_value,
            video_path=_ensure_path(record.source_path) if record.source_path else None,
        )
        clip = self._prepare_clip(clip)
        return RealVideoClipRecord(
            video_id=record.video_id,
            label_template=record.label_template,
            pair_group=record.pair_group,
            paired_template=record.paired_template,
            placeholders=record.placeholders,
            description=record.description,
            split=record.split,
            raw_metadata=dict(row),
            clip=clip,
            source_path=record.source_path,
        )

    def _collect_records_from_dataset(
        self,
        dataset: Any,
        *,
        split: str,
        label_feature: Any = None,
    ) -> List[RealVideoClipRecord]:
        template_targets = self._template_targets_for_split(split)
        target_total = sum(template_targets.values())
        collected: Dict[str, List[RealVideoClipRecord]] = {key: [] for key in template_targets}
        seen_ids: set[str] = set()
        skipped_rows = 0
        decode_failures = 0

        for row_index, row in enumerate(dataset):
            if row_index >= self.config.max_source_scan:
                break
            if label_feature is not None and hasattr(label_feature, "int2str"):
                label_value = row.get(self.config.text_column)
                if isinstance(label_value, (int, np.integer)):
                    row = dict(row)
                    row[self.config.text_column] = label_feature.int2str(int(label_value))
            try:
                record = self._record_from_row(row, split=split)
            except RealVideoDataError:
                skipped_rows += 1
                continue
            norm_template = _normalize_template(record.label_template)
            if norm_template not in collected:
                skipped_rows += 1
                continue
            if record.video_id in seen_ids:
                skipped_rows += 1
                continue
            if len(collected[norm_template]) >= template_targets[norm_template]:
                skipped_rows += 1
                continue

            try:
                record = self._materialize_source_record(record, row)
            except RealVideoDataError:
                decode_failures += 1
                continue

            collected[norm_template].append(record)
            seen_ids.add(record.video_id)

            if all(len(items) >= template_targets[key] for key, items in collected.items()):
                break

        records = [record for items in collected.values() for record in items]
        if len(records) < target_total:
            available = len(records)
            raise RealVideoDataError(
                f"Collected only {available} video records for split {split!r}, but the requested subset "
                f"requires {target_total}. Skipped {skipped_rows} non-matching/duplicate rows and "
                f"{decode_failures} decode failures. Try increasing max_source_scan or using a larger split."
            )
        return records

    def _load_records_from_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        split: str,
        manifest_dir: Path,
        source_split: Optional[str] = None,
    ) -> List[RealVideoClipRecord]:
        template_targets = self._template_targets_for_split(split)
        target_total = sum(template_targets.values())
        collected: Dict[str, List[RealVideoClipRecord]] = {key: [] for key in template_targets}
        effective_split = source_split or self.source_split_for_request(split)

        for row in rows:
            row_split = str(row.get("split") or effective_split)
            if row_split != effective_split:
                continue
            try:
                record = self._record_from_row(row, split=split)
            except RealVideoDataError:
                continue
            norm_template = _normalize_template(record.label_template)
            if norm_template not in collected:
                continue
            if len(collected[norm_template]) >= template_targets[norm_template]:
                continue

            clip = None
            cache_path = row.get("cache_path") or row.get("clip_path")
            source_path = row.get("source_path") or row.get("video_path") or row.get("path")
            if cache_path:
                cache_file = _resolve_path(manifest_dir, cache_path)
                if cache_file.exists():
                    clip = torch.load(cache_file, map_location="cpu")
            if clip is None and source_path:
                try:
                    clip = self._prepare_clip(self._decode_clip(_resolve_path(manifest_dir, source_path), row))
                except RealVideoDataError:
                    continue

            collected[norm_template].append(
                RealVideoClipRecord(
                    video_id=record.video_id,
                    label_template=record.label_template,
                    pair_group=record.pair_group,
                    paired_template=record.paired_template,
                    placeholders=record.placeholders,
                    description=record.description,
                    split=split,
                    raw_metadata=dict(row),
                    clip=clip,
                    cache_path=str(manifest_dir / str(cache_path)) if cache_path else None,
                    source_path=str(source_path) if source_path else None,
                )
            )
            if all(len(items) >= template_targets[key] for key, items in collected.items()):
                break

        records = [record for items in collected.values() for record in items]
        if len(records) < target_total:
            raise RealVideoDataError(
                f"Loaded only {len(records)} records from the manifest/cache rows, but {target_total} were required."
            )
        return records


class SomethingSomethingV2SubsetAdapter(RealVideoDatasetAdapter):
    """Subset adapter for Something-Something-style datasets with label text and optional placeholders."""

    def __init__(self, config: Optional[RealVideoSubsetConfig] = None) -> None:
        super().__init__(config or RealVideoSubsetConfig())

    def _load_source_records(self, split: str) -> List[RealVideoClipRecord]:
        if self.config.local_manifest_path:
            return self._load_from_local_manifest(split, Path(self.config.local_manifest_path))
        return self._load_from_hf_dataset(split)

    def _load_from_local_manifest(self, split: str, manifest_path: Path) -> List[RealVideoClipRecord]:
        if not manifest_path.exists():
            raise RealVideoDataError(
                f"Local manifest file does not exist: {manifest_path}. "
                "Provide a JSONL manifest with video_path, label_template, and placeholders fields."
            )
        rows = _read_jsonl(manifest_path)
        return self._load_records_from_rows(
            rows,
            split=split,
            manifest_dir=manifest_path.parent,
            source_split=self.source_split_for_request(split),
        )

    def _load_from_hf_dataset(self, split: str) -> List[RealVideoClipRecord]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RealVideoDependencyError(
                "The 'datasets' package is required to load the configured real-video dataset from Hugging Face. "
                "Install datasets, torchvision, av, and torchcodec for the notebook demo."
            ) from exc

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.source_split_for_request(split),
                streaming=self.config.streaming,
            )
        except Exception as exc:  # pragma: no cover - dataset/network dependent
            raise RealVideoDataError(
                f"Failed to load dataset {self.config.dataset_name!r} split {split!r}. "
                "Check internet access, the dataset name, and Hugging Face credentials if required."
            ) from exc

        label_feature = None
        try:
            features = getattr(dataset, "features", None)
            if features is not None:
                label_feature = features.get(self.config.text_column)
        except Exception:
            label_feature = None
        return self._collect_records_from_dataset(dataset, split=split, label_feature=label_feature)


class UCF101SubsetAdapter(SomethingSomethingV2SubsetAdapter):
    """Subset adapter for UCF101 hosted as a standard Hugging Face video dataset."""

    def __init__(self, config: Optional[RealVideoSubsetConfig] = None) -> None:
        base_config = config or RealVideoSubsetConfig()
        super().__init__(
            RealVideoSubsetConfig(
                dataset_name=base_config.dataset_name,
                cache_root=base_config.cache_root,
                sequence_length=base_config.sequence_length,
                observed_length=base_config.observed_length,
                future_length=base_config.future_length,
                image_size=base_config.image_size,
                train_examples_per_template=base_config.train_examples_per_template,
                eval_examples_per_template=base_config.eval_examples_per_template,
                confident_eval_examples_per_template=base_config.confident_eval_examples_per_template,
                max_source_scan=base_config.max_source_scan,
                seed=base_config.seed,
                split_column=base_config.split_column,
                text_column=base_config.text_column or "label",
                video_id_column=base_config.video_id_column or "video_id",
                placeholders_column=base_config.placeholders_column,
                video_column=base_config.video_column or "video",
                template_specs=base_config.template_specs or DEFAULT_UCF101_TEMPLATE_SPECS,
                streaming=base_config.streaming,
                local_manifest_path=base_config.local_manifest_path,
            )
        )

    def _load_source_records(self, split: str) -> List[RealVideoClipRecord]:
        if self.config.local_manifest_path:
            return self._load_from_local_manifest(split, Path(self.config.local_manifest_path))
        return self._load_from_hf_dataset(split)

    def _load_from_local_manifest(self, split: str, manifest_path: Path) -> List[RealVideoClipRecord]:
        if not manifest_path.exists():
            raise RealVideoDataError(
                f"Local manifest file does not exist: {manifest_path}. "
                "Provide a JSONL manifest with video_path, label, and optional description fields."
            )
        rows = _read_jsonl(manifest_path)
        return self._load_records_from_rows(
            rows,
            split=split,
            manifest_dir=manifest_path.parent,
            source_split=self.source_split_for_request(split),
        )

    def _load_from_hf_dataset(self, split: str) -> List[RealVideoClipRecord]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise RealVideoDependencyError(
                "The 'datasets' package is required to load the default UCF101 fallback subset from Hugging Face."
            ) from exc

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.source_split_for_request(split),
                streaming=self.config.streaming,
            )
        except Exception as exc:  # pragma: no cover
            raise RealVideoDataError(
                f"Failed to load dataset {self.config.dataset_name!r} split {split!r}. "
                "Check internet access, the dataset name, and Hugging Face availability."
            ) from exc

        label_feature = None
        try:
            features = getattr(dataset, "features", None)
            if features is not None:
                label_feature = features.get(self.config.text_column)
        except Exception:
            label_feature = None
        return self._collect_records_from_dataset(dataset, split=split, label_feature=label_feature)


class LocalVideoManifestAdapter(RealVideoDatasetAdapter):
    """Generic clip-based adapter for a local JSONL video manifest."""

    def __init__(self, config: RealVideoSubsetConfig) -> None:
        super().__init__(config)

    def _load_source_records(self, split: str) -> List[RealVideoClipRecord]:
        if not self.config.local_manifest_path:
            raise RealVideoDataError(
                "LocalVideoManifestAdapter requires config.local_manifest_path to point to a JSONL manifest."
            )
        manifest_path = Path(self.config.local_manifest_path)
        if not manifest_path.exists():
            raise RealVideoDataError(f"Local manifest file does not exist: {manifest_path}")
        rows = _read_jsonl(manifest_path)
        return self._load_records_from_rows(rows, split=split, manifest_dir=manifest_path.parent)

    def _load_records_from_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        split: str,
        manifest_dir: Path,
    ) -> List[RealVideoClipRecord]:
        template_targets = self._template_targets_for_split(split)
        target_total = sum(template_targets.values())
        collected: Dict[str, List[RealVideoClipRecord]] = {key: [] for key in template_targets}
        effective_split = self.source_split_for_request(split)

        for row in rows:
            row_split = self.source_split_for_request(str(row.get("split") or effective_split))
            if row_split != effective_split:
                continue
            try:
                record = self._record_from_row(row, split=split)
            except RealVideoDataError:
                continue
            norm_template = _normalize_template(record.label_template)
            if norm_template not in collected:
                continue
            if len(collected[norm_template]) >= template_targets[norm_template]:
                continue

            clip = None
            cache_path = row.get("cache_path") or row.get("clip_path")
            source_path = row.get("video_path") or row.get("path") or row.get("source_path")
            if cache_path:
                cache_file = _resolve_path(manifest_dir, cache_path)
                if cache_file.exists():
                    clip = torch.load(cache_file, map_location="cpu")
            if clip is None and source_path:
                try:
                    clip = self._prepare_clip(self._decode_clip(_resolve_path(manifest_dir, source_path), row))
                except RealVideoDataError:
                    continue

            collected[norm_template].append(
                RealVideoClipRecord(
                    video_id=record.video_id,
                    label_template=record.label_template,
                    pair_group=record.pair_group,
                    paired_template=record.paired_template,
                    placeholders=record.placeholders,
                    description=record.description,
                    split=split,
                    raw_metadata=dict(row),
                    clip=clip,
                    cache_path=str(manifest_dir / str(cache_path)) if cache_path else None,
                    source_path=str(source_path) if source_path else None,
                )
            )
            if all(len(items) >= template_targets[key] for key, items in collected.items()):
                break

        records = [record for items in collected.values() for record in items]
        if len(records) < target_total:
            raise RealVideoDataError(
                f"Loaded only {len(records)} records from the local manifest, but {target_total} were required."
            )
        return records

    def _template_targets_for_split(self, split: str) -> Dict[str, int]:
        count = self.config.target_examples_for_split(split) // len(self.config.template_specs)
        return {_normalize_template(spec.label_template): count for spec in self.config.template_specs}

    def _record_from_row(self, row: Mapping[str, Any], *, split: str) -> RealVideoClipRecord:
        label_template = str(
            row.get(self.config.text_column)
            or row.get("text")
            or row.get("template")
            or row.get("label")
            or row.get("label_template")
            or ""
        )
        if not label_template:
            raise RealVideoDataError("Each source row must contain a text/label_template field.")
        norm_template = _normalize_template(label_template)
        spec = self.config.template_lookup.get(norm_template)
        if spec is None:
            raise RealVideoDataError(
                f"Row does not match one of the configured local-manifest templates: {row!r}"
            )

        placeholders = _as_tuple(row.get(self.config.placeholders_column) or row.get("placeholders"))
        video_id = str(
            row.get(self.config.video_id_column)
            or row.get("video_id")
            or row.get("id")
            or row.get("name")
            or ""
        )
        if not video_id:
            video_id = (
                f"{split}_"
                f"{abs(hash(json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False))) % (10**12)}"
            )

        description = str(row.get("description") or _format_description(spec.label_template, placeholders))
        return RealVideoClipRecord(
            video_id=video_id,
            label_template=spec.label_template,
            pair_group=spec.pair_group,
            paired_template=spec.paired_template,
            placeholders=placeholders,
            description=description,
            split=split,
            raw_metadata=dict(row),
            source_path=str(
                row.get("video_path")
                or row.get("path")
                or row.get("file")
                or row.get("source_path")
                or ""
            ),
        )


def save_real_video_manifest(records: Sequence[RealVideoClipRecord], path: str | Path) -> Path:
    """Save a simple JSONL manifest for future reuse."""

    path = Path(path)
    _write_jsonl(path, [record.as_dict() for record in records])
    return path


def make_ucf101_fallback_config(base_config: Optional[RealVideoSubsetConfig] = None) -> RealVideoSubsetConfig:
    config = (base_config or RealVideoSubsetConfig()).validate()
    return RealVideoSubsetConfig(
        dataset_name="MichiganNLP/ucf-101",
        cache_root=config.cache_root,
        sequence_length=config.sequence_length,
        observed_length=config.observed_length,
        future_length=config.future_length,
        image_size=config.image_size,
        train_examples_per_template=config.train_examples_per_template,
        eval_examples_per_template=config.eval_examples_per_template,
        confident_eval_examples_per_template=config.confident_eval_examples_per_template,
        max_source_scan=config.max_source_scan,
        seed=config.seed,
        split_column=config.split_column,
        text_column="label",
        video_id_column=config.video_id_column,
        placeholders_column=config.placeholders_column,
        video_column=config.video_column,
        template_specs=DEFAULT_UCF101_TEMPLATE_SPECS,
        streaming=config.streaming,
        local_manifest_path=config.local_manifest_path,
    )


def available_something_something_templates() -> Tuple[str, ...]:
    return tuple(spec.label_template for spec in DEFAULT_SOMETHING_SOMETHING_TEMPLATE_SPECS)


def available_ucf101_templates() -> Tuple[str, ...]:
    return tuple(spec.label_template for spec in DEFAULT_UCF101_TEMPLATE_SPECS)


def available_real_video_templates(config: Optional[RealVideoSubsetConfig] = None) -> Tuple[str, ...]:
    active_config = config or RealVideoSubsetConfig()
    return tuple(spec.label_template for spec in active_config.template_specs)
