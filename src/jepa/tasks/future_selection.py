"""Future-selection task layer built on top of Moving MNIST.

Shape convention:
- Full Moving MNIST sequence: ``(T, C, H, W)``
- Observed clip: ``(T_obs, C, H, W)``
- Candidate futures: ``(K, T_future, C, H, W)``

This module turns a 16-frame Moving MNIST sequence into a supervised future-selection
example with one true continuation and three plausible negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from jepa.data.moving_mnist import MovingMNISTConfig, MovingMNISTDataset

__all__ = [
    "FutureSelectionCandidate",
    "FutureSelectionConfig",
    "FutureSelectionDataset",
    "FutureSelectionExample",
    "build_future_selection_dataset",
    "build_future_selection_loader",
    "generate_future_selection_example",
    "save_future_selection_example_artifacts",
    "save_future_selection_examples",
    "summarize_future_selection_example",
]


CandidateStrategy = Literal[
    "true_continuation",
    "shuffled_temporal_order",
    "wrong_velocity_continuation",
    "wrong_direction_continuation",
    "future_segment_from_other_sample",
    "mirrored_or_perturbed_continuation",
]


DEFAULT_NEGATIVE_STRATEGIES: tuple[CandidateStrategy, ...] = (
    "shuffled_temporal_order",
    "wrong_velocity_continuation",
    "wrong_direction_continuation",
    "future_segment_from_other_sample",
    "mirrored_or_perturbed_continuation",
)


@dataclass(frozen=True)
class FutureSelectionConfig:
    """Configuration for the future-selection task."""

    sequence_length: int = 16
    observed_length: int = 8
    future_length: int = 8
    num_candidates: int = 4
    candidate_strategies: tuple[CandidateStrategy, ...] = DEFAULT_NEGATIVE_STRATEGIES
    seed: int = 123
    max_example_visualizations: int = 3

    def validate(self) -> "FutureSelectionConfig":
        if self.sequence_length != self.observed_length + self.future_length:
            raise ValueError("sequence_length must equal observed_length + future_length.")
        if self.observed_length < 1 or self.future_length < 1:
            raise ValueError("observed_length and future_length must both be positive.")
        if self.num_candidates < 2:
            raise ValueError("num_candidates must be at least 2.")
        if len(self.candidate_strategies) == 0:
            raise ValueError("candidate_strategies must not be empty.")
        if self.num_candidates - 1 > len(set(self.candidate_strategies)):
            raise ValueError(
                "Not enough unique negative strategies for the requested number of candidates."
            )
        return self

    @property
    def observed_shape_convention(self) -> str:
        return "(T_obs, C, H, W)"

    @property
    def candidates_shape_convention(self) -> str:
        return "(K, T_future, C, H, W)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "observed_length": self.observed_length,
            "future_length": self.future_length,
            "num_candidates": self.num_candidates,
            "candidate_strategies": list(self.candidate_strategies),
            "seed": self.seed,
            "max_example_visualizations": self.max_example_visualizations,
            "observed_shape_convention": self.observed_shape_convention,
            "candidates_shape_convention": self.candidates_shape_convention,
        }


@dataclass
class FutureSelectionCandidate:
    """One candidate future and its generation metadata."""

    future: Tensor
    strategy: CandidateStrategy
    is_true: bool
    source_index: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "future": self.future,
            "strategy": self.strategy,
            "is_true": self.is_true,
            "source_index": self.source_index,
            "details": dict(self.details),
        }


@dataclass
class FutureSelectionExample:
    """A single future-selection training example."""

    observed: Tensor
    candidates: Tensor
    correct_index: int
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observed": self.observed,
            "candidates": self.candidates,
            "correct_index": self.correct_index,
            "metadata": dict(self.metadata),
        }


def _rng_from_seed(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(int(seed + index))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_frame(frame: Tensor) -> Tensor:
    return frame.clamp(0.0, 1.0)


def _center_of_mass(frame: Tensor) -> Tensor:
    frame = frame.squeeze(0) if frame.ndim == 3 else frame
    weights = frame.clamp_min(0.0).to(dtype=torch.float32)
    total = float(weights.sum().item())
    height, width = weights.shape[-2:]
    if total <= 1e-6:
        return torch.tensor([height / 2.0, width / 2.0], dtype=torch.float32)

    ys = torch.arange(height, dtype=torch.float32, device=weights.device)
    xs = torch.arange(width, dtype=torch.float32, device=weights.device)
    y_mass = (weights * ys.view(-1, 1)).sum() / total
    x_mass = (weights * xs.view(1, -1)).sum() / total
    return torch.stack([y_mass, x_mass])


def _estimate_velocity(observed: Tensor) -> Tensor:
    if observed.shape[0] < 2:
        return torch.zeros(2, dtype=torch.float32)
    prev_com = _center_of_mass(observed[-2])
    last_com = _center_of_mass(observed[-1])
    velocity = last_com - prev_com
    if torch.allclose(velocity, torch.zeros_like(velocity)):
        velocity = torch.tensor([1.0, 0.0], dtype=torch.float32)
    return velocity


def _translate_frame(frame: Tensor, shift_y: float, shift_x: float) -> Tensor:
    if frame.ndim != 3:
        raise ValueError("Expected frame shape (C, H, W).")
    c, h, w = frame.shape
    device = frame.device
    dtype = frame.dtype

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    if w > 1:
        grid[..., 0] -= (2.0 * shift_x) / float(w - 1)
    if h > 1:
        grid[..., 1] -= (2.0 * shift_y) / float(h - 1)

    sampled = F.grid_sample(
        frame.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).clamp(0.0, 1.0)


def _build_velocity_continuation(
    observed: Tensor,
    velocity_scale: float,
    direction_scale: float,
    future_length: int,
    *,
    add_noise: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    base_frame = observed[-1]
    velocity = _estimate_velocity(observed)
    velocity = velocity * float(velocity_scale)
    velocity = velocity * float(direction_scale)

    future_frames: List[Tensor] = []
    current = base_frame
    for step in range(future_length):
        step_shift = velocity * float(step + 1)
        next_frame = _translate_frame(current, float(step_shift[0].item()), float(step_shift[1].item()))
        if add_noise and rng is not None:
            noise = torch.from_numpy(
                rng.normal(loc=0.0, scale=0.03, size=tuple(next_frame.shape)).astype(np.float32)
            ).to(next_frame.device)
            next_frame = (next_frame + noise).clamp(0.0, 1.0)
        future_frames.append(next_frame)
    return torch.stack(future_frames, dim=0)


def _mirror_or_perturb_future(future: Tensor, rng: np.random.Generator) -> Tensor:
    mirrored = future.flip(-1)
    shift_y = int(rng.integers(-2, 3))
    shift_x = int(rng.integers(-2, 3))
    translated = torch.stack(
        [
            _translate_frame(frame, float(shift_y), float(shift_x))
            for frame in mirrored
        ],
        dim=0,
    )
    noise = torch.from_numpy(rng.normal(0.0, 0.02, size=tuple(translated.shape)).astype(np.float32))
    return (translated + noise).clamp(0.0, 1.0)


def _shuffle_temporal_order(future: Tensor, rng: np.random.Generator) -> Tensor:
    order = np.arange(future.shape[0])
    rng.shuffle(order)
    return future[torch.as_tensor(order, dtype=torch.long)]


def _select_other_sample_future(
    dataset: Dataset[Tensor],
    current_index: int,
    rng: np.random.Generator,
    future_length: int,
    observed_length: int,
) -> tuple[Tensor, int]:
    if len(dataset) <= 1:
        raise ValueError("Need at least two samples to draw a future from another sample.")
    candidate_index = int(rng.integers(0, len(dataset)))
    if candidate_index == current_index:
        candidate_index = (candidate_index + 1) % len(dataset)
    other_sequence = dataset[candidate_index]
    if other_sequence.shape[0] < observed_length + future_length:
        raise ValueError("Other sample is too short for the requested split.")
    return other_sequence[observed_length : observed_length + future_length], candidate_index


def _build_candidates_for_sequence(
    observed: Tensor,
    true_future: Tensor,
    *,
    dataset: Optional[Dataset[Tensor]],
    index: int,
    config: FutureSelectionConfig,
) -> tuple[Tensor, List[Dict[str, Any]], int]:
    rng = _rng_from_seed(config.seed, index)
    candidates: List[Tensor] = []
    metadata: List[Dict[str, Any]] = []

    candidates.append(true_future.clone())
    metadata.append(
        {
            "strategy": "true_continuation",
            "generation_type": "true_continuation",
            "is_true": True,
            "source_index": index,
            "details": {"description": "Ground-truth future segment."},
        }
    )

    strategies = list(config.candidate_strategies)
    rng.shuffle(strategies)
    chosen = strategies[: config.num_candidates - 1]

    for strategy in chosen:
        if strategy == "shuffled_temporal_order":
            candidate = _shuffle_temporal_order(true_future, rng)
            details = {"description": "True future frames in shuffled order."}
            source_index = index
        elif strategy == "wrong_velocity_continuation":
            candidate = _build_velocity_continuation(
                observed,
                velocity_scale=1.6,
                direction_scale=1.0,
                future_length=config.future_length,
                add_noise=True,
                rng=rng,
            )
            details = {
                "description": "Continuation with the right direction but an incorrect speed.",
                "velocity_scale": 1.6,
            }
            source_index = index
        elif strategy == "wrong_direction_continuation":
            candidate = _build_velocity_continuation(
                observed,
                velocity_scale=1.0,
                direction_scale=-1.0,
                future_length=config.future_length,
                add_noise=True,
                rng=rng,
            )
            details = {
                "description": "Continuation moving in the opposite direction.",
                "direction_scale": -1.0,
            }
            source_index = index
        elif strategy == "future_segment_from_other_sample":
            if dataset is None:
                raise ValueError("dataset is required for future_segment_from_other_sample.")
            candidate, source_index = _select_other_sample_future(
                dataset=dataset,
                current_index=index,
                rng=rng,
                future_length=config.future_length,
                observed_length=config.observed_length,
            )
            details = {"description": "Future segment taken from another sample."}
        elif strategy == "mirrored_or_perturbed_continuation":
            candidate = _mirror_or_perturb_future(true_future, rng)
            details = {"description": "Mirrored future with small spatial perturbations."}
            source_index = index
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        candidates.append(candidate)
        metadata.append(
            {
                "strategy": strategy,
                "generation_type": strategy,
                "is_true": False,
                "source_index": source_index,
                "details": details,
            }
        )

    order = np.arange(len(candidates))
    rng.shuffle(order)
    stacked_candidates = torch.stack([candidates[int(i)] for i in order], dim=0)
    metadata = [metadata[int(i)] for i in order]
    correct_index = next(i for i, item in enumerate(metadata) if item["is_true"])
    return stacked_candidates, metadata, correct_index


def generate_future_selection_example(
    sequence: Tensor,
    *,
    dataset: Optional[Dataset[Tensor]] = None,
    index: int = 0,
    config: Optional[FutureSelectionConfig] = None,
) -> FutureSelectionExample:
    """Build a future-selection example from a full Moving MNIST sequence."""

    config = (config or FutureSelectionConfig()).validate()
    if sequence.shape[0] != config.sequence_length:
        raise ValueError(
            f"Expected sequence length {config.sequence_length}, got {sequence.shape[0]}."
        )
    if sequence.ndim != 4:
        raise ValueError("Expected sequence shape (T, C, H, W).")

    observed = sequence[: config.observed_length].clone()
    true_future = sequence[config.observed_length : config.sequence_length].clone()

    candidates, metadata, correct_index = _build_candidates_for_sequence(
        observed,
        true_future,
        dataset=dataset,
        index=index,
        config=config,
    )

    example = FutureSelectionExample(
        observed=observed,
        candidates=candidates,
        correct_index=correct_index,
        metadata={
            "index": index,
            "sequence_length": config.sequence_length,
            "observed_length": config.observed_length,
            "future_length": config.future_length,
            "candidate_count": int(candidates.shape[0]),
            "candidate_strategies": metadata,
            "observed_shape_convention": config.observed_shape_convention,
            "candidates_shape_convention": config.candidates_shape_convention,
        },
    )
    return example


class FutureSelectionDataset(Dataset[FutureSelectionExample]):
    """Wrap a Moving MNIST dataset and expose future-selection examples."""

    def __init__(
        self,
        base_dataset: Dataset[Tensor] | MovingMNISTDataset,
        *,
        config: Optional[FutureSelectionConfig] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.config = (config or FutureSelectionConfig()).validate()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> FutureSelectionExample:
        sequence = self.base_dataset[index]
        return generate_future_selection_example(
            sequence,
            dataset=self.base_dataset,
            index=index,
            config=self.config,
        )


def _example_collate(batch: Sequence[FutureSelectionExample]) -> Dict[str, Any]:
    observed = torch.stack([item.observed for item in batch], dim=0)
    candidates = torch.stack([item.candidates for item in batch], dim=0)
    correct_index = torch.tensor([item.correct_index for item in batch], dtype=torch.long)
    metadata = [item.metadata for item in batch]
    return {
        "observed": observed,
        "candidates": candidates,
        "correct_index": correct_index,
        "metadata": metadata,
    }


def build_future_selection_loader(
    base_dataset: Dataset[Tensor] | MovingMNISTDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    config: Optional[FutureSelectionConfig] = None,
) -> DataLoader[Dict[str, Any]]:
    """Create a DataLoader for the future-selection task."""

    task_dataset = FutureSelectionDataset(base_dataset, config=config)
    return DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_example_collate,
    )


def summarize_future_selection_example(example: FutureSelectionExample) -> Dict[str, Any]:
    """Return a compact summary for notebook checks and logs."""

    return {
        "observed_shape": list(example.observed.shape),
        "candidates_shape": list(example.candidates.shape),
        "correct_index": int(example.correct_index),
        "candidate_strategies": [item["strategy"] for item in example.metadata["candidate_strategies"]],
        "observed_shape_convention": example.metadata["observed_shape_convention"],
        "candidates_shape_convention": example.metadata["candidates_shape_convention"],
    }


def _frame_to_pil(frame: Tensor) -> Image.Image:
    frame = _normalize_frame(frame.detach().cpu())
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (C, H, W); got {tuple(frame.shape)}")
    channels = int(frame.shape[0])
    if channels == 1:
        frame = frame.repeat(3, 1, 1)
    elif channels != 3:
        raise ValueError(f"Expected 1 or 3 channels; got {channels}")
    array = frame.mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _sequence_to_pil_frames(sequence: Tensor) -> List[Image.Image]:
    if sequence.ndim != 4:
        raise ValueError(f"Expected sequence shape (T, C, H, W); got {tuple(sequence.shape)}")
    if sequence.shape[0] < 1:
        raise ValueError("Cannot create frames from an empty sequence.")
    return [_frame_to_pil(frame) for frame in sequence]


def _sequence_to_grid(sequence: Tensor, *, columns: int = 4, padding: int = 2) -> Image.Image:
    frames = _sequence_to_pil_frames(sequence)
    if not frames:
        raise ValueError("Cannot create a grid from an empty sequence.")
    width, height = frames[0].size
    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    grid = Image.new(
        "RGB",
        (
            columns * width + padding * (columns - 1),
            rows * height + padding * (rows - 1),
        ),
        color=(0, 0, 0),
    )
    for frame_index, frame in enumerate(frames):
        row = frame_index // columns
        column = frame_index % columns
        x = column * (width + padding)
        y = row * (height + padding)
        grid.paste(frame, (x, y))
    return grid


def _draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 18), color=(0, 0, 0))
    canvas.paste(image, (0, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(255, 255, 255))
    return canvas


def _candidate_label(example: FutureSelectionExample, candidate_index: int) -> str:
    strategy = example.metadata["candidate_strategies"][candidate_index]["strategy"]
    return f"candidate {candidate_index} | {strategy}"


def _save_gif_frames(
    frames: Sequence[Image.Image],
    output_path: str | Path,
    *,
    duration_ms: int = 160,
) -> Path:
    if not frames:
        raise ValueError("Cannot save an empty GIF.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


def save_sequence_gif(
    sequence: Tensor,
    output_path: str | Path,
    *,
    duration_ms: int = 160,
    label: Optional[str] = None,
) -> Path:
    frames = _sequence_to_pil_frames(sequence)
    if label is not None:
        frames = [_draw_label(frame, f"{label} | t={frame_index:02d}") for frame_index, frame in enumerate(frames)]
    return _save_gif_frames(frames, output_path, duration_ms=duration_ms)


def _make_example_panel(example: FutureSelectionExample) -> Image.Image:
    panels = [_draw_label(_sequence_to_grid(example.observed), "observed (0:8)")]
    for candidate_index, candidate in enumerate(example.candidates):
        label = _candidate_label(example, candidate_index)
        panels.append(_draw_label(_sequence_to_grid(candidate), label))

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels) + 2 * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + 2
    return canvas


def _make_example_animation_frames(
    example: FutureSelectionExample,
    *,
    panel_padding: int = 4,
) -> List[Image.Image]:
    labels = ["observed (0:8)"] + [
        _candidate_label(example, candidate_index)
        for candidate_index in range(example.candidates.shape[0])
    ]
    sequences = [example.observed, *[candidate for candidate in example.candidates]]
    sequence_frames = [_sequence_to_pil_frames(sequence) for sequence in sequences]

    frame_width = max(frame.width for frames in sequence_frames for frame in frames)
    frame_height = max(frame.height for frames in sequence_frames for frame in frames)
    max_steps = max(len(frames) for frames in sequence_frames)

    animation_frames: List[Image.Image] = []
    for frame_index in range(max_steps):
        labeled_panels: List[Image.Image] = []
        for label, frames in zip(labels, sequence_frames):
            if frame_index < len(frames):
                frame = frames[frame_index]
            else:
                frame = Image.new("RGB", (frame_width, frame_height), color=(0, 0, 0))
            if frame.size != (frame_width, frame_height):
                padded = Image.new("RGB", (frame_width, frame_height), color=(0, 0, 0))
                padded.paste(frame, (0, 0))
                frame = padded
            labeled_panels.append(_draw_label(frame, f"{label} | t={frame_index:02d}"))

        width = max(panel.width for panel in labeled_panels)
        height = sum(panel.height for panel in labeled_panels) + panel_padding * (len(labeled_panels) - 1)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        y = 0
        for panel in labeled_panels:
            canvas.paste(panel, (0, y))
            y += panel.height + panel_padding
        animation_frames.append(canvas)
    return animation_frames


def save_future_selection_example_artifacts(
    example: FutureSelectionExample,
    output_dir: str | Path,
    *,
    stem: str = "future_selection_example",
    duration_ms: int = 160,
    save_panel_png: bool = True,
    save_candidate_pngs: bool = True,
    save_individual_gifs: bool = True,
    save_comparison_gif: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Any] = {
        "panel_png": None,
        "comparison_gif": None,
        "observed_gif": None,
        "candidate_pngs": [],
        "candidate_gifs": [],
    }

    if save_panel_png:
        panel_path = output_dir / f"{stem}_panel.png"
        _make_example_panel(example).save(panel_path)
        artifacts["panel_png"] = panel_path

    if save_individual_gifs:
        observed_path = output_dir / f"{stem}_observed.gif"
        artifacts["observed_gif"] = save_sequence_gif(
            example.observed,
            observed_path,
            duration_ms=duration_ms,
            label="observed (0:8)",
        )

    for candidate_index, candidate in enumerate(example.candidates):
        label = _candidate_label(example, candidate_index)
        if save_candidate_pngs:
            candidate_grid = _draw_label(_sequence_to_grid(candidate), label)
            candidate_path = output_dir / f"{stem}_candidate_{candidate_index}.png"
            candidate_grid.save(candidate_path)
            artifacts["candidate_pngs"].append(candidate_path)
        if save_individual_gifs:
            candidate_gif_path = output_dir / f"{stem}_candidate_{candidate_index}.gif"
            artifacts["candidate_gifs"].append(
                save_sequence_gif(
                    candidate,
                    candidate_gif_path,
                    duration_ms=duration_ms,
                    label=label,
                )
            )

    if save_comparison_gif:
        comparison_path = output_dir / f"{stem}_comparison.gif"
        comparison_frames = _make_example_animation_frames(example)
        artifacts["comparison_gif"] = _save_gif_frames(
            comparison_frames,
            comparison_path,
            duration_ms=duration_ms,
        )

    return artifacts


def _flatten_artifact_paths(artifacts: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for value in artifacts.values():
        if value is None:
            continue
        if isinstance(value, Path):
            paths.append(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            paths.extend(path for path in value if isinstance(path, Path))
    return paths


def save_future_selection_examples(
    examples: Sequence[FutureSelectionExample],
    output_dir: str | Path | None = None,
    *,
    max_examples: int = 3,
) -> List[Path]:
    """Save example visualizations under results/task_examples/."""

    output_dir = Path(output_dir) if output_dir is not None else _repo_root() / "results" / "task_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for example_index, example in enumerate(examples[:max_examples]):
        artifacts = save_future_selection_example_artifacts(
            example,
            output_dir,
            stem=f"future_selection_{example_index:03d}",
            save_panel_png=True,
            save_candidate_pngs=True,
            save_individual_gifs=True,
            save_comparison_gif=True,
        )
        saved_paths.extend(_flatten_artifact_paths(artifacts))

    return saved_paths


def build_future_selection_dataset(
    moving_mnist_dataset: MovingMNISTDataset,
    *,
    config: Optional[FutureSelectionConfig] = None,
) -> FutureSelectionDataset:
    """Convenience helper for wrapping an existing Moving MNIST dataset."""

    return FutureSelectionDataset(moving_mnist_dataset, config=config)
