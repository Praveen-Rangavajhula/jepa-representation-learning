"""Procedural Moving MNIST generation for notebook-first video experiments.

Shape convention:
- Single sample: ``(T, C, H, W)``
- DataLoader batch: ``(B, T, C, H, W)``

The dataset uses torchvision's MNIST digits as static sprites and synthesizes motion
procedurally by sampling initial positions and velocity vectors per sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


Split = Literal["train", "test"]


@dataclass
class MovingMNISTConfig:
    """Configuration for procedurally generated Moving MNIST sequences."""

    sequence_length: int = 20
    image_size: int = 64
    num_digits: int = 2
    velocity_range: tuple[float, float] = (1.5, 4.0)
    train_size: int = 10_000
    test_size: int = 2_000
    mnist_root: str = "data"
    seed: int = 17

    def validate(self) -> "MovingMNISTConfig":
        low, high = self.velocity_range
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        if self.image_size < 28:
            raise ValueError("image_size must be at least 28 so MNIST digits fit.")
        if self.num_digits < 1:
            raise ValueError("num_digits must be at least 1.")
        if self.train_size < 1 or self.test_size < 1:
            raise ValueError("train_size and test_size must both be positive.")
        if low < 0 or high <= 0 or high < low:
            raise ValueError("velocity_range must satisfy 0 <= low <= high and high > 0.")
        return self

    @property
    def shape_convention(self) -> str:
        return "(T, C, H, W)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "image_size": self.image_size,
            "num_digits": self.num_digits,
            "velocity_range": list(self.velocity_range),
            "train_size": self.train_size,
            "test_size": self.test_size,
            "mnist_root": self.mnist_root,
            "seed": self.seed,
            "shape_convention": self.shape_convention,
        }


def _reflect_coordinate(position: float, velocity: float, max_position: float) -> tuple[float, float]:
    if max_position <= 0:
        return 0.0, 0.0

    while position < 0 or position > max_position:
        if position < 0:
            position = -position
            velocity = -velocity
        elif position > max_position:
            position = (2.0 * max_position) - position
            velocity = -velocity
    return position, velocity


def _sample_velocity_vector(
    generator: np.random.Generator,
    velocity_range: tuple[float, float],
) -> np.ndarray:
    speed = float(generator.uniform(*velocity_range))
    angle = float(generator.uniform(0.0, 2.0 * math.pi))
    velocity = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32) * speed
    if np.allclose(velocity, 0.0):
        velocity[0] = max(velocity_range[0], 1.0)
    return velocity


def _prepare_digit_bank(mnist: MNIST) -> Tensor:
    return mnist.data.to(dtype=torch.float32).div(255.0)


class MovingMNISTDataset(Dataset[Tensor]):
    """Generate Moving MNIST sequences on the fly from torchvision MNIST digits."""

    def __init__(
        self,
        config: MovingMNISTConfig,
        split: Split = "train",
        *,
        download: bool = True,
    ) -> None:
        self.config = config.validate()
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'.")

        self.split = split
        self.length = self.config.train_size if split == "train" else self.config.test_size
        self.seed_offset = 0 if split == "train" else 1_000_000

        mnist_root = Path(self.config.mnist_root)
        self.mnist = MNIST(root=str(mnist_root), train=(split == "train"), download=download)
        self.digit_bank = _prepare_digit_bank(self.mnist)
        self.targets = self.mnist.targets.clone()
        self.digit_size = int(self.digit_bank.shape[-1])
        self.max_position = float(self.config.image_size - self.digit_size)

    def __len__(self) -> int:
        return self.length

    def _generator_for_index(self, index: int) -> np.random.Generator:
        seed = int(self.config.seed + self.seed_offset + index)
        return np.random.default_rng(seed)

    def _sample_digit_indices(self, generator: np.random.Generator) -> np.ndarray:
        return generator.integers(
            low=0,
            high=len(self.digit_bank),
            size=self.config.num_digits,
            endpoint=False,
        )

    def _sample_positions(self, generator: np.random.Generator) -> np.ndarray:
        if self.max_position <= 0:
            return np.zeros((self.config.num_digits, 2), dtype=np.float32)
        return generator.uniform(
            low=0.0,
            high=self.max_position,
            size=(self.config.num_digits, 2),
        ).astype(np.float32)

    def _sample_velocities(self, generator: np.random.Generator) -> np.ndarray:
        velocities = [
            _sample_velocity_vector(generator, self.config.velocity_range)
            for _ in range(self.config.num_digits)
        ]
        return np.stack(velocities, axis=0).astype(np.float32)

    def _render_sequence(
        self,
        digits: Tensor,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Tensor:
        sequence = torch.zeros(
            self.config.sequence_length,
            1,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )

        current_positions = positions.copy()
        current_velocities = velocities.copy()

        for frame_index in range(self.config.sequence_length):
            frame = sequence[frame_index, 0]
            for digit_index in range(self.config.num_digits):
                top = int(round(float(current_positions[digit_index, 0])))
                left = int(round(float(current_positions[digit_index, 1])))
                top = int(np.clip(top, 0, self.config.image_size - self.digit_size))
                left = int(np.clip(left, 0, self.config.image_size - self.digit_size))

                digit = digits[digit_index]
                existing = frame[top : top + self.digit_size, left : left + self.digit_size]
                frame[top : top + self.digit_size, left : left + self.digit_size] = torch.maximum(
                    existing,
                    digit,
                )

            current_positions = current_positions + current_velocities
            for digit_index in range(self.config.num_digits):
                current_positions[digit_index, 0], current_velocities[digit_index, 0] = _reflect_coordinate(
                    float(current_positions[digit_index, 0]),
                    float(current_velocities[digit_index, 0]),
                    self.max_position,
                )
                current_positions[digit_index, 1], current_velocities[digit_index, 1] = _reflect_coordinate(
                    float(current_positions[digit_index, 1]),
                    float(current_velocities[digit_index, 1]),
                    self.max_position,
                )

        return sequence.clamp_(0.0, 1.0)

    def __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= self.length:
            raise IndexError(index)

        generator = self._generator_for_index(index)
        digit_indices = self._sample_digit_indices(generator)
        digits = self.digit_bank[digit_indices]
        positions = self._sample_positions(generator)
        velocities = self._sample_velocities(generator)
        return self._render_sequence(digits=digits, positions=positions, velocities=velocities)


def create_moving_mnist_dataloaders(
    config: MovingMNISTConfig,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, Any]:
    """Create train/test datasets and DataLoaders for Moving MNIST."""

    train_dataset = MovingMNISTDataset(config=config, split="train")
    test_dataset = MovingMNISTDataset(config=config, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


def summarize_batch(batch: Tensor) -> Dict[str, Any]:
    """Return a compact summary of a batch tensor for notebook verification."""

    if not isinstance(batch, torch.Tensor):
        raise TypeError("summarize_batch expects a tensor batch.")
    return {
        "shape": list(batch.shape),
        "dtype": str(batch.dtype),
        "min": float(batch.min().item()),
        "max": float(batch.max().item()),
        "mean": float(batch.mean().item()),
        "shape_convention": "(B, T, C, H, W)",
    }


def _sequence_to_pil_frames(sequence: Tensor) -> List[Image.Image]:
    frames: List[Image.Image] = []
    sequence = sequence.detach().cpu()
    for frame in sequence:
        frame_2d = frame.squeeze(0).clamp(0.0, 1.0).mul(255).to(torch.uint8).numpy()
        frames.append(Image.fromarray(frame_2d, mode="L"))
    return frames


def save_sequence_gif(sequence: Tensor, output_path: str | Path, *, duration_ms: int = 120) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _sequence_to_pil_frames(sequence)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


def save_sequence_grid(
    sequence: Tensor,
    output_path: str | Path,
    *,
    max_frames: int = 8,
    columns: int = 4,
    padding: int = 2,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _sequence_to_pil_frames(sequence[:max_frames])
    width, height = frames[0].size
    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)

    grid = Image.new(
        "L",
        (
            columns * width + padding * (columns - 1),
            rows * height + padding * (rows - 1),
        ),
        color=0,
    )

    for frame_index, frame in enumerate(frames):
        row = frame_index // columns
        column = frame_index % columns
        x = column * (width + padding)
        y = row * (height + padding)
        grid.paste(frame, (x, y))

    grid.save(output_path)
    return output_path


def save_sample_visualizations(
    dataset: MovingMNISTDataset,
    output_dir: str | Path,
    *,
    sample_indices: Sequence[int] = (0, 1, 2),
    max_frames: int = 8,
) -> List[Path]:
    """Save a few sample GIFs and frame grids for notebook inspection."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for sample_index in sample_indices:
        sequence = dataset[sample_index]
        stem = f"{dataset.split}_sample_{sample_index:03d}"
        saved_paths.append(save_sequence_grid(sequence, output_dir / f"{stem}_grid.png", max_frames=max_frames))
        saved_paths.append(save_sequence_gif(sequence, output_dir / f"{stem}.gif"))
    return saved_paths


def describe_dataset_usage() -> str:
    """Summarize how the data fits later world-model or agent stages."""

    return (
        "Each sample is a grayscale video tensor with shape (T, C, H, W). "
        "Later world-model stages can consume batches shaped (B, T, C, H, W), "
        "slice early frames as context, and predict later frames or latent dynamics."
    )
