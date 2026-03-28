# Real-World Readiness Notes

## Current controlled benchmark

Moving MNIST remains the main benchmark because it keeps the future-selection task easy to inspect and cheap to run in Colab.

## Why a dataset swap is still needed later

V-JEPA is pretrained for natural video, so zero-shot performance on Moving MNIST is expected to be limited. The current benchmark is useful for controlled comparisons, but it is not the final target domain.

## What is already dataset-agnostic

- the future-selection task interface uses generic clip tensors
- the commentary layer consumes score evidence, not Moving-MNIST-specific semantics
- the engineered and learned scorers both operate on observed clips and candidate futures
- benchmark reporting is evaluator-based rather than dataset-specific

## What would change for real-world video

- replace the dataset loader
- redesign candidate-future construction to use real action or motion negatives
- revisit preprocessing choices such as frame sampling and crop strategy
- possibly add richer observed clip summaries for commentary

## Default next dataset

KTH Actions is the default next target because it is:

- lightweight enough for Colab
- visually interpretable for a professor-facing demo
- more aligned with V-JEPA than Moving MNIST
