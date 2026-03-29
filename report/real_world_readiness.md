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

## What has now changed for real-world video

- the repo now includes a generic real-video adapter that still emits the existing future-selection example format
- the primary real-world path is now a Something-Something V2 adapter that targets a standard parquet/video Hugging Face dataset route
- the earlier script-based Something-Something Hugging Face loader path is still unsupported in current `datasets` runtimes, so the adapter no longer relies on that route
- a UCF101 fallback adapter is available when the Something-Something ingest path is unavailable in a given runtime
- candidate construction now supports:
  - true continuation
  - shuffled temporal order
  - same-template other-sample future
  - paired-template counterfactual future with fallback logic
- commentary inputs now accept neutral observed descriptions plus human-readable candidate descriptions

## Why this is still portable

- the future-selection task remains clip-based, not dataset-specific
- the engineered scorer still operates on observed clip + candidate futures
- the commentary layer consumes structured evidence, not dataset-specific tensors
- a local JSONL manifest adapter is available for future non-Hugging-Face datasets without changing the downstream evaluation path

## Default real-world dataset

The primary target real-world dataset is now Something-Something V2 because it is:

- strong on temporal reasoning
- visually understandable for a class demo
- better aligned with the future-selection objective than generic action classification datasets

The temporary runnable fallback is UCF101 because it is:

- available in a standard parquet-backed Hugging Face video dataset
- visually understandable for a class demo
- much closer to natural video than Moving MNIST while still being manageable on Colab when cached as a small subset
