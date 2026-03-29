# V-JEPA 2 Integration Notes

## Architecture

- `src/jepa/models/model_config.py`
  - centralizes runtime defaults for the selected checkpoint, device, dtype, cache, and fallback backend
- `src/jepa/models/video_preprocessing.py`
  - normalizes `(T, C, H, W)` and `(B, T, C, H, W)` inputs
  - handles channel conversion, spatial resize, and temporal resampling
- `src/jepa/models/vjepa2_adapter.py`
  - loads V-JEPA 2 from Hugging Face first
  - falls back to official `torch.hub` only if needed
  - exposes `encode_video`, `encode_batch`, masked predictor token extraction, and `score_future_candidates`
- `src/jepa/scoring/vjepa_future_scorer.py`
  - defines the V-JEPA future-selection scoring bundle
  - implements `masked_future_prediction`, `overlap_transition`, and `prefix_future_cosine`
- `src/jepa/agents/future_selection_agents.py`
  - keeps the task API unchanged
  - makes evaluator mode configurable without changing the task dataset

## Scoring method

- Primary frozen-V-JEPA scorer: `masked_future_prediction`
- For each candidate, build a combined 16-frame clip from:
  - observed frames `0:8`
  - candidate future frames `8:16`
- Resample the combined clip to the V-JEPA frame count and create:
  - context masks covering the observed-prefix portion
  - target masks covering equal future temporal groups when possible
- Run the V-JEPA predictor on the masked task clip and compare predicted future tokens to target future tokens.
- Compute score components such as:
  - boundary future alignment
  - full future alignment across masked groups
  - order score against reversed future-group alignment
  - transition consistency across predicted future groups
- Final score is a weighted combination of those predictor-aware components.

## Encoder-only fallback

- `overlap_transition` remains available as an encoder-only fallback.
- It scores three overlapping 8-frame segments:
  - `A = 0:8`
  - `B = 4:12`
  - `C = 8:16`
- It computes:
  - `cosine_a_b`
  - `cosine_b_c`
  - `transition_smoothness = 1 / (1 + ||(zB - zA) - (zC - zB)||)`
- Final score:
  - `0.4 * cosine_a_b + 0.4 * cosine_b_c + 0.2 * transition_smoothness`

## Why this design

- It preserves the existing Moving MNIST and future-selection interfaces.
- It keeps preprocessing reusable for future real video datasets.
- It uses the V-JEPA predictor and masks more faithfully than plain pooled encoder similarity, while still keeping the backbone frozen.
- It lets the LLM-agentic layer consume score tables and summaries rather than raw tensors.
- It prioritizes a concrete working representation-based evaluator before introducing any learned compatibility head.
- Because the notebook preprocessing already handles resize and temporal resampling, the Hugging Face video processor is called with duplicate resize, center-crop, and rescale steps disabled.
