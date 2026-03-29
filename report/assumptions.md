# V-JEPA Integration Assumptions

This file is the authoritative assumptions log for the V-JEPA-backed stage of the project.

## Backbone choice

- The primary evaluator backend is the official Hugging Face checkpoint `facebook/vjepa2-vitl-fpc64-256`.
- The official `facebookresearch/vjepa2` `torch.hub` path is kept as a fallback only.
- The default Colab target is a single-GPU runtime such as a T4, so the ViT-L checkpoint is favored over larger variants for the first working integration.

## Input and preprocessing

- The existing future-selection task API remains unchanged:
  - observed clip: `(T_obs, C, H, W)`
  - candidate futures: `(K, T_future, C, H, W)`
- Moving MNIST remains grayscale at the dataset/task layer and is converted to RGB only inside V-JEPA preprocessing.
- Current task clips are shorter than V-JEPA pretraining clips, so they are deterministically resampled to 64 frames at scoring time.
- Spatial inputs are resized to `256x256`, matching the selected checkpoint family.
- Input pixel values are assumed to already be in `[0, 1]`.
- The Hugging Face video processor is used for model-compatible normalization, but duplicate resize, crop, and rescale steps are disabled because those are already handled in repo-side preprocessing.

## Scoring design

- The first V-JEPA scorer is representation-based and engineered, not learned.
- The default scoring variant is `overlap_transition`, which scores three overlapping 8-frame segments from each 16-frame observed+candidate clip.
- The lighter `prefix_future_cosine` method is implemented as an ablation, not the default evaluator.
- The heuristic motion-based evaluator is retained only as a baseline and fallback.
- The next modeling step keeps V-JEPA frozen and adds a small predictor head that maps observed-prefix latents to expected future latents.
- The first learned latent scorer ranks candidates by similarity to the predicted future latent rather than by pixel-space reconstruction.

## Commentary design

- The human-facing layer is grounded anticipatory commentary, not free-form narration.
- Commentary must be supported by the actual ranking summary, candidate score table, component breakdowns, confidence margins, confidence tiers, and optional baseline comparison.
- A deterministic template-based commentary path is always available locally.
- An LLM-ready commentary package is built from the same tensor-free evidence so a live language layer can be added without touching raw tensors.
- A live Gemini backend is now the default when `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or a Colab secret such as `google-api-key` is available.
- An OpenAI Responses API backend remains supported as a secondary live option when OpenAI credentials are available.
- If no live backend is available, the system falls back to the deterministic commentary path automatically.
- Commentary should warn explicitly when confidence is low or when the heuristic baseline disagrees with the representation-based evaluator.

## Runtime behavior

- The notebook is the primary execution surface.
- The intended execution backend is a Colab-connected Jupyter kernel, including the PyCharm-to-Colab workflow.
- Hugging Face model files are cached under `/content/.cache/huggingface` in Colab by default.
- The default live-LLM path in Colab is Gemini via the Google GenAI SDK, and the backend will attempt to read Colab secrets such as `google-api-key` automatically.
- Extra fallback dependencies such as `timm` and `einops` are not installed unless the torch-hub path is explicitly needed.

## Evaluation defaults

- The Moving MNIST control benchmark can still use a small slice for quick checks.
- The default real-world evaluation slice is `128` examples, with `320` examples reserved as the more confident validation pass.
- Confidence is defined as the softmax top-1 minus top-2 margin over candidate scores.
- The canonical categorical label for that margin is now `confidence_tier`; legacy `uncertainty` fields are treated as compatibility aliases only.
- Confidence tiers are:
  - `high >= 0.25`
  - `medium >= 0.10 and < 0.25`
  - `low < 0.10`

## Known limitations

- Moving MNIST remains a controlled benchmark even though it is mismatched with V-JEPA's natural-video pretraining domain.
- The learned latent predictor is intentionally small and benchmark-focused; it is a first serious world-model-style step, not a final world model.
- No pixel-space future generation is attempted in this stage.
- A live LLM runtime is optional rather than required; when it is unavailable, the repo falls back to deterministic grounded commentary.
- The current desktop workspace does not expose a straightforward local Python runtime for full end-to-end execution, so the primary runtime verification target remains Colab.

## Real-world readiness

- The future-selection task API remains clip-based so that dataset swaps do not require major scorer or commentary rewrites.
- Moving-MNIST-specific preprocessing assumptions are isolated in the data and model preprocessing layers.
- The primary real-world track is now a lightweight Something-Something V2 subset with cacheable 16-frame clips built from a fixed set of eight interpretable templates.
- Real-video candidates are constructed from a fixed recipe (true continuation, temporal shuffle, same-template other sample, paired-template counterfactual/fallback) and then shuffled deterministically before scoring so the correct answer is not tied to a single slot.
- The cache root defaults to `data/real_video_cache`, and cached clip manifests are treated as reusable notebook artifacts.
