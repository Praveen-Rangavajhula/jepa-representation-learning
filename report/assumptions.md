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

## Runtime behavior

- The notebook is the primary execution surface.
- The intended execution backend is a Colab-connected Jupyter kernel, including the PyCharm-to-Colab workflow.
- Hugging Face model files are cached under `/content/.cache/huggingface` in Colab by default.
- Extra fallback dependencies such as `timm` and `einops` are not installed unless the torch-hub path is explicitly needed.

## Evaluation defaults

- The first small benchmark run uses `N=64` task examples by default to stay practical on Colab GPU.
- Confidence is defined as the softmax top-1 minus top-2 margin over candidate scores.
- Uncertainty buckets are:
  - `high >= 0.25`
  - `medium >= 0.10 and < 0.25`
  - `low < 0.10`

## Known limitations

- No learned compatibility head is introduced in this stage.
- No training loop for the V-JEPA compatibility metric is introduced in this stage.
- The current desktop workspace does not expose a straightforward local Python runtime for full end-to-end execution, so the primary runtime verification target remains Colab.
