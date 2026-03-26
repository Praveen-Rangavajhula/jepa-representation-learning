# Assumptions

This file records the assumptions made while moving the project forward without waiting for more decisions.

## Colab and notebook workflow

- The notebook is the primary execution surface.
- The runtime is a Google Colab-backed kernel, even if the notebook is opened from PyCharm.
- The Colab runtime can access GitHub to clone the repository into `/content`.
- Manual dataset upload is not required; `torchvision.datasets.MNIST` downloads the source digits automatically.

## Moving MNIST dataset stage

- The current dataset stage uses 16-frame grayscale sequences with shape `(T, C, H, W)`.
- The default split for the future-selection task is:
  - observed context: frames `0:8`
  - future segment: frames `8:16`
- The Moving MNIST generator remains lightweight and on-the-fly rather than precomputing a large artifact set.

## Future-selection task layer

- Each task example contains 4 candidate futures total.
- Candidate order is randomized per example so the true continuation is not always in a fixed slot.
- The task metadata stores candidate strategy labels and the `correct_index`.
- Negative candidates are designed to be plausible rather than adversarially optimal.
- The current task layer is written to be reusable for other sequence datasets later, but it is only verified against Moving MNIST right now.

## Negative-candidate strategies

- The implemented strategy pool includes:
  - `shuffled_temporal_order`
  - `wrong_velocity_continuation`
  - `wrong_direction_continuation`
  - `future_segment_from_other_sample`
  - `mirrored_or_perturbed_continuation`
- Only 3 negatives are sampled per example, so not every strategy appears in every item.

## Agent pipeline skeleton

- The current `PlannerAgent`, `EvaluatorAgent`, `CriticAgent`, and `ExecutiveAgent` are placeholder components.
- The evaluator uses motion-consistency heuristics instead of a learned representation model.
- The design leaves hooks for a future pretrained representation model, but that model is not yet implemented or integrated.
- The current metrics should be treated as pipeline sanity checks, not as final research claims.

## Notebook polish and saved artifacts

- The notebook is structured to tell a coherent story for a class or professor review.
- Saved artifacts live under:
  - `results/moving_mnist/`
  - `results/task_examples/`
  - `results/agent_demo/`
- JSON logs and metrics are saved from the notebook rather than a separate training script for now.

## Scope limits

- No learned world model is implemented yet.
- No training loop for the future-selection task is included yet.
- No representation-learning backbone is attached to the agent pipeline yet.
- The current code aims for clarity, reproducibility, and forward progress rather than final model performance.
