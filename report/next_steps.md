# Next Steps

## Immediate

- Run the updated notebook on a Colab GPU runtime from top to bottom.
- Confirm that the Something-Something subset cache builds successfully from either Hugging Face or a local manifest.
- Inspect one real-video example end to end:
  - observed prefix
  - candidate futures
  - engineered V-JEPA ranking
  - grounded LLM commentary or deterministic fallback
- Confirm that the 5-example live agent transcript is written under `results/agent_live/`.
- Confirm that the 128-example real-world evaluation artifacts are written under `results/real_video_eval/`.

## After the first successful real-world run

- Expand the real-world evaluation from `128` to the `320`-example confident pass.
- Review which candidate-generation strategies are easiest and hardest for the engineered scorer on Something-Something.
- Decide whether to keep the Moving MNIST control benchmark in the main notebook or move it to an appendix once the real-video path is stable.

## Before a fuller agentic/world-model stage

- Improve the observed-clip description so the LLM receives better neutral motion summaries without leaking the true label.
- Decide whether to add a pairwise compatibility head in addition to the predictor-style latent future model.
- If Something-Something proves too heavy for repeated Colab demos, prepare a small local-manifest fallback using the same real-video adapter interface.
