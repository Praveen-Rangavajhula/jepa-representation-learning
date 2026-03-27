# Next Steps

## Immediate

- Run the updated notebook on a Colab GPU runtime from top to bottom.
- Confirm that the official V-JEPA checkpoint loads successfully through the Hugging Face path.
- Inspect the preprocessing and embedding sanity outputs before trusting the evaluation results.

## After first successful Colab run

- Compare heuristic and V-JEPA rankings on a small benchmark slice.
- Decide whether the `overlap_transition` score should remain the default or whether another latent compatibility variant is stronger.
- Expand evaluation to a larger subset once runtime and memory behavior are stable.

## Before the LLM-agentic stage

- Finalize the tensor-free handoff contract for:
  - observed clip summaries
  - candidate score tables
  - ranking summaries
  - uncertainty indicators
- Decide whether to add a lightweight learned compatibility head on top of frozen V-JEPA embeddings.
- Generalize the preprocessing and scoring path from Moving MNIST to a real dataset backend.
