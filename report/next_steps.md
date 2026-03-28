# Next Steps

## Immediate

- Run the updated notebook on a Colab GPU runtime from top to bottom.
- Confirm that the official V-JEPA checkpoint loads successfully through the Hugging Face path.
- Inspect the preprocessing and embedding sanity outputs before trusting the evaluation results.
- Inspect the grounded commentary text on both the engineered V-JEPA scorer and the learned latent predictor.
- Check whether the learned latent predictor materially improves over the engineered V-JEPA scorer on the controlled benchmark.

## After first successful Colab run

- Tune the latent predictor training slice, hidden dimension, and epochs for a better quality/runtime balance on Colab.
- Decide whether the engineered `overlap_transition` scorer should remain as the main representation baseline once the learned latent predictor is stable.
- Expand evaluation to a larger subset once runtime and memory behavior are stable.

## Before the LLM-agentic stage

- Wire a live LLM to the new commentary context package so the commentary layer can move beyond templates while staying evidence-grounded.
- Decide whether to add a pairwise compatibility head in addition to the predictor-style latent future model.
- Prepare the next dataset migration to KTH Actions while keeping the same future-selection and commentary interfaces.
