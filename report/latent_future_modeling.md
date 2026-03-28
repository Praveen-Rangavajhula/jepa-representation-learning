# Latent Future Modeling Stage

## Goal

Move the project beyond hand-engineered latent scoring by introducing a first learned future model on top of frozen V-JEPA embeddings.

## Design

- V-JEPA remains the representation backbone.
- The encoder stays frozen.
- A small MLP predictor head maps the observed-prefix latent to an expected future latent.
- Candidate futures are encoded with the same frozen V-JEPA backbone.
- Candidates are ranked by similarity to the predicted future latent.

## Why this is more world-model-style

The engineered scorer asks whether a candidate looks temporally compatible with the observed clip. The learned latent predictor instead asks what latent future should come next and then compares candidates to that prediction.

That is still lightweight, but it is a real step toward a world-model trajectory:

- observed latent state
- predicted next latent state
- candidate evaluation against the predicted latent future

## Training flow

1. Build training pairs from the existing future-selection dataset.
2. Encode observed prefixes and true futures with frozen V-JEPA.
3. Train the predictor head with a cosine-plus-MSE loss in latent space.
4. Score candidates by predicted-future similarity.

## Current scope

- no decoder
- no pixel-space reconstruction
- no latent rollout beyond the first future segment
- no large-scale training infrastructure

## Intended next step

Once this predictor stage is stable, the project can extend to:

- longer latent rollouts
- pairwise compatibility heads as an ablation
- real-world dataset backends using the same task API
