# Grounded Commentary Design

## Goal

The commentary layer turns future-selection outputs into short, professor-facing anticipatory commentary without inventing unsupported stories.

## Inputs

The commentary system consumes the existing tensor-free evidence bundle:

- observed clip summary
- observed description
- candidate clip summary
- candidate score table
- ranking summary
- selected candidate
- runner-up candidate
- confidence margin and confidence tier
- candidate metadata
- candidate descriptions
- optional baseline comparison

## Deterministic path

The deterministic path is implemented with templates so it always works in a local or Colab notebook.

It produces:

- the selected candidate index
- a short anticipation summary
- an explanation tied to the top score components
- an uncertainty statement
- a warning when the top two candidates are close
- a disagreement note when the heuristic baseline differs

## Live LLM path

The repo now supports both:

- an LLM-ready prompt/context package
- an optional live LLM path with deterministic fallback

The live path prepares:

- system instructions describing the grounding rules
- a user prompt with the structured evidence JSON
- a compact evidence payload for later language generation

When a live backend is available, the generated JSON is validated against a strict schema and rejected if it:

- selects a different candidate than the scorer
- mentions candidate indices outside the candidate set
- violates the required grounded output shape

If validation fails, the system falls back to deterministic commentary automatically.

## Guardrails

- no invented object identities or causes
- no unsupported claims about intent
- confidence margin and confidence tier must be surfaced explicitly
- baseline disagreement must be called out when present
- score evidence must remain visible next to the generated text in the notebook
