# Grounded Commentary Design

## Goal

The commentary layer turns future-selection outputs into short, professor-facing anticipatory commentary without inventing unsupported stories.

## Inputs

The commentary system consumes the existing tensor-free evidence bundle:

- observed clip summary
- candidate clip summary
- candidate score table
- ranking summary
- selected candidate
- runner-up candidate
- confidence and uncertainty
- candidate metadata
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

## LLM-ready path

The LLM-ready path does not call an LLM yet. Instead, it prepares:

- system instructions describing the grounding rules
- a user prompt with the structured evidence JSON
- a compact evidence payload for later language generation

This keeps the commentary layer modular and prevents later LLM integration from needing raw tensors.

## Guardrails

- no invented object identities or causes
- no unsupported claims about intent
- uncertainty must be surfaced explicitly
- baseline disagreement must be called out when present
- score evidence must remain visible next to the generated text in the notebook
