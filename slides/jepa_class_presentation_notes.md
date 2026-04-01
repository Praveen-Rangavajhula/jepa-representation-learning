# JEPA Class Presentation Notes

## Slide 1 - Title
- Open with the core question: can a JEPA-style video model choose a likely next future and explain that choice?
- Frame this as a self-driven project for UGA CSCI 6860 rather than a polished product.
- Set expectations early: this is a working prototype with honest strengths and honest limitations.

## Slide 2 - Why This Project
- Frame the motivation as a move from plain action recognition toward a more world-model-style decision problem.
- Emphasize the two outputs: future selection and grounded commentary.
- Transition: before touching real video, you needed a controllable task that made the pipeline easy to debug.

## Slide 3 - Future-Selection Task
- Walk through the observed clip plus candidate futures at a high level.
- Explain that this is a decision problem: pick the best continuation, not the class label.
- Point out that this task structure stays the same across toy and real-video settings.

## Slide 4 - Control Demo: Moving MNIST
- Use this slide to show the pipeline works in a cheap, interpretable environment.
- Mention that the saved control result is strong and helped stabilize preprocessing and scoring.
- If the GIF renders statically in Typst/PDF, keep the original GIF open in a browser during the talk.
- Transition: once the pipeline was stable, you moved to natural video.

## Slide 5 - Real-Video Setup
- Explain why Something-Something V2 was a better temporal reasoning dataset than plain action classification.
- Keep the task explanation simple: true continuation, temporal negative, paired counterfactual.
- Mention that you deliberately simplified the candidate set for a class-demo path.

## Slide 6 - Successful Example
- Tell the audience this is the strongest qualitative example from the real-video run.
- Highlight the interesting part: the model picked the true continuation while the heuristic baseline disagreed.
- Stress that the margin is tiny, so you are not overselling it.
- Transition: now that the model can make a choice, the next question is whether the live commentary layer actually works.

## Slide 7 - Live AI Commentary Works
- This is the proof slide for the agent angle.
- Explicitly say: all 5 transcript entries used `colab_ai`, and fallback stayed at 0.
- Explain that this means the commentary layer was not just deterministic template text.
- Transition: once the live loop worked, the most useful next thing was to inspect where it still fails.

## Slide 8 - Failure Example
- Present this as the most informative failure, not as an embarrassment.
- The model is not failing on obvious nonsense; it is failing on plausible but temporally scrambled futures.
- Point out that the heuristic baseline getting this example right tells you the scoring wrapper still needs work.

## Slide 9 - What Was Built
- Keep this slide focused on system contributions, not file names.
- Mention that the project includes real-video ingestion, frozen V-JEPA scoring, grounded commentary, and live transcript generation.
- This is the slide that answers what was actually implemented beyond the pretrained backbone itself.

## Slide 10 - Evaluation Snapshot
- Keep the tone honest and calm.
- Say clearly that on the tiny saved real-video slice, the heuristic baseline outperformed the frozen V-JEPA scorer.
- Then immediately reframe: the main achievement here is the end-to-end demo system plus the identified failure mode.
- Transition: so what was worth learning from the project even without a strong real-video score?

## Slide 11 - Key Lessons
- Talk about the difference between having a strong pretrained backbone and having the right downstream scoring rule.
- Mention that temporal order, not raw semantics, became the key challenge.
- End by saying the project taught you how to turn a cutting-edge model into a real experimental system.

## Slide 12 - Next Step
- Position live video commentary as the natural continuation of the exact same architecture.
- Be careful not to claim it already works on live camera input today.
- Say the system already has the right abstraction boundary: observed clip, candidate futures, scorer, commentary.
- Close with confidence: this is a working platform for further exploration, not a dead-end project.
