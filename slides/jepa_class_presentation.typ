// Compile this in typst.app with the repository contents available so
// the relative asset paths under ../results and ../report resolve.

#import "@preview/diatypst:0.9.1": *

#let deep-teal = rgb("#0F5B53")
#let soft-bg = rgb("#F4F8F7")
#let warm-bg = rgb("#FFF6E9")
#let rose-bg = rgb("#FDECEC")
#let muted = rgb("#5B6470")
#let uga-red = rgb("#BA0C2F")

#show: slides.with(
  title: "World-Model-Style Future Selection with JEPA",
  subtitle: "CSCI 6860: Computational Neuroscience",
  date: "March 2026",
  authors: "Praveen Rangavajhula | University of Georgia",
  ratio: 16/9,
  layout: "medium",
  theme: "normal",
  title-color: deep-teal,
  toc: false,
  count: "dot-section",
  footer: true,
  footer-title: "JEPA Future Selection",
  footer-subtitle: "UGA CSCI 6860 | Computational Neuroscience",
)

#let card(title, body, fill: soft-bg, stroke: deep-teal) = box(
  width: 100%,
  inset: 14pt,
  radius: 10pt,
  fill: fill,
  stroke: (paint: stroke, thickness: 0.8pt),
  [
    #text(weight: "bold")[#title]
    #v(0.35em)
    #body
  ],
)

#let stat(label, value, note: none, fill: soft-bg) = box(
  width: 100%,
  inset: 12pt,
  radius: 10pt,
  fill: fill,
  stroke: (paint: deep-teal, thickness: 0.8pt),
  [
    #text(size: 10pt, fill: muted)[#label]
    #linebreak()
    #text(size: 22pt, weight: "bold")[#value]
    #if note != none [
      #v(0.2em)
      #text(size: 9pt, fill: muted)[#note]
    ]
  ],
)

#let quote_box(title, body, fill: white, stroke: deep-teal) = box(
  width: 100%,
  inset: 14pt,
  radius: 10pt,
  fill: fill,
  stroke: (paint: stroke, thickness: 0.8pt),
  [
    #text(weight: "bold")[#title]
    #v(0.35em)
    #body
  ],
)

#let slide_caption(body) = align(center)[
  #text(size: 9pt, fill: muted)[#body]
]

= Framing

== Why This Project

#quote_box([Project framing], [
  This project evaluates whether a JEPA-style video representation can support *future selection* rather than only action recognition: given an observed clip prefix, select the most plausible next event and generate grounded commentary for that decision.
], fill: soft-bg)

#v(0.75em)
#grid(
  columns: (1fr, 0.14fr, 1fr, 0.14fr, 1fr),
  gutter: 0.22cm,
  [
    #card([Observed prefix], [A short video prefix provides temporal context for the decision.])
  ],
  [
    #align(center + horizon)[#text(size: 20pt, fill: muted)[->]]
  ],
  [
    #card([Candidate futures], [A small set of plausible next clips is compared instead of assigning a class label.])
  ],
  [
    #align(center + horizon)[#text(size: 20pt, fill: muted)[->]]
  ],
  [
    #card([Score + commentary], [A scorer ranks the futures, and the commentary layer explains the top choice in readable terms.])
  ],
)

#v(0.45em)
#align(center)[
  #text(size: 9pt, fill: muted)[University of Georgia | CSCI 6860 | Computational Neuroscience]
]

== Future-Selection Task

#grid(
  columns: (1.45fr, 0.55fr),
  gutter: 0.8cm,
  [
    #image("../results/task_examples/future_selection_000_panel.png", width: 100%)
  ],
  [
    #card([Core task], [
      - Observe a short prefix.
      - Compare a small set of candidate futures.
      - Select the most plausible continuation.
    ])
    #v(0.55em)
    #quote_box([Why this framing], [
      The task emphasizes short-horizon prediction and temporal reasoning rather than only static recognition.
    ], fill: warm-bg, stroke: deep-teal)
  ],
)

#v(0.35em)
#slide_caption([Control-task panel: observed clip at the top, followed by candidate futures to rank.])

= From Toy To Real Video

== Control Demo: Moving MNIST

#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 0.8cm,
  [
    #image("../results/moving_mnist/train_sample_000.gif", width: 88%)
    #slide_caption([Control demo asset from the Moving MNIST path.])
  ],
  [
    #stat([Control top-1], [1.00], note: [Frozen V-JEPA on the saved 8-example control slice])
    #v(0.45em)
    #stat([Heuristic baseline], [0.875], note: [Strong baseline in the toy setting])
    #v(0.45em)
    #card([Why this stage mattered], [
      - It validated preprocessing, scoring, and artifact writing.
      - It made failures cheap to inspect before moving to natural video.
      - It provided a stable control benchmark for comparison.
    ])
  ],
)

== Real-Video Setup: Something-Something V2

#grid(
  columns: (1.35fr, 0.65fr),
  gutter: 0.8cm,
  [
    #image("../results/real_video_eval/single_example_panel.png", width: 100%)
    #slide_caption([Observed prefix on top, then a true continuation, a counterfactual future, and a temporal-order negative.])
  ],
  [
    #card([Real-video task], [
      - Dataset slice: Something-Something V2
      - Simplified 3-choice setup
      - Designed for visually legible motion pairs
    ])
    #v(0.55em)
    #card([Three candidate types], [
      - True continuation
      - Temporal-order negative
      - Paired counterfactual
    ])
    #v(0.55em)
    #card([Why this dataset], [
      It emphasizes temporal reasoning more than static object appearance, which makes it a better test for future selection.
    ], fill: warm-bg, stroke: uga-red)
  ],
)

== Successful Example: The Model Picks The Right Future

#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 0.8cm,
  [
    #stat([Example 0], [Correct], note: [The V-JEPA path selected the true continuation])
    #v(0.45em)
    #stat([Top-two score gap], [0.0010], note: [A near tie, but still the right answer])
    #v(0.45em)
    #card([Why this example matters], [
      - The V-JEPA path selected the true continuation on real video.
      - The heuristic baseline preferred a temporal reversal on the same example.
      - The commentary layer converted the decision into presentation-ready evidence.
    ])
  ],
  [
    #quote_box([Grounded takeaway], [
      On this example, the system selects the true continuation for *Opening something*. The margin is small, but the correct future still outranks the temporal reversal and yields a grounded explanation of the choice.
    ])
    #v(0.55em)
    #card([Saved proof points], [
      - Selected index: 0
      - Correct index: 0
      - Heuristic baseline disagreed
      - Commentary artifact written under `results/real_video_eval/`
    ], fill: soft-bg)
  ],
)

= Live Agent

== Live AI Commentary Works

#grid(
  columns: (0.8fr, 1.2fr),
  gutter: 0.8cm,
  [
    #stat([Live agent examples], [5], note: [Saved as JSONL + Markdown transcripts])
    #v(0.45em)
    #stat([Colab AI backend], [5 / 5], note: [All transcript entries used `colab_ai`])
    #v(0.45em)
    #stat([Fallback count], [0], note: [No deterministic fallback was needed])
  ],
  [
    #card([What the transcript records], [
      - selected candidate
      - correctness
      - score and confidence margins
      - grounded natural-language commentary
      - saved artifacts for later presentation
    ])
    #v(0.55em)
    #quote_box([Why this matters], [
      The important win here is not just the score. The system can now produce a grounded explanation on each example and log that explanation as a reusable live-demo transcript.
    ], fill: white)
    #v(0.55em)
    #box(
      width: 100%,
      inset: 12pt,
      radius: 10pt,
      fill: soft-bg,
      stroke: (paint: deep-teal, thickness: 0.8pt),
      [
        `backend_counts = { colab_ai: 5 }`
        #linebreak()
        `fallback_count = 0`
      ],
    )
  ],
)

== Failure Example: Temporal Order Is Still Hard

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8cm,
  [
    #card([Example 3 decision], [
      - Observed action: *Opening something*
      - Predicted: reverse-block temporal negative
      - Correct: true continuation
      - Confidence margin: 0.0008
    ], fill: rose-bg, stroke: uga-red)
    #v(0.55em)
    #quote_box([Failure caption], [
      This is the clearest real-video failure mode: the wrong future looks plausible frame by frame, so the scorer overweights local compatibility and misses the correct temporal order.
    ], fill: white, stroke: uga-red)
  ],
  [
    #card([Why it failed], [
      - The top two candidates were almost tied.
      - The model preferred a temporally reversed version of the same action.
      - The heuristic got this one right, so the scoring rule is still the weak point.
    ], fill: warm-bg, stroke: uga-red)
    #v(0.55em)
    #stat([Main challenge], [Temporal order], note: [The hardest confusion in the real-video slice])
    #v(0.45em)
    #card([Interpretation], [
      The backbone is not useless here. The problem is that the frozen scoring wrapper is still too weak at separating a true future from a very plausible temporal scramble.
    ])
  ],
)

= Takeaways

== What Was Built

#grid(
  columns: (1fr, 1fr),
  gutter: 0.65cm,
  [
    #card([Real-video path], [
      A clip-based adapter for Something-Something V2 that preserves the same future-selection task contract.
    ])
  ],
  [
    #card([Frozen V-JEPA scoring], [
      A boundary-focused scorer that compares candidate futures without fine-tuning the backbone end to end.
    ])
  ],
  [
    #card([Grounded commentary], [
      A commentary layer that turns score evidence into readable explanations instead of free-form narration.
    ])
  ],
  [
    #card([Live agent artifacts], [
      A transcript writer that logs example-by-example predictions and commentary through Colab AI.
    ])
  ],
)

== Evaluation Snapshot

#grid(
  columns: (0.8fr, 0.8fr, 1.4fr),
  gutter: 0.65cm,
  [
    #stat([Heuristic top-1], [0.50], note: [Tiny 8-example real-video slice])
  ],
  [
    #stat([Boundary-hybrid top-1], [0.25], note: [Promising demo path, not yet a strong benchmark result])
  ],
  [
    #card([Honest takeaway], [
      - On the saved real-video slice, the heuristic baseline still outperformed the frozen V-JEPA scorer.
      - The hardest errors are temporally scrambled futures that still look locally plausible.
      - So the project is strongest today as an end-to-end demo system, not a finished benchmark result.
    ], fill: warm-bg, stroke: uga-red)
  ],
)

== Key Lessons

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.6cm,
  [
    #card([Backbone != task], [
      A powerful pretrained video model still needs the right scoring interface for the downstream task.
    ])
  ],
  [
    #card([Temporal order matters], [
      The most interesting failure mode was not semantics, but distinguishing correct order from a plausible reversal.
    ])
  ],
  [
    #card([Systems work matters], [
      Adapters, caching, commentary, and logging are what turned a research model into a usable class demo.
    ])
  ],
)

#v(0.6em)
#quote_box([Main takeaway], [
  The central lesson is that a strong pretrained backbone is only part of the system. Task design, scoring, logging, and commentary are what make the model inspectable and usable in practice.
])

== Next Step: Toward Live Video Commentary

#grid(
  columns: (1fr, 0.18fr, 1fr, 0.18fr, 1fr),
  gutter: 0.3cm,
  [
    #card([1. Stream recent frames], [
      Replace cached clips with live webcam or phone video.
    ])
  ],
  [
    #align(center + horizon)[#text(size: 22pt, fill: muted)[->]]
  ],
  [
    #card([2. Propose likely futures], [
      Keep a small set of plausible next snippets or candidate futures for comparison.
    ])
  ],
  [
    #align(center + horizon)[#text(size: 22pt, fill: muted)[->]]
  ],
  [
    #card([3. Score + narrate], [
      Reuse the same future-selection and commentary interface, but run it continuously.
    ])
  ],
)

#v(0.65em)
#card([Closing thought], [
  This project already works as a future-selection plus commentary prototype. The natural next step is to make the observed clip live instead of cached.
], fill: soft-bg)
