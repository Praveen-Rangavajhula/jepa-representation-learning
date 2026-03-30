# Simple Real-Video Evaluation Summary

## Run metadata
- dataset_name: mteb/SomethingSomethingV2
- eval_split: test
- source_window_length: 32
- real_vjepa_model_id: facebook/vjepa2-vitl-fpc16-256-ssv2
- real_vjepa_target_frames: 16
- demo_mode: True
- candidate_count: 3
- confusion_report: {'heuristic': {'temporal_order_block_reverse': {'count': 3, 'fraction_of_errors': 0.75, 'fraction_of_examples': 0.375}, 'temporal_order_two_block_swap': {'count': 1, 'fraction_of_errors': 0.25, 'fraction_of_examples': 0.125}}, 'boundary_hybrid': {'temporal_order_two_block_swap': {'count': 1, 'fraction_of_errors': 0.16666666666666666, 'fraction_of_examples': 0.125}, 'temporal_order_block_reverse': {'count': 5, 'fraction_of_errors': 0.8333333333333334, 'fraction_of_examples': 0.625}}}

## Metrics
- Evaluation count: 8
- heuristic Top-1 accuracy: 0.5000
- heuristic MRR: 0.7292
- heuristic average correct rank: 1.6250
- heuristic score margin mean: 0.1581
- heuristic score margin variance: 0.0194
- heuristic confidence margin mean: 0.0000
- heuristic confidence margin variance: 0.0000
- boundary_hybrid Top-1 accuracy: 0.2500
- boundary_hybrid MRR: 0.6250
- boundary_hybrid average correct rank: 1.7500
- boundary_hybrid score margin mean: 0.0054
- boundary_hybrid score margin variance: 0.0000
- boundary_hybrid confidence margin mean: 0.0018
- boundary_hybrid confidence margin variance: 0.0000

## Focused Confusions
- heuristic -> temporal_order_block_reverse: count=3, fraction_of_errors=0.7500
- heuristic -> temporal_order_two_block_swap: count=1, fraction_of_errors=0.2500
- boundary_hybrid -> temporal_order_block_reverse: count=5, fraction_of_errors=0.8333
- boundary_hybrid -> temporal_order_two_block_swap: count=1, fraction_of_errors=0.1667

## Artifact paths
- benchmark_summary: `/content/jepa-representation-learning/results/real_video_eval/benchmark_summary.json`
- per_negative_type: `/content/jepa-representation-learning/results/real_video_eval/per_negative_type.json`
- confusion_report: `/content/jepa-representation-learning/results/real_video_eval/confusion_report.json`
- candidate_rankings: `/content/jepa-representation-learning/results/real_video_eval/candidate_rankings.csv`
- per_example_rankings: `/content/jepa-representation-learning/results/real_video_eval/per_example_rankings.json`
