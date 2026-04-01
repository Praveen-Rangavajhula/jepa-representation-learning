# Simple Real-Video Evaluation Summary

## Run metadata
- dataset_name: mteb/SomethingSomethingV2
- eval_split: test
- source_window_length: 32
- real_vjepa_model_id: facebook/vjepa2-vitl-fpc16-256-ssv2
- real_vjepa_target_frames: 16
- semantic_score_mode: max_probability
- hybrid_temporal_weight: 1.0
- hybrid_semantic_weight: 1.0
- demo_mode: True
- candidate_count: 3
- confusion_report: {'heuristic': {'temporal_order_block_reverse': {'count': 6, 'fraction_of_errors': 0.46153846153846156, 'fraction_of_examples': 0.1875}, 'temporal_order_two_block_swap': {'count': 3, 'fraction_of_errors': 0.23076923076923078, 'fraction_of_examples': 0.09375}, 'paired_template_counterfactual': {'count': 4, 'fraction_of_errors': 0.3076923076923077, 'fraction_of_examples': 0.125}}, 'boundary_hybrid': {'temporal_order_two_block_swap': {'count': 8, 'fraction_of_errors': 0.5, 'fraction_of_examples': 0.25}, 'temporal_order_block_reverse': {'count': 8, 'fraction_of_errors': 0.5, 'fraction_of_examples': 0.25}}, 'semantic_only': {'paired_template_counterfactual': {'count': 1, 'fraction_of_errors': 0.1, 'fraction_of_examples': 0.03125}, 'temporal_order_block_reverse': {'count': 3, 'fraction_of_errors': 0.3, 'fraction_of_examples': 0.09375}, 'temporal_order_two_block_swap': {'count': 6, 'fraction_of_errors': 0.6, 'fraction_of_examples': 0.1875}}, 'hybrid': {'paired_template_counterfactual': {'count': 2, 'fraction_of_errors': 0.2857142857142857, 'fraction_of_examples': 0.0625}, 'temporal_order_block_reverse': {'count': 3, 'fraction_of_errors': 0.42857142857142855, 'fraction_of_examples': 0.09375}, 'temporal_order_two_block_swap': {'count': 2, 'fraction_of_errors': 0.2857142857142857, 'fraction_of_examples': 0.0625}}}

## Metrics
- Evaluation count: 32
- heuristic Top-1 accuracy: 0.5938
- heuristic MRR: 0.7917
- heuristic average correct rank: 1.4375
- heuristic score margin mean: 0.1324
- heuristic score margin variance: 0.0111
- heuristic confidence margin mean: 0.0000
- heuristic confidence margin variance: 0.0000
- boundary_hybrid Top-1 accuracy: 0.5000
- boundary_hybrid MRR: 0.7500
- boundary_hybrid average correct rank: 1.5000
- boundary_hybrid score margin mean: 0.0085
- boundary_hybrid score margin variance: 0.0001
- boundary_hybrid confidence margin mean: 0.0029
- boundary_hybrid confidence margin variance: 0.0000
- semantic_only Top-1 accuracy: 0.6875
- semantic_only MRR: 0.8333
- semantic_only average correct rank: 1.3750
- semantic_only score margin mean: 0.1335
- semantic_only score margin variance: 0.0242
- semantic_only confidence margin mean: 0.0487
- semantic_only confidence margin variance: 0.0032
- hybrid Top-1 accuracy: 0.7812
- hybrid MRR: 0.8854
- hybrid average correct rank: 1.2500
- hybrid score margin mean: 0.2521
- hybrid score margin variance: 0.0481
- hybrid confidence margin mean: 0.0950
- hybrid confidence margin variance: 0.0068

## Focused Confusions
- heuristic -> temporal_order_block_reverse: count=6, fraction_of_errors=0.4615
- heuristic -> paired_template_counterfactual: count=4, fraction_of_errors=0.3077
- heuristic -> temporal_order_two_block_swap: count=3, fraction_of_errors=0.2308
- boundary_hybrid -> temporal_order_two_block_swap: count=8, fraction_of_errors=0.5000
- boundary_hybrid -> temporal_order_block_reverse: count=8, fraction_of_errors=0.5000
- semantic_only -> temporal_order_two_block_swap: count=6, fraction_of_errors=0.6000
- semantic_only -> temporal_order_block_reverse: count=3, fraction_of_errors=0.3000
- semantic_only -> paired_template_counterfactual: count=1, fraction_of_errors=0.1000
- hybrid -> temporal_order_block_reverse: count=3, fraction_of_errors=0.4286
- hybrid -> paired_template_counterfactual: count=2, fraction_of_errors=0.2857
- hybrid -> temporal_order_two_block_swap: count=2, fraction_of_errors=0.2857

## Artifact paths
- benchmark_summary: `/content/jepa-representation-learning/results/real_video_eval/benchmark_summary.json`
- per_negative_type: `/content/jepa-representation-learning/results/real_video_eval/per_negative_type.json`
- confusion_report: `/content/jepa-representation-learning/results/real_video_eval/confusion_report.json`
- candidate_rankings: `/content/jepa-representation-learning/results/real_video_eval/candidate_rankings.csv`
- per_example_rankings: `/content/jepa-representation-learning/results/real_video_eval/per_example_rankings.json`
