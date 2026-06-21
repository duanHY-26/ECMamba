# Reproducibility notes

## Inputs required

The current workflow requires:

- a training table for the split100 partition;
- evaluation tables for the NEW and Price benchmark sets;
- precomputed ESM representations stored in PyTorch, NumPy, pickle or HDF5 format;
- one MSA per protein in A3M, A2M, FASTA or related alignment format.

## Important reporting constraints

The training EC vocabulary is derived strictly from the training partition. Evaluation labels outside that vocabulary cannot be predicted, and samples with no covered labels are excluded only after coverage accounting. Any benchmark report should therefore state label coverage together with predictive metrics.

Threshold sweeps are implemented in code, but test-optimal thresholds are diagnostic only. Final operating thresholds should be selected on an internal validation or calibration split.

## Expected outputs

Each run writes the following files to `out_dir`:

- `dataset_info.csv`
- `label_coverage_report.json`
- `metrics_threshold_sweep.csv`
- `summary_at_selected_threshold.csv`
- `best_threshold_on_test_for_diagnosis_only.csv`
- `per_label_metrics_at_selected_threshold.csv`
- `run_config.json`
- plot files produced by the metric comparison routine

## Efficiency note

The repository currently includes projected resource ranges in `RESOURCE_ESTIMATE.md`. These are planning estimates, not measured profiler results, and should not be presented as measured throughput.
