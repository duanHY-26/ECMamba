# Supplementary Material for ECMamba

## S1 Compliance note for *Bioinformatics* Application Notes

This supplementary file contains material that is useful for review and reproducibility but is not essential to keep in the 4-page main text. The main article therefore retains only the central software description, one benchmark figure and the key performance summary. Extended methodological details, machine-learning dataset notes, benchmark tables and projected resource estimates are moved here to keep the main article consistent with the short Application Note format.

## S2 Machine-learning dataset description

The released workflow trains on the split100 partition and evaluates on the NEW and Price benchmark sets. The HARD split is deliberately excluded in the current implementation. The EC vocabulary is defined strictly from labels observed in the training partition, and evaluation samples with no covered labels are excluded only after coverage accounting. This point matters because unsupported labels and sample filtering can change the apparent benchmark difficulty.

The code accepts protein identifiers, EC labels and optional raw sequences from common table fields or explicitly mapped columns. EC assignments can be separated by semicolons, commas, tabs, pipes or spaces and are normalized before deduplication. On the feature side, the implementation reads precomputed ESM representations from PyTorch, NumPy, pickle or HDF5 containers. On the evolutionary side, the MSA loader accepts A3M, A2M, FASTA and related formats, removes lower-case insertions, retains gaps as a dedicated token, removes exact duplicate rows and preserves the query as the first row.

The training and test partitions are handled separately in code, and the benchmark metrics reported in the manuscript come from independent evaluation sets rather than cross-validation averages. A full release intended for journal submission should additionally state the provenance of split100, NEW and Price, and should document how homology between train and test proteins was controlled when those datasets were assembled.

## S3 Model and optimization details

The language-model branch applies LayerNorm, linear projection, dropout and attentive, mean and max pooling to the pretrained sequence embeddings. The MSA branch embeds aligned residues and then alternates row-wise Mamba with column-wise multi-head attention. Pooling is applied to the evolutionarily updated query row, after which a feature-wise gate combines language-model and MSA representations using the two vectors themselves, their absolute difference and their element-wise product.

Default training settings in the released script are 20 epochs, batch size 256, learning rate 5e-4, weight decay 1e-4, two warm-up epochs, automatic mixed precision on CUDA and gradient clipping at 1.0. The default objective is focal binary cross-entropy with focal exponent 2. Thresholds are swept from 0.005 to 0.5, while the selected operating point in the script is 0.07. The code explicitly warns that test-optimal thresholds are diagnostic only and that final reporting should use a threshold chosen on an internal validation or calibration partition.

## S4 Extended benchmark tables

### Table S1. Qualitative positioning relative to mainstream method families

| Method family | Representative methods | Typical limitation | ECMamba advantage |
|---|---|---|---|
| Homology search | BLASTp | Performance depends strongly on close matches and search configuration; no learnable fusion of heterogeneous evidence | Combines pretrained semantics with family-level alignment context and shows a larger F1 margin on the shifted Price-149 set |
| Sequence-only classifiers | ProteInfer, DeepEC, DEEPre, ECPred | Use single-sequence evidence only and do not model residue agreement across homologs | Adds explicit evolutionary context while keeping a compact protein-level classifier |
| PLM-only predictors | baseline_mlp, CLEAN | Strong sequence representations but no direct cross-homolog interaction in the forward pass | Feature-wise gated fusion improves recall with only minor precision loss |
| Full MSA attention models | MSA Transformer-style encoders | Rich alignment modeling but high all-cell attention cost on long or deep MSAs | Uses row-wise Mamba for linear-time sequence mixing and reserves attention for cross-homolog columns |

### Table S2. Benchmark scores transcribed from the supplied comparison figure

| Method | NEW P | NEW R | NEW F1 | Price P | Price R | Price F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECMamba | 0.662 | 0.634 | 0.601 | 0.603 | 0.592 | 0.570 |
| baseline_mlp | 0.674 | 0.602 | 0.592 | 0.608 | 0.520 | 0.527 |
| CLEAN | 0.597 | 0.481 | 0.499 | 0.584 | 0.467 | 0.495 |
| BLASTp | NR | NR | NR | 0.508 | 0.375 | 0.385 |
| ProteInfer | 0.409 | 0.284 | 0.309 | 0.243 | 0.138 | 0.166 |
| DeepEC | 0.298 | 0.217 | 0.230 | 0.118 | 0.072 | 0.085 |
| DEEPre | NR | NR | NR | 0.042 | 0.040 | 0.039 |
| ECPred | 0.118 | 0.095 | 0.100 | 0.020 | 0.020 | 0.020 |

`NR` indicates that the corresponding metric was not reported in the supplied figure for that dataset.

### Figure S1. Full benchmark comparison

![Figure S1. Benchmark comparison on NEW-392 and Price-149. ECMamba corresponds to the red series in the supplied composite figure.](e:/研究/ECMamba/model_comparison_full.png)

## S5 Gated-fusion equation

![Figure S2. Feature-wise gated fusion used in ECMamba.](e:/研究/ECMamba/assets/ecmamba_gate_equation.png)

The gate is initialized with a positive bias so that optimization starts from a stable preference for pretrained sequence features and only later increases the contribution of explicit evolutionary evidence where useful.

## S6 Projected computational profile

The current experiment package contains analytical planning ranges rather than measured profiler outputs. Under the reference workload of 100,000 proteins, 20 epochs, precomputed ESM features, one NVIDIA A100 80 GB accelerator, mixed precision, batch size 64, ESM length capped at 1000, and MSA tensors capped at 64 × 512, the projected cost is summarized below.

| Context | Setting | Metric | Value |
|---|---|---|---|
| Reference workload | 100,000 proteins; 20 epochs; one A100 80 GB; mixed precision; batch size 64 | Training wall time | 18-30 accelerator-hours |
| Reference workload | Same setting as above | Peak accelerator memory | 24-36 GB GPU memory |
| Inference workload | 10,000 proteins after ESM feature preparation and MSA construction | Inference latency | 3-8 min |
| Scope note | Planning values rather than profiler measurements | Excluded cost | ESM feature generation and MSA database search |

These values should not be presented as measured throughput. A submission that aims to claim efficiency should additionally report exact hardware, CUDA and PyTorch versions, trainable parameter count, sequence length distributions, padding efficiency, preprocessing cost and repeated-run variation.

## S7 Reproducibility files produced by the code

The training and evaluation script writes dataset statistics, label coverage, metric sweeps, selected-threshold summaries, per-label metrics, run configuration metadata and comparison plots to the output directory. These outputs are important because they preserve the basis for post hoc error analysis and make threshold choice auditable.

The most useful machine-readable outputs are:

- `dataset_info.csv`
- `label_coverage_report.json`
- `metrics_threshold_sweep.csv`
- `summary_at_selected_threshold.csv`
- `best_threshold_on_test_for_diagnosis_only.csv`
- `per_label_metrics_at_selected_threshold.csv`
- `run_config.json`

## S8 Current limitations

The supplied materials do not yet include multi-seed uncertainty, significance testing, ablation of the MSA stream, fixed-weight fusion controls or gate-value analysis. The benchmark figure is therefore encouraging but not yet sufficient for strong mechanistic claims. A polished journal submission should add these controls together with the archived code snapshot and, ideally, measured runtime and memory statistics.
