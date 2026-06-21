# ECMamba

ECMamba is a research implementation for multi-label Enzyme Commission annotation that combines pretrained protein language model features with explicit evolutionary context from multiple sequence alignments. The method uses a lightweight pooled language-model branch, an axial MSA encoder with row-wise Mamba and column-wise homolog attention, and a feature-wise gate that fuses both sources before classification.

## Highlights

- adds explicit evolutionary context beyond sequence-only baselines;
- confines Mamba to the MSA branch instead of attaching another heavy backbone to pretrained language-model tokens;
- improves recall and F1 on the supplied NEW-392 and Price-149 benchmarks;
- records coverage, threshold sweeps and run configuration files for reproducibility;
- includes manuscript-ready main text and supplementary material for a `Bioinformatics` Application Note submission.

## Benchmark summary

The supplied benchmark figure reports support-weighted precision, recall and F1 on the NEW-392 and Price-149 evaluation sets.

| Method | NEW P | NEW R | NEW F1 | Price P | Price R | Price F1 |
|---|---:|---:|---:|---:|---:|---:|
| **ECMamba** | **0.662** | **0.634** | **0.601** | **0.603** | **0.592** | **0.570** |
| baseline_mlp | 0.674 | 0.602 | 0.592 | 0.608 | 0.520 | 0.527 |
| CLEAN | 0.597 | 0.481 | 0.499 | 0.584 | 0.467 | 0.495 |
| BLASTp | NR | NR | NR | 0.508 | 0.375 | 0.385 |
| ProteInfer | 0.409 | 0.284 | 0.309 | 0.243 | 0.138 | 0.166 |
| DeepEC | 0.298 | 0.217 | 0.230 | 0.118 | 0.072 | 0.085 |
| DEEPre | NR | NR | NR | 0.042 | 0.040 | 0.039 |
| ECPred | 0.118 | 0.095 | 0.100 | 0.020 | 0.020 | 0.020 |

`NR` means the corresponding value was not reported in the supplied composite figure for that dataset.

## Project structure

```text
ECMamba/
├─ src/ecmamba/                  # package wrapper and CLI entry point
├─ configs/                      # example configuration files
├─ docs/                         # reproducibility and project layout notes
├─ tests/                        # lightweight repository tests
├─ manuscript/                   # main manuscript and supplementary material
├─ assets/                       # equation and manuscript assets
├─ esm2_3b_msa_mamba_only_gated_split100_train_eval.py
├─ model_comparison_full.png
├─ requirements.txt
├─ pyproject.toml
└─ RESOURCE_ESTIMATE.md
```

## Installation

Create a CUDA-enabled Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

If you want the package-style CLI entry point:

```bash
pip install -e .
```

`mamba-ssm` must match the installed PyTorch and CUDA stack.

## Running the workflow

The repository keeps the original research script as the main executable entry point:

```bash
python esm2_3b_msa_mamba_only_gated_split100_train_eval.py --help
```

After editable installation, the same entry point is also exposed as:

```bash
ecmamba-train-eval --help
```

An example parameter file is provided in `configs/train_eval_example.yaml`.

## Inputs

The current workflow accepts:

- tabular protein identifiers, EC annotations and optional raw query sequences;
- precomputed language-model representations in PyTorch, NumPy, pickle or HDF5 containers;
- one MSA per protein in A3M, A2M, FASTA or related alignment formats.

The released script evaluates the NEW and PRICE partitions and deliberately excludes the HARD split.

## Reproducibility notes

The EC vocabulary is derived strictly from the training partition. Evaluation labels outside that vocabulary cannot be predicted, and samples with no covered labels are excluded after coverage accounting. Report label coverage together with predictive metrics.

Threshold sweeps are implemented in code, but final operating thresholds should be selected on an internal validation or calibration split. Test-optimal thresholds are diagnostic only.

See `docs/reproducibility.md` for the expected outputs written to `out_dir`.

## Computational profile

The current efficiency figures are analytical planning ranges rather than profiler measurements. Under the reference workload of 100,000 proteins, 20 epochs, precomputed ESM features, one NVIDIA A100 80 GB accelerator, mixed precision, batch size 64, ESM length capped at 1000, and MSA tensors capped at `64 x 512`, the projected cost is:

- peak accelerator memory: `24-36 GB`;
- training wall time: `18-30 accelerator-hours`;
- inference for 10,000 proteins: `3-8 min` after ESM feature preparation and MSA construction.

These ranges exclude ESM feature generation and MSA database search. See `RESOURCE_ESTIMATE.md` for assumptions, scope and reporting limitations.

## Manuscript files

The `manuscript/` directory contains:

- `ECMamba_application_note_main.docx`: shortened main text suitable for the `Bioinformatics` Application Note format;
- `ECMamba_supplementary_material.docx`: supplementary methods, extended tables and projected resource profile;
- Markdown source files for both documents.

## Citation

A formal citation will be added with the first archived release. Until then, cite the public repository and archived release identifier.
