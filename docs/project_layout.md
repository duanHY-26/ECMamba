# Project layout

## Core files

- `esm2_3b_msa_mamba_only_gated_split100_train_eval.py`: original end-to-end training and evaluation entry point
- `src/ecmamba/cli.py`: package-style wrapper so the repository can expose a stable CLI entry point
- `requirements.txt`: runtime dependency list
- `pyproject.toml`: package metadata and CLI definition

## Documentation

- `README.md`: repository overview, benchmark summary and basic usage
- `RESOURCE_ESTIMATE.md`: projected memory and time ranges with reporting caveats
- `docs/reproducibility.md`: required inputs, expected outputs and reporting notes
- `manuscript/`: manuscript and supplementary materials prepared for submission

## Configuration and tests

- `configs/train_eval_example.yaml`: example parameter set for a full run
- `tests/test_cli_import.py`: simple import-level sanity check for the CLI wrapper

## Assets

- `model_comparison_full.png`: supplied benchmark figure
- `assets/ecmamba_gate_equation.png`: rendered fusion equation used in manuscript materials
