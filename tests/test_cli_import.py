from pathlib import Path

from ecmamba.cli import main


def test_entry_script_exists():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "esm2_3b_msa_mamba_only_gated_split100_train_eval.py").exists()
    assert callable(main)
