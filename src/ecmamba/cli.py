"""Repository-local CLI wrapper for the ECMamba training and evaluation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_entry_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "esm2_3b_msa_mamba_only_gated_split100_train_eval.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find entry script: {script_path}")
    spec = importlib.util.spec_from_file_location("ecmamba_train_eval_entry", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_entry_module()
    if not hasattr(module, "main"):
        raise AttributeError("Entry script does not define main()")
    module.main()


if __name__ == "__main__":
    main()
