#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESM2-3B pooled embedding + MSA-Mamba encoder gated fusion for enzyme EC multi-label classification.

This split100 version trains on split100.csv and does not use HARD split data.

This script keeps the original RawSeq model's downstream architecture as much as possible:
  MSA (.a3m/.fa/.fasta) -> lightweight axial MSA encoder -> pooling
  ESM2 token/per-protein embedding -> LayerNorm + Linear projection -> masked pooling only
  MSA (.a3m/.fa/.fasta) -> lightweight axial MSA encoder with row-wise Mamba -> pooling
  -> learnable gated fusion -> MLP classifier -> multi-label EC logits.

Compared with the ESM-only version:
  1. An MSA stream is added as an evolutionary-context input.
  2. MSA features are extracted with a lightweight axial encoder: row-wise Mamba sequence mixing + column-wise homolog attention.
  3. ESM2 features and MSA features are fused by a learnable sigmoid gate.

Supported ESM embedding formats:
  - ESM extract .pt/.pth files containing dict keys such as:
      representations[layer], mean_representations[layer], contacts, etc.
  - .npy files: vector [D] or token matrix [L, D]
  - .npz files with keys embedding / esm / repr / representations / mean_representations
  - .pkl/.pickle containing ndarray/tensor/dict
  - .h5/.hdf5 where each protein ID is a dataset key

For Mamba sequence modeling, token-level ESM representations [L, 2560] are preferred.
If only mean-pooled vectors [2560] are available, the ESM branch treats each protein as a length-1 sequence and only pools/projects it.
Mamba is used only in the MSA encoder branch.

Typical command:
nohup python esm2_3b_mamba_split100_focal_train_eval.py \
  --root /nfs/hb236/dhy/app \
  --train_csv /nfs/hb236/dhy/app/data/split100.csv \
  --sequence_col Sequence \
  --new_csv /nfs/hb236/dhy/app/data/new_fixed.csv \
  --price_csv /nfs/hb236/dhy/app/data/price_fixed.csv \
  --eval_datasets NEW,PRICE \
  --esm_dir /nfs/hb236/dhy/app/data/esm_data_split100,/nfs/hb236/dhy/app/data/esm_data_new,/nfs/hb236/dhy/app/data/esm_data_price,/nfs/hb236/dhy/app/data/esm_data \
  --model_arch mamba \
  --loss focal_bce \
  --pos_weight_mode log \
  --pos_weight_max 8 \
  --epochs 20 \
  --batch_size 64 \
  --d_model 512 \
  --depth 8 \
  --classifier_hidden_dim 2048 \
  --mamba_expand 2 \
  --expansion 4 \
  --lr 5e-5 \
  --warmup_epochs 2 \
  --selected_threshold 0.07 \
  --out_dir /nfs/hb236/dhy/app-copy/analysis_results_esm2_3b_mamba_split100_focal_bce_d512_depth8 \
  > esm2_3b_mamba_split100_focal_bce_d512_depth8.log 2>&1 &
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import re
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except Exception:
    h5py = None
    HAS_H5PY = False

try:
    from mamba_ssm import Mamba  # type: ignore
    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


# =============================================================================
# IO / reproducibility
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_table_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Table file not found: {path}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep="\t")


def infer_id_col(df: pd.DataFrame) -> str:
    candidates = ["Entry", "entry", "ID", "id", "sequence_id", "seq_id", "protein_id", "Protein_ID", "name", "Name"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return df.columns[0]


def infer_ec_col(df: pd.DataFrame) -> str:
    candidates = [
        "EC_number", "ec_number", "EC", "ec", "ecs", "ECs", "ec_numbers", "EC_numbers",
        "label", "labels", "Label", "Labels", "ec_label",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if len(df.columns) < 2:
        raise ValueError(f"Cannot infer EC label column. Columns: {list(df.columns)}")
    return df.columns[1]


def get_full_path(root: str, name: str) -> str:
    candidates = [
        os.path.join(root, name), os.path.join(root, f"{name}.csv"), os.path.join(root, f"{name}_fixed.csv"),
        os.path.join(root, "data", name), os.path.join(root, "data", f"{name}.csv"), os.path.join(root, "data", f"{name}_fixed.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(root, "data", f"{name}.csv")


def normalize_ec_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        for sep in [";", ",", "|", "\t"]:
            if sep in text:
                return [p.strip() for p in text.split(sep) if p.strip() and p.strip().lower() != "nan"]
        if " " in text:
            return [p.strip() for p in text.split() if p.strip() and p.strip().lower() != "nan"]
        return [text]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for x in value:
            out.extend(normalize_ec_list(x))
        return out
    return [str(value)]


def get_ec_id_dict(csv_path: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    df = read_table_auto(csv_path)
    id_col = infer_id_col(df)
    ec_col = infer_ec_col(df)
    id_ec: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        ecs = [ec for ec in normalize_ec_list(row[ec_col]) if ec and ec.lower() != "nan"]
        if not sid or sid.lower() == "nan" or not ecs:
            continue
        id_ec.setdefault(sid, [])
        id_ec[sid] = sorted(set(id_ec[sid] + ecs))
    ec_counter: Dict[str, int] = {}
    for ecs in id_ec.values():
        for ec in ecs:
            ec_counter[ec] = ec_counter.get(ec, 0) + 1
    return id_ec, ec_counter


def infer_sequence_col(df: pd.DataFrame, sequence_col: Optional[str] = None) -> str:
    """Infer the raw amino-acid sequence column from a CSV/TSV table.

    If --sequence_col is supplied, this function accepts exact or case-insensitive
    matches and raises a clear error when the column is absent.
    """
    lower_map = {str(c).lower(): c for c in df.columns}
    if sequence_col:
        if sequence_col in df.columns:
            return sequence_col
        if sequence_col.lower() in lower_map:
            return lower_map[sequence_col.lower()]
        raise ValueError(f"Requested sequence column '{sequence_col}' not found. Columns: {list(df.columns)}")

    candidates = [
        "Sequence", "sequence", "seq", "Seq", "AA_sequence", "aa_sequence", "protein_sequence",
        "Protein_sequence", "amino_acid_sequence", "Amino_acid_sequence", "sequence_raw",
        "raw_sequence", "fasta", "FASTA", "Sequence_EC",
    ]
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(
        "Cannot infer raw amino-acid sequence column. Please pass --sequence_col. "
        f"Columns: {list(df.columns)}"
    )


def normalize_sequence(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    seq = str(value).strip().upper()
    if not seq or seq == "NAN":
        return ""
    # Remove FASTA header if a complete FASTA text was placed in a table cell.
    if seq.startswith(">"):
        lines = [ln.strip() for ln in seq.splitlines() if ln.strip() and not ln.startswith(">")]
        seq = "".join(lines)
    # Remove whitespace and terminal stop markers; map unknown/non-amino-acid chars to X.
    seq = re.sub(r"\s+", "", seq).replace("*", "")
    seq = re.sub(r"[^A-Z]", "X", seq)
    return seq


def get_sequence_dict(csv_path: str, sequence_col: Optional[str] = None) -> Dict[str, str]:
    df = read_table_auto(csv_path)
    id_col = infer_id_col(df)
    seq_col = infer_sequence_col(df, sequence_col)
    id_seq: Dict[str, str] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        seq = normalize_sequence(row[seq_col])
        if not sid or sid.lower() == "nan" or not seq:
            continue
        id_seq[sid] = seq
    return id_seq


def parse_thresholds(text: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    if not values:
        raise ValueError("No valid thresholds.")
    for v in values:
        if v <= 0 or v >= 1:
            raise ValueError("Thresholds must be in open interval (0, 1).")
    return values


def compute_label_support(id_ec_dict: Dict[str, Any], ec2idx: Dict[str, int]) -> np.ndarray:
    support = np.zeros(len(ec2idx), dtype=np.int64)
    for ecs_raw in id_ec_dict.values():
        for ec in normalize_ec_list(ecs_raw):
            if ec in ec2idx:
                support[ec2idx[ec]] += 1
    return support


def compute_label_coverage(id_ec_dict: Dict[str, Any], train_vocab: Iterable[str]) -> Dict[str, Any]:
    train_vocab = set(train_vocab)
    total_assignments = 0
    covered_assignments = 0
    unique_labels = set()
    covered_unique_labels = set()
    unseen_labels = set()
    sample_coverages = []
    for ecs_raw in id_ec_dict.values():
        ecs = normalize_ec_list(ecs_raw)
        if not ecs:
            continue
        n_total = len(ecs)
        n_covered = 0
        for ec in ecs:
            unique_labels.add(ec)
            total_assignments += 1
            if ec in train_vocab:
                covered_assignments += 1
                covered_unique_labels.add(ec)
                n_covered += 1
            else:
                unseen_labels.add(ec)
        sample_coverages.append(n_covered / max(n_total, 1))
    return {
        "n_samples_with_labels": len(sample_coverages),
        "n_unique_labels": len(unique_labels),
        "n_unique_labels_covered": len(covered_unique_labels),
        "n_unique_labels_unseen": len(unseen_labels),
        "unique_label_coverage": len(covered_unique_labels) / max(len(unique_labels), 1),
        "n_label_assignments": total_assignments,
        "n_label_assignments_covered": covered_assignments,
        "n_label_assignments_unseen": total_assignments - covered_assignments,
        "assignment_coverage": covered_assignments / max(total_assignments, 1),
        "mean_sample_label_coverage": float(np.mean(sample_coverages)) if sample_coverages else 0.0,
        "unseen_labels_preview": sorted(list(unseen_labels))[:50],
    }


# =============================================================================
# ESM embedding store and dataset
# =============================================================================

EMBED_EXTS = (".pt", ".pth", ".npy", ".npz", ".pkl", ".pickle")
H5_EXTS = (".h5", ".hdf5")

# MSA tokenizer for the evolutionary-context encoder.
# 0 is padding; 1 is unknown; 2 is alignment gap. Standard residues plus common
# ambiguous/special letters are assigned stable IDs so checkpoints remain reproducible.
MSA_PAD_IDX = 0
MSA_UNK_IDX = 1
MSA_GAP_IDX = 2
MSA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
MSA_TO_IDX = {aa: i + 3 for i, aa in enumerate(MSA_ALPHABET)}
MSA_VOCAB_SIZE = len(MSA_TO_IDX) + 3
MSA_EXTS = (".a3m", ".a2m", ".fa", ".fasta", ".faa", ".aln", ".msa")

# Backward-compatible names for old checkpoint metadata or helper code.
AA_PAD_IDX = MSA_PAD_IDX
AA_UNK_IDX = MSA_UNK_IDX
AA_ALPHABET = MSA_ALPHABET
AA_TO_IDX = MSA_TO_IDX
AA_VOCAB_SIZE = MSA_VOCAB_SIZE


def normalize_msa_sequence(value: Any, remove_a3m_insertions: bool = True) -> str:
    """Normalize one aligned MSA row.

    For A3M input, lowercase letters denote insertions relative to the query and
    are removed by default so all rows stay in the query alignment coordinate.
    Gaps are preserved as '-'. Unknown residues are mapped to X.
    """
    if value is None:
        return ""
    seq = str(value).strip()
    if not seq or seq.lower() == "nan":
        return ""
    if remove_a3m_insertions:
        seq = "".join(ch for ch in seq if not ch.islower())
    seq = seq.upper().replace(".", "-").replace("*", "")
    seq = re.sub(r"\s+", "", seq)
    seq = re.sub(r"[^A-Z\-]", "X", seq)
    return seq


def encode_msa_sequences(seqs: Sequence[str], max_depth: int, max_len: int, query_sequence: Optional[str] = None) -> torch.Tensor:
    """Encode MSA rows into a compact [N, L] LongTensor without batch padding.

    The first row should be the query sequence. If no MSA is available, the query
    sequence is encoded as a depth-1 MSA so the model can still run.
    """
    cleaned: List[str] = []
    seen = set()
    for seq in seqs:
        s = normalize_msa_sequence(seq)
        if not s:
            continue
        # De-duplicate exact rows while preserving order; large redundant MSAs slow training.
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if max_depth > 0 and len(cleaned) >= max_depth:
            break

    if not cleaned and query_sequence:
        cleaned = [normalize_sequence(query_sequence)]
    if not cleaned:
        cleaned = ["X"]

    if max_depth > 0:
        cleaned = cleaned[:max_depth]
    L = max(len(s) for s in cleaned)
    if max_len > 0:
        L = min(L, max_len)
    L = max(L, 1)

    out = torch.full((len(cleaned), L), MSA_PAD_IDX, dtype=torch.long)
    for i, s in enumerate(cleaned):
        s = s[:L]
        ids: List[int] = []
        for ch in s:
            if ch == "-":
                ids.append(MSA_GAP_IDX)
            else:
                ids.append(MSA_TO_IDX.get(ch, MSA_UNK_IDX))
        if ids:
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def split_paths(text: str) -> List[str]:
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def safe_filename_variants(sid: str) -> List[str]:
    sid = str(sid).strip()
    variants = [sid]
    # Common FASTA/header normalizations.
    if "|" in sid:
        parts = [p for p in sid.split("|") if p]
        variants.extend(parts)
    variants.append(sid.replace("/", "_"))
    variants.append(sid.replace("|", "_"))
    variants.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", sid))
    # De-duplicate while preserving order.
    out: List[str] = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def to_numpy_array(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _select_from_layer_dict(d: Dict[Any, Any], layer: int) -> Any:
    if layer in d:
        return d[layer]
    slayer = str(layer)
    if slayer in d:
        return d[slayer]
    # If requested layer absent, use the numerically largest layer if possible.
    keys = list(d.keys())
    numeric_keys = []
    for k in keys:
        try:
            numeric_keys.append((int(k), k))
        except Exception:
            pass
    if numeric_keys:
        _, best_key = sorted(numeric_keys)[-1]
        return d[best_key]
    if keys:
        return d[keys[0]]
    raise KeyError("Empty representation dictionary.")


def extract_embedding_from_object(obj: Any, esm_layer: int = 36, repr_type: str = "auto") -> np.ndarray:
    """Extract ESM embedding as [D] or [L, D].

    repr_type:
      - token: prefer token-level representations
      - mean: prefer mean_representations
      - auto: prefer token-level representations, then mean vectors
    """
    if isinstance(obj, dict):
        token_keys = ["representations", "per_tok", "per_token", "token_representations", "tokens"]
        mean_keys = ["mean_representations", "mean", "mean_repr", "embedding", "emb", "esm", "repr"]
        if repr_type == "mean":
            key_order = mean_keys + token_keys
        elif repr_type == "token":
            key_order = token_keys + mean_keys
        else:
            key_order = token_keys + mean_keys
        for key in key_order:
            if key not in obj:
                continue
            val = obj[key]
            if isinstance(val, dict):
                val = _select_from_layer_dict(val, esm_layer)
            arr = to_numpy_array(val)
            if arr.size > 0:
                return arr
        # Some files may directly map layer -> tensor.
        try:
            val = _select_from_layer_dict(obj, esm_layer)
            return to_numpy_array(val)
        except Exception:
            pass
        raise KeyError(f"Cannot find ESM embedding in dict keys: {list(obj.keys())[:20]}")
    return to_numpy_array(obj)


class ESMEmbeddingStore:
    def __init__(self, paths: Sequence[str], esm_layer: int = 36, repr_type: str = "auto", recursive: bool = False) -> None:
        self.paths = [os.path.abspath(p) for p in paths]
        self.esm_layer = int(esm_layer)
        self.repr_type = repr_type
        self.recursive = bool(recursive)
        self.file_index: Dict[str, str] = {}
        self.h5_paths: List[str] = []
        self._h5_handles: Dict[str, Any] = {}
        self._build_index()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5_handles"] = {}
        return state

    def _add_file_key(self, key: str, path: str) -> None:
        if key and key not in self.file_index:
            self.file_index[key] = path

    def _build_index(self) -> None:
        for p in self.paths:
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in H5_EXTS:
                    self.h5_paths.append(p)
                elif ext in EMBED_EXTS:
                    base = os.path.splitext(os.path.basename(p))[0]
                    self._add_file_key(base, p)
                continue
            if not os.path.isdir(p):
                print(f"[WARNING] ESM path does not exist: {p}")
                continue
            pattern = "**/*" if self.recursive else "*"
            for fp in glob.iglob(os.path.join(p, pattern), recursive=self.recursive):
                if not os.path.isfile(fp):
                    continue
                ext = os.path.splitext(fp)[1].lower()
                if ext in H5_EXTS:
                    self.h5_paths.append(fp)
                elif ext in EMBED_EXTS:
                    base = os.path.splitext(os.path.basename(fp))[0]
                    self._add_file_key(base, fp)
        print(f"[ESM] indexed embedding files: {len(self.file_index):,}; h5 files: {len(self.h5_paths):,}")

    def _candidate_file_paths(self, sid: str) -> List[str]:
        out: List[str] = []
        for v in safe_filename_variants(sid):
            if v in self.file_index:
                out.append(self.file_index[v])
        return out

    def has(self, sid: str) -> bool:
        if self._candidate_file_paths(sid):
            return True
        # H5 key check can be expensive, but is acceptable at dataset construction.
        if self.h5_paths and HAS_H5PY:
            for hp in self.h5_paths:
                try:
                    with h5py.File(hp, "r") as f:  # type: ignore[union-attr]
                        for v in safe_filename_variants(sid):
                            if v in f:
                                return True
                except Exception:
                    continue
        return False

    def _load_from_file(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            return extract_embedding_from_object(obj, self.esm_layer, self.repr_type)
        if ext == ".npy":
            return to_numpy_array(np.load(path, allow_pickle=False))
        if ext == ".npz":
            z = np.load(path, allow_pickle=True)
            preferred = ["embedding", "emb", "esm", "repr", "representations", "mean_representations"]
            for k in preferred:
                if k in z:
                    return extract_embedding_from_object(z[k], self.esm_layer, self.repr_type)
            if len(z.files) == 1:
                return extract_embedding_from_object(z[z.files[0]], self.esm_layer, self.repr_type)
            raise KeyError(f"Cannot select embedding key from npz: {path}; keys={z.files}")
        if ext in {".pkl", ".pickle"}:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return extract_embedding_from_object(obj, self.esm_layer, self.repr_type)
        raise ValueError(f"Unsupported embedding file extension: {path}")

    def _get_h5_handle(self, path: str):
        if not HAS_H5PY:
            raise ImportError("h5py is not installed, but h5/hdf5 embedding file was provided.")
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")  # type: ignore[union-attr]
        return self._h5_handles[path]

    def _load_from_h5(self, sid: str) -> Optional[np.ndarray]:
        for hp in self.h5_paths:
            try:
                f = self._get_h5_handle(hp)
                for v in safe_filename_variants(sid):
                    if v in f:
                        return to_numpy_array(f[v][()])
            except Exception:
                continue
        return None

    def load(self, sid: str) -> np.ndarray:
        fps = self._candidate_file_paths(sid)
        if fps:
            arr = self._load_from_file(fps[0])
        else:
            arr = self._load_from_h5(sid)
            if arr is None:
                raise FileNotFoundError(f"No ESM embedding found for ID: {sid}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError(f"Invalid scalar embedding for ID: {sid}")
        if arr.ndim > 2:
            # Common shape [1, L, D] or [layers, L, D]; keep first if singleton, otherwise flatten leading dims carefully.
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape(-1, arr.shape[-1])
        return arr



class MSAStore:
    """Index and load per-protein MSA files.

    Expected layout: one MSA file per protein ID, named like <ID>.a3m, <ID>.fasta,
    <safe_ID>.a3m, etc. The first sequence in the file is treated as the query.
    """

    def __init__(self, paths: Sequence[str], recursive: bool = False) -> None:
        self.paths = [os.path.abspath(p) for p in paths if p]
        self.recursive = bool(recursive)
        self.file_index: Dict[str, str] = {}
        self._build_index()

    def _add_file_key(self, key: str, path: str) -> None:
        if key and key not in self.file_index:
            self.file_index[key] = path

    def _build_index(self) -> None:
        for p in self.paths:
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in MSA_EXTS:
                    base = os.path.splitext(os.path.basename(p))[0]
                    self._add_file_key(base, p)
                continue
            if not os.path.isdir(p):
                print(f"[WARNING] MSA path does not exist: {p}")
                continue
            pattern = "**/*" if self.recursive else "*"
            for fp in glob.iglob(os.path.join(p, pattern), recursive=self.recursive):
                if not os.path.isfile(fp):
                    continue
                ext = os.path.splitext(fp)[1].lower()
                if ext in MSA_EXTS:
                    base = os.path.splitext(os.path.basename(fp))[0]
                    self._add_file_key(base, fp)
        print(f"[MSA] indexed MSA files: {len(self.file_index):,}")

    def _candidate_file_paths(self, sid: str) -> List[str]:
        out: List[str] = []
        for v in safe_filename_variants(sid):
            if v in self.file_index:
                out.append(self.file_index[v])
        return out

    def has(self, sid: str) -> bool:
        return bool(self._candidate_file_paths(sid))

    @staticmethod
    def _parse_fasta_like(path: str) -> List[str]:
        seqs: List[str] = []
        cur: List[str] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                    continue
                # Some Stockholm-like files contain comments or terminators; ignore them.
                if line.startswith("#") or line.startswith("//"):
                    continue
                # If a line looks like "name SEQUENCE", keep the sequence field.
                parts = line.split()
                if len(parts) >= 2 and not set(parts[0]).issubset(set("ACDEFGHIKLMNPQRSTVWYBXZJUOacdefghiklmnpqrstvwybxzjuo-.")):
                    cur.append(parts[-1])
                else:
                    cur.append(parts[-1] if parts else line)
        if cur:
            seqs.append("".join(cur))
        return seqs

    def load(self, sid: str) -> List[str]:
        fps = self._candidate_file_paths(sid)
        if not fps:
            raise FileNotFoundError(f"No MSA found for ID: {sid}")
        seqs = self._parse_fasta_like(fps[0])
        if not seqs:
            raise ValueError(f"MSA file is empty or could not be parsed for ID {sid}: {fps[0]}")
        return seqs


class ESMMSAECDataset(Dataset):
    """Dataset returning both ESM embeddings and encoded MSA tokens.

    Each item contains:
      - token/per-protein ESM embedding: [L_esm, D]
      - MSA token IDs: [N_msa, L_msa]
      - multi-hot EC label vector
      - sample ID
    """

    def __init__(
        self,
        id_ec_dict: Dict[str, Any],
        ec2idx: Dict[str, int],
        emb_store: ESMEmbeddingStore,
        id_seq_dict: Dict[str, str],
        msa_store: Optional[MSAStore],
        dataset_name: str,
        seq_max_len: int = 1000,
        msa_max_depth: int = 64,
        msa_max_len: int = 512,
        min_embedding_len: int = 1,
        min_msa_depth: int = 1,
        allow_msa_fallback_to_sequence: bool = True,
    ) -> None:
        self.id_ec_dict = id_ec_dict
        self.ec2idx = ec2idx
        self.num_labels = len(ec2idx)
        self.emb_store = emb_store
        self.id_seq_dict = id_seq_dict
        self.msa_store = msa_store
        self.dataset_name = dataset_name
        self.seq_max_len = int(seq_max_len)
        self.msa_max_depth = int(msa_max_depth)
        self.msa_max_len = int(msa_max_len)
        self.min_embedding_len = int(min_embedding_len)
        self.min_msa_depth = int(min_msa_depth)
        self.allow_msa_fallback_to_sequence = bool(allow_msa_fallback_to_sequence)

        self.samples: List[Tuple[str, List[str], str, bool]] = []
        self.missing_embedding: List[str] = []
        self.missing_msa_and_sequence: List[str] = []
        self.missing_msa_used_sequence_fallback: List[str] = []
        self.short_msa: List[str] = []
        self.short_embedding: List[str] = []
        self.no_covered_label: List[str] = []

        for sid_raw, ecs_raw in id_ec_dict.items():
            sid = str(sid_raw).strip()
            ecs = [ec for ec in normalize_ec_list(ecs_raw) if ec in ec2idx]
            if not ecs:
                self.no_covered_label.append(sid)
                continue
            if not self.emb_store.has(sid):
                self.missing_embedding.append(sid)
                continue

            has_msa = bool(self.msa_store is not None and self.msa_store.has(sid))
            fallback_seq = normalize_sequence(id_seq_dict.get(sid, ""))
            if not has_msa:
                if self.allow_msa_fallback_to_sequence and fallback_seq:
                    self.missing_msa_used_sequence_fallback.append(sid)
                else:
                    self.missing_msa_and_sequence.append(sid)
                    continue
            # Do not load all ESM embeddings/MSAs during construction.
            self.samples.append((sid, ecs, fallback_seq, has_msa))

        self.stats = {
            "dataset": dataset_name,
            "raw_samples": len(id_ec_dict),
            "used_samples_with_embedding_msa_or_fallback_and_covered_labels": len(self.samples),
            "dropped_no_covered_label": len(self.no_covered_label),
            "missing_embedding": len(self.missing_embedding),
            "missing_msa_and_sequence": len(self.missing_msa_and_sequence),
            "missing_msa_used_sequence_fallback": len(self.missing_msa_used_sequence_fallback),
            "short_msa_lt_min_msa_depth": len(self.short_msa),
            "short_embedding_lt_min_embedding_len": len(self.short_embedding),
            "used_fraction_of_raw": len(self.samples) / max(len(id_ec_dict), 1),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _make_label(self, ecs: List[str]) -> torch.Tensor:
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for ec in ecs:
            if ec in self.ec2idx:
                y[self.ec2idx[ec]] = 1.0
        return y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        sid, ecs, fallback_seq, has_msa = self.samples[idx]
        arr = self.emb_store.load(sid)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Embedding for {sid} must be [D] or [L,D], got shape={arr.shape}")
        if self.seq_max_len > 0 and arr.shape[0] > self.seq_max_len:
            arr = arr[: self.seq_max_len]
        if arr.shape[0] < self.min_embedding_len:
            raise ValueError(f"Embedding for {sid} length {arr.shape[0]} < min_embedding_len {self.min_embedding_len}")

        if has_msa and self.msa_store is not None:
            msa_rows = self.msa_store.load(sid)
        else:
            msa_rows = [fallback_seq]
        msa_tokens = encode_msa_sequences(msa_rows, self.msa_max_depth, self.msa_max_len, query_sequence=fallback_seq)
        if msa_tokens.shape[0] < self.min_msa_depth:
            raise ValueError(f"MSA for {sid} depth {msa_tokens.shape[0]} < min_msa_depth {self.min_msa_depth}")
        return torch.from_numpy(arr.astype(np.float32, copy=False)), msa_tokens, self._make_label(ecs), sid


# Backward-compatible aliases for helper type annotations.
ESMRawSeqECDataset = ESMMSAECDataset
ESMEmbeddingECDataset = ESMMSAECDataset


def esm_msa_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    embs, msa_tokens, labels, ids = zip(*batch)

    esm_lengths = torch.tensor([x.shape[0] for x in embs], dtype=torch.long)
    esm_dim = int(embs[0].shape[-1])
    esm_max_len = int(esm_lengths.max().item()) if len(esm_lengths) else 0
    esm_x = torch.zeros(len(embs), esm_max_len, esm_dim, dtype=torch.float32)
    esm_mask = torch.zeros(len(embs), esm_max_len, dtype=torch.bool)
    for i, e in enumerate(embs):
        L = e.shape[0]
        esm_x[i, :L] = e
        esm_mask[i, :L] = True

    msa_depths = torch.tensor([x.shape[0] for x in msa_tokens], dtype=torch.long)
    msa_lengths = torch.tensor([x.shape[1] for x in msa_tokens], dtype=torch.long)
    max_depth = int(msa_depths.max().item()) if len(msa_depths) else 0
    max_len = int(msa_lengths.max().item()) if len(msa_lengths) else 0
    msa_x = torch.full((len(msa_tokens), max_depth, max_len), MSA_PAD_IDX, dtype=torch.long)
    msa_mask = torch.zeros(len(msa_tokens), max_depth, max_len, dtype=torch.bool)
    msa_row_mask = torch.zeros(len(msa_tokens), max_depth, dtype=torch.bool)
    for i, m in enumerate(msa_tokens):
        n, L = m.shape
        msa_x[i, :n, :L] = m
        msa_mask[i, :n, :L] = m.ne(MSA_PAD_IDX)
        msa_row_mask[i, :n] = True

    return esm_x, esm_mask, msa_x, msa_mask, msa_row_mask, torch.stack(list(labels), dim=0), list(ids)


# Keep old collate function names as aliases so older launch code does not crash.
esm_rawseq_collate = esm_msa_collate
esm_collate = esm_msa_collate


def infer_embedding_dim(dataset: ESMMSAECDataset, max_tries: int = 100) -> int:
    if len(dataset) == 0:
        raise RuntimeError("Cannot infer embedding dimension from empty dataset.")
    last_err: Optional[Exception] = None
    for i in range(min(len(dataset), max_tries)):
        try:
            emb, _msa_tokens, _label, sid = dataset[i]
            if emb.ndim != 2:
                raise ValueError(f"Invalid embedding shape for {sid}: {tuple(emb.shape)}")
            return int(emb.shape[-1])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to infer embedding dimension. Last error: {last_err}")


# =============================================================================
# Original modern model blocks retained
# =============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GRN1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 15, expansion: int = 4, dropout: float = 0.1, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.grn = GRN1D(expansion * dim)
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class GatedFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.dropout(F.gelu(a) * b))


class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, drop_path: float = 0.0, ffn_expansion: int = 4) -> None:
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm is not installed. Use --model_arch convnextv2 or install mamba-ssm.")
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim, expansion=ffn_expansion, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.mamba(self.norm1(x)))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class MaskedAttentivePooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
    return (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    return torch.nan_to_num(y.max(dim=1).values, nan=0.0, neginf=0.0, posinf=0.0)


class MSAAxialBlock(nn.Module):
    """Lightweight MSA encoder block.

    Row path: Mamba mixes residues along each aligned sequence.
    Column path: Multi-head attention mixes homologous residues at each alignment column.
    This captures evolutionary covariation/context at much lower cost than a full MSA Transformer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 1,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("MSA row-wise sequence mixing requires mamba-ssm.")
        self.row_block = MambaBlock(dim, mamba_d_state, mamba_d_conv, mamba_expand, dropout, drop_path, ffn_expansion)
        self.col_norm = nn.LayerNorm(dim)
        self.col_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.col_drop = nn.Dropout(dropout)
        self.col_drop_path = DropPath(drop_path)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim, expansion=ffn_expansion, dropout=dropout)
        self.ffn_drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, msa_mask: torch.Tensor, msa_row_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L, C], msa_mask: [B, N, L]
        B, N, L, C = x.shape

        # Row-wise sequence mixing over L for every homolog row.
        row_x = x.reshape(B * N, L, C)
        row_mask = msa_mask.reshape(B * N, L)
        row_x = self.row_block(row_x, mask=row_mask)
        x = row_x.reshape(B, N, L, C)

        # Column-wise homolog attention over N for every alignment column.
        col_x = x.permute(0, 2, 1, 3).reshape(B * L, N, C)
        col_valid = msa_mask.permute(0, 2, 1).reshape(B * L, N)
        key_padding_mask = ~col_valid
        valid_col = col_valid.any(dim=1)
        # MultiheadAttention returns NaNs if every key is masked; unmask all-empty padded columns.
        if (~valid_col).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~valid_col] = False
        col_y = self.col_norm(col_x)
        attn_out, _ = self.col_attn(col_y, col_y, col_y, key_padding_mask=key_padding_mask, need_weights=False)
        col_x = col_x + self.col_drop_path(self.col_drop(attn_out))
        col_x = col_x + self.ffn_drop_path(self.ffn(self.ffn_norm(col_x)))
        col_x = col_x * col_valid.unsqueeze(-1).to(dtype=col_x.dtype)
        x = col_x.reshape(B, L, N, C).permute(0, 2, 1, 3)
        x = x * msa_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class MSAEncoder(nn.Module):
    """MSA encoder producing a pooled query-row feature.

    The first MSA row is assumed to be the query. Column attention lets homolog rows
    update the query representation before query-row attentive/mean/max pooling.
    """

    def __init__(
        self,
        msa_vocab_size: int,
        msa_pad_idx: int,
        msa_d_model: int,
        msa_max_depth: int,
        msa_max_len: int,
        msa_depth: int = 2,
        msa_col_heads: int = 4,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        use_pos_embedding: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        msa_mamba_expand: int = 1,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.msa_d_model = int(msa_d_model)
        self.msa_embedding = nn.Embedding(msa_vocab_size, msa_d_model, padding_idx=msa_pad_idx)
        self.col_pos_embedding = nn.Parameter(torch.zeros(1, 1, msa_max_len, msa_d_model)) if use_pos_embedding and msa_max_len > 0 else None
        self.row_pos_embedding = nn.Parameter(torch.zeros(1, msa_max_depth, 1, msa_d_model)) if use_pos_embedding and msa_max_depth > 0 else None
        if self.col_pos_embedding is not None:
            nn.init.trunc_normal_(self.col_pos_embedding, std=0.02)
        if self.row_pos_embedding is not None:
            nn.init.trunc_normal_(self.row_pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        dpr = torch.linspace(0, drop_path, steps=max(msa_depth, 1)).tolist()
        self.blocks = nn.ModuleList([
            MSAAxialBlock(
                msa_d_model,
                num_heads=msa_col_heads,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=msa_mamba_expand,
                dropout=dropout,
                drop_path=dpr[i],
                ffn_expansion=expansion,
            )
            for i in range(msa_depth)
        ])
        self.norm = nn.LayerNorm(msa_d_model)
        self.attn_pool = MaskedAttentivePooling(msa_d_model)

    def forward(self, msa_tokens: torch.Tensor, msa_mask: torch.Tensor, msa_row_mask: torch.Tensor) -> torch.Tensor:
        # msa_tokens: [B, N, L]
        x = self.msa_embedding(msa_tokens)
        if self.col_pos_embedding is not None:
            if x.shape[2] > self.col_pos_embedding.shape[2]:
                raise ValueError(
                    f"MSA alignment length {x.shape[2]} exceeds positional length {self.col_pos_embedding.shape[2]}. "
                    "Increase --msa_max_len or use --no_pos_embedding."
                )
            x = x + self.col_pos_embedding[:, :, : x.shape[2], :]
        if self.row_pos_embedding is not None:
            if x.shape[1] > self.row_pos_embedding.shape[1]:
                raise ValueError(
                    f"MSA depth {x.shape[1]} exceeds row positional depth {self.row_pos_embedding.shape[1]}. "
                    "Increase --msa_max_depth or use --no_pos_embedding."
                )
            x = x + self.row_pos_embedding[:, : x.shape[1], :, :]
        x = self.dropout(x)
        x = x * msa_mask.unsqueeze(-1).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, msa_mask=msa_mask, msa_row_mask=msa_row_mask)
        x = self.norm(x)
        x = x * msa_mask.unsqueeze(-1).to(dtype=x.dtype)

        # Pool the query row after it has attended to homolog columns.
        query_x = x[:, 0, :, :]
        query_mask = msa_mask[:, 0, :]
        # Very defensive fallback; should not happen because encode_msa_sequences always creates a non-empty first row.
        empty = ~query_mask.any(dim=1)
        if empty.any():
            query_mask = query_mask.clone()
            query_mask[empty, 0] = True
        return torch.cat([
            self.attn_pool(query_x, query_mask),
            masked_mean_pool(query_x, query_mask),
            masked_max_pool(query_x, query_mask),
        ], dim=-1)


class MSAESMGatedClassifier(nn.Module):
    """ESM2 pooled/projection branch + MSA-Mamba encoder branch with learnable gated fusion.

    This version follows the intended architecture:
      1. ESM branch: ESM2 token/per-protein embeddings -> LayerNorm + Linear projection -> pooling only.
         No Mamba / ConvNeXt / Hybrid backbone is applied to ESM2 embeddings.
      2. MSA branch: aligned homolog sequences -> axial MSA encoder with row-wise Mamba + column-wise attention.
      3. Fusion: fused = gate * esm_feat + (1 - gate) * msa_feat.

    The arguments model_arch, depth, conv_kernel, and mamba_expand are accepted for backward-compatible
    command lines, but they do not build an ESM-side sequence backbone in this class.
    """

    def __init__(
        self,
        num_labels: int,
        input_dim: int,
        seq_max_len: int,
        msa_max_depth: int,
        msa_max_len: int,
        msa_vocab_size: int = MSA_VOCAB_SIZE,
        msa_pad_idx: int = MSA_PAD_IDX,
        model_arch: str = "unused_esm_pooled",
        d_model: int = 256,
        msa_d_model: int = 128,
        depth: int = 0,
        msa_depth: int = 2,
        msa_col_heads: int = 4,
        conv_kernel: int = 15,
        expansion: int = 4,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        use_pos_embedding: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        msa_mamba_expand: int = 1,
        gate_init_bias: float = 2.0,
        classifier_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("This model uses row-wise Mamba inside the MSA encoder, so mamba-ssm is required.")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.msa_d_model = int(msa_d_model)

        # ESM2 branch: lightweight projection + masked pooling only.
        # No ESM-side Mamba/ConvNeXt/Hybrid blocks are constructed here.
        self.esm_input_norm = nn.LayerNorm(input_dim)
        self.esm_input_proj = nn.Linear(input_dim, d_model)
        self.esm_dropout = nn.Dropout(dropout)
        self.esm_norm = nn.LayerNorm(d_model)
        self.esm_attn_pool = MaskedAttentivePooling(d_model)

        # MSA branch: this is the only branch using Mamba.
        self.msa_encoder = MSAEncoder(
            msa_vocab_size=msa_vocab_size,
            msa_pad_idx=msa_pad_idx,
            msa_d_model=msa_d_model,
            msa_max_depth=msa_max_depth,
            msa_max_len=msa_max_len,
            msa_depth=msa_depth,
            msa_col_heads=msa_col_heads,
            dropout=dropout,
            drop_path=drop_path,
            use_pos_embedding=use_pos_embedding,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            msa_mamba_expand=msa_mamba_expand,
            expansion=expansion,
        )

        pooled_dim = d_model * 3
        msa_pooled_dim = msa_d_model * 3
        self.msa_to_esm = nn.Identity() if msa_pooled_dim == pooled_dim else nn.Sequential(
            nn.LayerNorm(msa_pooled_dim),
            nn.Linear(msa_pooled_dim, pooled_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_in_dim = pooled_dim * 4
        self.gate_norm = nn.LayerNorm(gate_in_dim)
        self.gate_linear = nn.Linear(gate_in_dim, pooled_dim)
        nn.init.constant_(self.gate_linear.bias, float(gate_init_bias))
        self.fusion_norm = nn.LayerNorm(pooled_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, max(classifier_hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(classifier_hidden_dim // 2, 1), num_labels),
        )

    def _pool_branch(self, x: torch.Tensor, mask: torch.Tensor, attn_pool: MaskedAttentivePooling) -> torch.Tensor:
        return torch.cat([attn_pool(x, mask), masked_mean_pool(x, mask), masked_max_pool(x, mask)], dim=-1)

    def encode_esm(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode ESM2 embeddings without any sequence backbone.

        emb can be token-level [B, L, D] or mean-level [B, 1, D]. We only apply a
        per-token projection and masked pooling, so ESM2 remains the pretrained feature source
        rather than being further modeled by Mamba.
        """
        x = self.esm_input_proj(self.esm_input_norm(emb))
        x = self.esm_dropout(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.esm_norm(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return self._pool_branch(x, mask, self.esm_attn_pool)

    def forward(
        self,
        emb: torch.Tensor,
        esm_mask: torch.Tensor,
        msa_tokens: torch.Tensor,
        msa_mask: torch.Tensor,
        msa_row_mask: torch.Tensor,
    ) -> torch.Tensor:
        esm_feat = self.encode_esm(emb, esm_mask)
        msa_feat = self.msa_to_esm(self.msa_encoder(msa_tokens, msa_mask, msa_row_mask))
        gate_input = torch.cat([esm_feat, msa_feat, torch.abs(esm_feat - msa_feat), esm_feat * msa_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(self.gate_norm(gate_input)))
        fused = gate * esm_feat + (1.0 - gate) * msa_feat
        fused = self.fusion_norm(fused)
        return self.classifier(fused)


# Backward-compatible names; these classes now expect MSA tensors in forward(). ESM side is pooled/projection only.
RawSeqESMGatedMambaClassifier = MSAESMGatedClassifier
ESM2MambaClassifier = MSAESMGatedClassifier


# =============================================================================
# Losses
# =============================================================================

class AsymmetricLossWithLogits(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8, disable_torch_grad_focal_loss: bool = True) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = torch.pow(1.0 - pt, gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= weight
        return -loss.mean()


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone().float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_weight = (1.0 - pt).clamp(min=0.0, max=1.0).pow(self.gamma)
        return (bce * focal_weight).mean()


def build_pos_weight(train_support: np.ndarray, n_train_samples: int, device: torch.device, mode: str, max_pos_weight: float) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    support = np.asarray(train_support, dtype=np.float32)
    pos = np.maximum(support, 1.0)
    neg = np.maximum(float(n_train_samples) - support, 1.0)
    w = neg / pos
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode == "log":
        w = np.log1p(w)
    elif mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown pos_weight_mode: {mode}")
    w = np.clip(w, 1.0, max_pos_weight).astype(np.float32)
    return torch.tensor(w, device=device, dtype=torch.float32)


def build_criterion(args: argparse.Namespace, pos_weight: Optional[torch.Tensor]) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossWithLogits(args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip)
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.loss == "focal_bce":
        return FocalBCEWithLogitsLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    raise ValueError("Unknown loss")


# =============================================================================
# Train / predict
# =============================================================================

def autocast_context(device: torch.device):
    return torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()


def make_scaler(device: torch.device):
    return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int, lr: float, weight_decay: float, grad_clip: float, criterion: nn.Module, warmup_epochs: int, min_lr_ratio: float) -> nn.Module:
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, len(train_loader) * epochs, len(train_loader) * warmup_epochs, min_lr_ratio)
    scaler = make_scaler(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Train MSA+ESM-Gated epoch {epoch}/{epochs}", leave=False)
        for emb, esm_mask, msa_tokens, msa_mask, msa_row_mask, labels, _ids in pbar:
            emb = emb.to(device, non_blocking=True)
            esm_mask = esm_mask.to(device, non_blocking=True)
            msa_tokens = msa_tokens.to(device, non_blocking=True)
            msa_mask = msa_mask.to(device, non_blocking=True)
            msa_row_mask = msa_row_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(emb, esm_mask, msa_tokens, msa_mask, msa_row_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        print(f"[MSA+ESM-Gated] epoch {epoch:03d}/{epochs} loss={total_loss / max(n_batches, 1):.6f} lr={optimizer.param_groups[0]['lr']:.3e}")
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    ids_all: List[str] = []
    for emb, esm_mask, msa_tokens, msa_mask, msa_row_mask, labels, ids in tqdm(loader, desc="Predict MSA+ESM-Gated logits", leave=False):
        emb = emb.to(device, non_blocking=True)
        esm_mask = esm_mask.to(device, non_blocking=True)
        msa_tokens = msa_tokens.to(device, non_blocking=True)
        msa_mask = msa_mask.to(device, non_blocking=True)
        msa_row_mask = msa_row_mask.to(device, non_blocking=True)
        with autocast_context(device):
            logits = model(emb, esm_mask, msa_tokens, msa_mask, msa_row_mask)
        logits_list.append(logits.float().cpu().numpy())
        labels_list.append(labels.float().cpu().numpy())
        ids_all.extend(ids)
    if not logits_list:
        return np.empty((0, 0)), np.empty((0, 0)), []
    return np.vstack(logits_list), np.vstack(labels_list), ids_all


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Metrics
# =============================================================================

def safe_prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Tuple[float, float, float]:
    try:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        return float(p), float(r), float(f1)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray, average: str) -> float:
    try:
        return float(average_precision_score(y_true, y_prob, average=average))
    except Exception:
        return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, average: str) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except Exception:
        return float("nan")


def compute_auprc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    yt = y_true.astype(np.int32)
    supported = np.where(yt.sum(axis=0) > 0)[0]
    out = {"n_test_supported_labels": int(len(supported)), "auprc_micro_supported": float("nan"), "auprc_macro_supported": float("nan"), "auprc_weighted_supported": float("nan"), "roc_auc_weighted_supported": float("nan")}
    if len(supported) > 0:
        ys = yt[:, supported]
        ps = y_prob[:, supported]
        out["auprc_micro_supported"] = safe_auprc(ys, ps, "micro")
        out["auprc_macro_supported"] = safe_auprc(ys, ps, "macro")
        out["auprc_weighted_supported"] = safe_auprc(ys, ps, "weighted")
        out["roc_auc_weighted_supported"] = safe_roc_auc(ys, ps, "weighted")
    return out


def compute_prf_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, train_support: np.ndarray, rare_train_support_max: int) -> Dict[str, Any]:
    yt = y_true.astype(np.int32)
    yp = (y_prob >= threshold).astype(np.int32)
    micro_p, micro_r, micro_f1 = safe_prf(yt, yp, "micro")
    macro_p, macro_r, macro_f1 = safe_prf(yt, yp, "macro")
    weighted_p, weighted_r, weighted_f1 = safe_prf(yt, yp, "weighted")
    samples_p, samples_r, samples_f1 = safe_prf(yt, yp, "samples")
    
    try:
        exact_match_acc = float(accuracy_score(yt, yp))
    except Exception:
        exact_match_acc = float("nan")
        
    present_cols = np.where(yt.sum(axis=0) > 0)[0]
    if len(present_cols) > 0:
        present_macro_p, present_macro_r, present_macro_f1 = safe_prf(yt[:, present_cols], yp[:, present_cols], "macro")
    else:
        present_macro_p = present_macro_r = present_macro_f1 = float("nan")
    rare_cols = np.where((train_support > 0) & (train_support <= rare_train_support_max) & (yt.sum(axis=0) > 0))[0]
    if len(rare_cols) > 0:
        rare_p, rare_r, rare_f1 = safe_prf(yt[:, rare_cols], yp[:, rare_cols], "macro")
    else:
        rare_p = rare_r = rare_f1 = float("nan")
    true_counts = yt.sum(axis=1)
    pred_counts = yp.sum(axis=1)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return {
        "threshold": float(threshold),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1_score": micro_f1,
        "macro_precision_all_labels": macro_p,
        "macro_recall_all_labels": macro_r,
        "macro_f1_score_all_labels": macro_f1,
        "macro_precision_present_labels": present_macro_p,
        "macro_recall_present_labels": present_macro_r,
        "macro_f1_score_present_labels": present_macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1_score": weighted_f1,
        "exact_match_accuracy": exact_match_acc,
        "samples_precision": samples_p,
        "samples_recall": samples_r,
        "samples_f1_score": samples_f1,
        "rare_macro_precision": rare_p,
        "rare_macro_recall": rare_r,
        "rare_macro_f1_score": rare_f1,
        "n_present_labels": int(len(present_cols)),
        "n_rare_test_supported_labels": int(len(rare_cols)),
        "avg_true_labels": float(np.mean(true_counts)) if len(true_counts) else 0.0,
        "avg_pred_labels": float(np.mean(pred_counts)) if len(pred_counts) else 0.0,
        "median_pred_labels": float(np.median(pred_counts)) if len(pred_counts) else 0.0,
        "zero_pred_rate": float(np.mean(pred_counts == 0)) if len(pred_counts) else 0.0,
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "fp_per_sample": float(fp / max(yt.shape[0], 1)),
        "fn_per_sample": float(fn / max(yt.shape[0], 1)),
    }


def compute_metrics_for_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float], train_support: np.ndarray, rare_train_support_max: int) -> List[Dict[str, Any]]:
    auprc = compute_auprc_metrics(y_true, y_prob)
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        row = compute_prf_metrics(y_true, y_prob, thr, train_support, rare_train_support_max)
        row.update(auprc)
        rows.append(row)
    return rows


def compute_per_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, labels: List[str], train_support: np.ndarray, dataset: str, model: str) -> List[Dict[str, Any]]:
    yt = y_true.astype(np.int32)
    yp = (y_prob >= threshold).astype(np.int32)
    rows: List[Dict[str, Any]] = []
    for j, ec in enumerate(labels):
        yj = yt[:, j]
        pj = yp[:, j]
        tp = int(np.logical_and(yj == 1, pj == 1).sum())
        fp = int(np.logical_and(yj == 0, pj == 1).sum())
        fn = int(np.logical_and(yj == 1, pj == 0).sum())
        tn = int(np.logical_and(yj == 0, pj == 0).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"dataset": dataset, "model": model, "label": ec, "threshold": float(threshold), "train_support": int(train_support[j]), "test_support": int(yj.sum()), "pred_support": int(pj.sum()), "precision": float(precision), "recall": float(recall), "f1_score": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return rows


def add_context(row: Dict[str, Any], dataset: str, model: str) -> Dict[str, Any]:
    out = {"dataset": dataset, "model": model}
    out.update(row)
    return out


def select_threshold_rows(rows: List[Dict[str, Any]], selected_threshold: float) -> List[Dict[str, Any]]:
    return [r for r in rows if abs(float(r["threshold"]) - float(selected_threshold)) < 1e-12]


def best_threshold_diagnostics(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, model), group_rows in groups.items():
        for metric in metrics:
            valid = [r for r in group_rows if metric in r and r.get(metric) == r.get(metric)]
            if not valid:
                continue
            best = max(valid, key=lambda r: float(r[metric]))
            out.append({
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "best_threshold_on_test_for_diagnosis_only": best["threshold"],
                "best_value": best[metric],
                "micro_precision": best.get("micro_precision"),
                "micro_recall": best.get("micro_recall"),
                "micro_f1_score": best.get("micro_f1_score"),
                "weighted_precision": best.get("weighted_precision"),
                "weighted_recall": best.get("weighted_recall"),
                "weighted_f1_score": best.get("weighted_f1_score"),
                "macro_precision_present_labels": best.get("macro_precision_present_labels"),
                "macro_recall_present_labels": best.get("macro_recall_present_labels"),
                "macro_f1_score_present_labels": best.get("macro_f1_score_present_labels"),
                "rare_macro_precision": best.get("rare_macro_precision"),
                "rare_macro_recall": best.get("rare_macro_recall"),
                "rare_macro_f1_score": best.get("rare_macro_f1_score"),
                "avg_pred_labels": best.get("avg_pred_labels"),
                "zero_pred_rate": best.get("zero_pred_rate"),
            })
    return out


def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    if not summary_rows:
        print("[WARNING] No selected-threshold summary rows.")
        return
    keep_cols = ["dataset", "model", "threshold", "micro_precision", "micro_recall", "micro_f1_score", "weighted_precision", "weighted_recall", "weighted_f1_score", "exact_match_accuracy", "roc_auc_weighted_supported", "macro_precision_present_labels", "macro_recall_present_labels", "macro_f1_score_present_labels", "rare_macro_precision", "rare_macro_recall", "rare_macro_f1_score", "avg_pred_labels", "zero_pred_rate"]
    df = pd.DataFrame(summary_rows)
    keep_cols = [c for c in keep_cols if c in df.columns]
    view = df[keep_cols].copy()
    print("\nSelected-threshold summary:")
    print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ESM2 pooled embedding + MSA-Mamba gated fusion EC multi-label training/evaluation; train on split100 and evaluate on NEW/PRICE only.")
    parser.add_argument("--root", type=str, default="/nfs/hb236/dhy/app")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--new_csv", type=str, default=None)
    parser.add_argument("--price_csv", type=str, default=None)
    parser.add_argument("--esm_dir", type=str, required=True, help="Comma-separated ESM embedding directories/files.")
    parser.add_argument("--sequence_col", type=str, default=None, help="Optional query sequence column used only as a fallback when an MSA file is missing. If omitted, common names such as Sequence/sequence/seq are inferred when needed.")
    parser.add_argument("--msa_dir", type=str, default="", help="Comma-separated MSA directories/files. Expected one .a3m/.fasta/.fa file per protein ID. If empty, the model falls back to single-sequence MSA from --sequence_col.")
    parser.add_argument("--recursive_msa_search", action="store_true")
    parser.add_argument("--esm_layer", type=int, default=36)
    parser.add_argument("--esm_repr", type=str, default="auto", choices=["auto", "token", "mean"], help="Prefer token-level or mean-pooled ESM representation.")
    parser.add_argument("--recursive_esm_search", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=0, help="0 means infer from first available embedding. ESM2-3B is usually 2560.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--seq_max_len", type=int, default=1000, help="Maximum token-level ESM embedding length. Mean-pooled vectors become length 1.")
    parser.add_argument("--msa_max_len", type=int, default=512, help="Maximum MSA alignment length used by the MSA encoder.")
    parser.add_argument("--msa_max_depth", type=int, default=64, help="Maximum number of MSA rows/homologs used by the MSA encoder.")
    parser.add_argument("--min_msa_depth", type=int, default=1)
    parser.add_argument("--no_msa_fallback_to_sequence", action="store_true", help="If set, samples without an MSA file are dropped instead of using the raw query sequence as a depth-1 MSA.")
    parser.add_argument("--min_embedding_len", type=int, default=1)
    parser.add_argument("--model_arch", type=str, default="esm_pooled", help="Deprecated/ignored: ESM branch no longer uses Mamba/ConvNeXt/Hybrid; Mamba is used only inside the MSA encoder.")
    parser.add_argument("--d_model", type=int, default=256, help="Per-token projection dimension for the lightweight pooled ESM branch.")
    parser.add_argument("--msa_d_model", type=int, default=128, help="Hidden dimension for the MSA encoder branch. Keep this smaller than --d_model for speed.")
    parser.add_argument("--depth", type=int, default=0, help="Deprecated/ignored: ESM branch is projection+pooling only. Kept for command-line compatibility.")
    parser.add_argument("--msa_depth", type=int, default=2, help="Depth of the axial MSA encoder. Recommended 1-2 for speed.")
    parser.add_argument("--msa_col_heads", type=int, default=4, help="Attention heads for column-wise homolog attention in the MSA encoder.")
    parser.add_argument("--conv_kernel", type=int, default=15)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--classifier_hidden_dim", type=int, default=512)
    parser.add_argument("--no_pos_embedding", action="store_true")
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--msa_mamba_expand", type=int, default=1, help="Mamba expand factor for row-wise MSA branch. Recommended 1 for speed.")
    parser.add_argument("--gate_init_bias", type=float, default=2.0, help="Initial fusion gate bias. Positive values make fusion prefer ESM at the start; 2.0 means about 88%% ESM.")
    parser.add_argument("--loss", type=str, default="focal_bce", choices=["asl", "bce", "weighted_bce", "focal_bce"])
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--pos_weight_mode", type=str, default="log", choices=["none", "sqrt", "log", "raw"])
    parser.add_argument("--pos_weight_max", type=float, default=8.0)
    parser.add_argument("--thresholds", type=str, default="0.005,0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--selected_threshold", type=float, default=0.07)
    parser.add_argument("--rare_train_support_max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval_datasets", type=str, default="NEW,PRICE")
    parser.add_argument("--out_dir", type=str, default="/nfs/hb236/dhy/app-copy/analysis_results_esm2_3b_mamba_split100_focal")
    parser.add_argument("--save_checkpoint", action="store_true")
    args = parser.parse_args()

    if not HAS_MAMBA:
        raise ImportError("This gated model uses an MSA encoder with row-wise Mamba mixing, so mamba-ssm is required. Install it with `pip install mamba-ssm`.")

    set_seed(args.seed, deterministic=args.deterministic)
    ensure_dir(args.out_dir)
    thresholds = parse_thresholds(args.thresholds)
    if args.selected_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [float(args.selected_threshold)]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"Using device: {device}")
    print(f"Mamba available: {HAS_MAMBA}")
    print(f"h5py available: {HAS_H5PY}")

    train_csv = args.train_csv or get_full_path(args.root, "split100")
    new_csv = args.new_csv or get_full_path(args.root, "new")
    price_csv = args.price_csv or get_full_path(args.root, "price")
    print("\nCSV paths:")
    print(f"  train(split100): {train_csv}")
    print(f"  new           : {new_csv}")
    print(f"  price         : {price_csv}")

    eval_names = [x.strip().upper() for x in args.eval_datasets.split(",") if x.strip()]
    bad = [x for x in eval_names if x not in {"NEW", "PRICE"}]
    if bad:
        raise ValueError(f"Unknown eval datasets: {bad}. This split100 script supports evaluation on NEW and/or PRICE only; HARD data are intentionally not used.")
    if not eval_names:
        raise ValueError("--eval_datasets cannot be empty. Use NEW, PRICE, or NEW,PRICE.")
    print(f"Eval datasets: {eval_names}")

    esm_paths = split_paths(args.esm_dir)
    if not esm_paths:
        raise ValueError("--esm_dir is required and cannot be empty.")
    emb_store = ESMEmbeddingStore(esm_paths, esm_layer=args.esm_layer, repr_type=args.esm_repr, recursive=args.recursive_esm_search)
    msa_paths = split_paths(args.msa_dir)
    msa_store = MSAStore(msa_paths, recursive=args.recursive_msa_search) if msa_paths else None
    if msa_store is None:
        print("[MSA] --msa_dir not provided; using query sequence as depth-1 MSA fallback for all samples.")

    id_ec_train, ec_counter = get_ec_id_dict(train_csv)
    id_seq_train = get_sequence_dict(train_csv, args.sequence_col)
    all_ec = sorted(list(ec_counter.keys()))
    ec2idx = {ec: i for i, ec in enumerate(all_ec)}
    num_labels = len(ec2idx)
    train_support_raw = compute_label_support(id_ec_train, ec2idx)
    print(f"\nTrain raw samples with EC labels: {len(id_ec_train)}")
    print(f"Train samples with fallback query sequences: {len(id_seq_train)}")
    print(f"Train EC label vocabulary size: {num_labels}")
    print(f"Raw train label support: min={train_support_raw.min() if len(train_support_raw) else 0}, median={np.median(train_support_raw) if len(train_support_raw) else 0:.1f}, max={train_support_raw.max() if len(train_support_raw) else 0}")

    eval_path_map = {"NEW": new_csv, "PRICE": price_csv}
    id_dicts: Dict[str, Dict[str, Any]] = {"TRAIN": id_ec_train}
    id_seq_dicts: Dict[str, Dict[str, str]] = {"TRAIN": id_seq_train}
    for name in eval_names:
        id_dicts[name] = get_ec_id_dict(eval_path_map[name])[0]
        id_seq_dicts[name] = get_sequence_dict(eval_path_map[name], args.sequence_col)

    datasets: Dict[str, ESMMSAECDataset] = {}
    allow_msa_fallback = not args.no_msa_fallback_to_sequence
    datasets["TRAIN"] = ESMMSAECDataset(
        id_ec_train, ec2idx, emb_store, id_seq_train, msa_store, "TRAIN",
        args.seq_max_len, args.msa_max_depth, args.msa_max_len,
        args.min_embedding_len, args.min_msa_depth, allow_msa_fallback,
    )
    for name in eval_names:
        datasets[name] = ESMMSAECDataset(
            id_dicts[name], ec2idx, emb_store, id_seq_dicts[name], msa_store, name,
            args.seq_max_len, args.msa_max_depth, args.msa_max_len,
            args.min_embedding_len, args.min_msa_depth, allow_msa_fallback,
        )

    dataset_info_rows: List[Dict[str, Any]] = []
    coverage_report: Dict[str, Any] = {}
    for name, ds in datasets.items():
        info = dict(ds.stats)
        coverage = compute_label_coverage(id_dicts[name], all_ec)
        coverage_report[name] = coverage
        info.update({
            "unique_label_coverage": coverage["unique_label_coverage"],
            "assignment_coverage": coverage["assignment_coverage"],
            "mean_sample_label_coverage": coverage["mean_sample_label_coverage"],
            "seq_max_len": args.seq_max_len,
            "msa_max_depth": args.msa_max_depth,
            "msa_max_len": args.msa_max_len,
            "min_embedding_len": args.min_embedding_len,
            "min_msa_depth": args.min_msa_depth,
            "msa_fallback_to_sequence": allow_msa_fallback,
            "esm_repr": args.esm_repr,
            "esm_layer": args.esm_layer,
        })
        dataset_info_rows.append(info)
        print(f"[{name}] {json.dumps(info, ensure_ascii=False)}")
    write_csv(os.path.join(args.out_dir, "dataset_info.csv"), dataset_info_rows)
    with open(os.path.join(args.out_dir, "label_coverage_report.json"), "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=False)

    train_dataset = datasets["TRAIN"]
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples with ESM embedding, MSA/fallback sequence, and covered labels. Check --esm_dir, --sequence_col, ID consistency, and EC labels.")

    input_dim = int(args.embedding_dim) if int(args.embedding_dim) > 0 else infer_embedding_dim(train_dataset)
    print(f"\nInferred/input ESM embedding dimension: {input_dim}")
    if input_dim != 2560:
        print(f"[WARNING] ESM2-3B embedding dimension is usually 2560, but detected/input dimension is {input_dim}.")

    id_ec_train_used = {sid: ecs for sid, ecs, _fallback_seq, _has_msa in train_dataset.samples}
    train_support = compute_label_support(id_ec_train_used, ec2idx)
    print(f"Used train label support: min={train_support.min() if len(train_support) else 0}, median={np.median(train_support) if len(train_support) else 0:.1f}, max={train_support.max() if len(train_support) else 0}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=esm_msa_collate,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loaders: Dict[str, DataLoader] = {}
    skipped_empty: List[str] = []
    for name, ds in datasets.items():
        if name == "TRAIN":
            continue
        if len(ds) == 0:
            skipped_empty.append(name)
            continue
        test_loaders[name] = DataLoader(ds, shuffle=False, **loader_kwargs)
    if skipped_empty:
        print(f"\n[WARNING] Empty evaluation datasets skipped: {skipped_empty}")
    if not test_loaders:
        raise RuntimeError("No non-empty evaluation datasets. Cannot evaluate.")

    pos_weight = build_pos_weight(train_support, len(train_dataset), device, args.pos_weight_mode, args.pos_weight_max)
    if pos_weight is not None:
        print(f"Prepared pos_weight: mode={args.pos_weight_mode}, max={args.pos_weight_max}, min/median/max={pos_weight.min().item():.3f}/{pos_weight.median().item():.3f}/{pos_weight.max().item():.3f}")
    criterion = build_criterion(args, pos_weight)
    print(f"Using loss: {args.loss}")
    print("Architecture: ESM2 branch = LayerNorm + Linear + masked pooling only; MSA branch = axial encoder with row-wise Mamba + column attention.")

    model_name = f"MSAMamba_ESM2Pooled_Gated_Split100_{args.loss}"
    print(f"\n>>> Training {model_name}")
    model = MSAESMGatedClassifier(
        num_labels=num_labels,
        input_dim=input_dim,
        seq_max_len=args.seq_max_len,
        msa_max_depth=args.msa_max_depth,
        msa_max_len=args.msa_max_len,
        msa_vocab_size=MSA_VOCAB_SIZE,
        msa_pad_idx=MSA_PAD_IDX,
        model_arch=args.model_arch,
        d_model=args.d_model,
        msa_d_model=args.msa_d_model,
        depth=args.depth,
        msa_depth=args.msa_depth,
        msa_col_heads=args.msa_col_heads,
        conv_kernel=args.conv_kernel,
        expansion=args.expansion,
        dropout=args.dropout,
        drop_path=args.drop_path,
        use_pos_embedding=not args.no_pos_embedding,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        msa_mamba_expand=args.msa_mamba_expand,
        gate_init_bias=args.gate_init_bias,
        classifier_hidden_dim=args.classifier_hidden_dim,
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model = train_model(model, train_loader, device, args.epochs, args.lr, args.weight_decay, args.grad_clip, criterion, args.warmup_epochs, args.min_lr_ratio)

    if args.save_checkpoint:
        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        ensure_dir(ckpt_dir)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "ec_labels": all_ec, "input_dim": input_dim, "msa_alphabet": MSA_ALPHABET}, os.path.join(ckpt_dir, "msa_encoder_esm2_3b_gated_split100_model.pt"))

    all_metric_rows: List[Dict[str, Any]] = []
    all_per_label_rows: List[Dict[str, Any]] = []
    for dataset_name, loader in test_loaders.items():
        print(f"\n>>> Evaluating {model_name} on {dataset_name}")
        logits, y_true, ids = predict_logits(model, loader, device)
        y_prob = sigmoid_np(logits)
        metric_rows = compute_metrics_for_thresholds(y_true, y_prob, thresholds, train_support, args.rare_train_support_max)
        all_metric_rows.extend([add_context(r, dataset_name, model_name) for r in metric_rows])
        all_per_label_rows.extend(compute_per_label_metrics(y_true, y_prob, args.selected_threshold, all_ec, train_support, dataset_name, model_name))

    metrics_path = os.path.join(args.out_dir, "metrics_threshold_sweep.csv")
    write_csv(metrics_path, all_metric_rows)
    summary_rows = select_threshold_rows(all_metric_rows, args.selected_threshold)
    summary_path = os.path.join(args.out_dir, "summary_at_selected_threshold.csv")
    write_csv(summary_path, summary_rows)
    best_metrics = ["micro_precision", "micro_recall", "micro_f1_score", "weighted_precision", "weighted_recall", "weighted_f1_score", "exact_match_accuracy", "roc_auc_weighted_supported", "macro_precision_present_labels", "macro_recall_present_labels", "macro_f1_score_present_labels", "rare_macro_precision", "rare_macro_recall", "rare_macro_f1_score", "auprc_micro_supported", "auprc_macro_supported", "auprc_weighted_supported"]
    best_path = os.path.join(args.out_dir, "best_threshold_on_test_for_diagnosis_only.csv")
    write_csv(best_path, best_threshold_diagnostics(all_metric_rows, best_metrics))
    per_label_path = os.path.join(args.out_dir, "per_label_metrics_at_selected_threshold.csv")
    write_csv(per_label_path, all_per_label_rows)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print_summary_table(summary_rows)
    print("\nDone.")
    print(f"Results saved to: {args.out_dir}")
    print(f"Dataset info:              {os.path.join(args.out_dir, 'dataset_info.csv')}")
    print(f"Label coverage:            {os.path.join(args.out_dir, 'label_coverage_report.json')}")
    print(f"Threshold sweep metrics:   {metrics_path}")
    print(f"Selected-threshold summary:{summary_path}")
    print(f"Best-threshold diagnostics:{best_path}")
    print(f"Per-label metrics:         {per_label_path}")
    print("\nImportant: best_threshold_on_test_for_diagnosis_only.csv is diagnostic only.")
    print("For final reporting, select thresholds on an inner calibration split, not on test sets.")

    # --- CLEAN-style ICML Plot ---
    try:
        plot_icml_style_metrics(summary_rows, args.out_dir)
    except Exception as e:
        print(f"[WARNING] Failed to generate ICML style plot: {e}")

def plot_icml_style_metrics(summary_rows: List[Dict[str, Any]], out_dir: str) -> None:
    """Generate an ICML-style bar plot for the key CLEAN evaluation metrics across datasets."""
    # Set ICML style parameters
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    
    datasets = []
    w_prec = []
    w_rec = []
    w_f1 = []
    acc = []
    w_roc_auc = []
    
    for row in summary_rows:
        datasets.append(row["dataset"])
        w_prec.append(row.get("weighted_precision", 0.0) or 0.0)
        w_rec.append(row.get("weighted_recall", 0.0) or 0.0)
        w_f1.append(row.get("weighted_f1_score", 0.0) or 0.0)
        acc.append(row.get("exact_match_accuracy", 0.0) or 0.0)
        w_roc_auc.append(row.get("roc_auc_weighted_supported", 0.0) or 0.0)
        
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - 2*width, w_prec, width, label='Weighted Precision', color='#4C72B0')
    rects2 = ax.bar(x - width, w_rec, width, label='Weighted Recall', color='#55A868')
    rects3 = ax.bar(x, w_f1, width, label='Weighted F1', color='#C44E52')
    rects4 = ax.bar(x + width, acc, width, label='Exact Match Acc', color='#8172B2')
    rects5 = ax.bar(x + 2*width, w_roc_auc, width, label='Weighted ROC-AUC', color='#CCB974')
    
    ax.set_ylabel('Score')
    ax.set_title('CLEAN Evaluation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), ncol=2)
    
    ax.set_ylim(0, 1.1)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, rotation=90)
                
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "clean_metrics_comparison.pdf")
    png_path = os.path.join(out_dir, "clean_metrics_comparison.png")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    plt.close()
    print(f"Saved ICML-style plots to {plot_path} and {png_path}")


if __name__ == "__main__":
    main()
