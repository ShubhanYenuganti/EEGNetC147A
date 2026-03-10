"""
BCIDataLoader — PyTorch DataLoader for BCI Competition IV Dataset 2a.

## Overview

This module wraps the preprocessed .npy files and split config (configs/data_splits.json)
into a standard PyTorch DataLoader. Two evaluation modes are supported:

  - **subject_dependent**: Train, validate, and test within a single subject using
    4-fold blockwise CV (first 4 MI runs, runs 3–6). All data from T session.

  - **loso** (Leave-One-Subject-Out): 90 folds (9 subjects × 10 random reps).
    Train on 5 subjects (T), validate on 3 subjects (T), test on 1 subject (E).

## How It Works

1. `BCIDataset` (internal) loads .npy arrays and selects the right indices/subjects
   depending on the mode. It acts as a standard `torch.utils.data.Dataset`.

2. `BCIDataLoader` builds a `BCIDataset`, wraps it in `torch.utils.data.DataLoader`,
   and exposes the same DataLoader interface (iteration, `len`, etc.).

3. The split config (JSON) drives which trial indices or subjects belong to each split.
   This keeps the data selection logic separate from the model code.

## API

### Subject-dependent

    from src.data.dataloader import BCIDataLoader

    loader = BCIDataLoader(
        mode='subject_dependent',
        subject='A01',          # which subject (A01–A09)
        fold=0,                 # 0–3; blockwise CV fold
        split='train',          # 'train', 'val', or 'test'
        batch_size=32,
        shuffle=True,
    )

    for X_batch, y_batch in loader:
        # X_batch: (batch_size, 22, 256)  float32
        # y_batch: (batch_size,)            long  (0-indexed: 0=Left, 1=Right, 2=Feet, 3=Tongue)
        ...

### LOSO

    loader = BCIDataLoader(
        mode='loso',
        fold_key='A01_rep0',    # '{subject}_rep{0-9}'; or integer 0–89
        split='train',          # 'train', 'val', or 'test'
        batch_size=32,
        shuffle=True,
    )

### Optional arguments

    BCIDataLoader(
        ...,
        data_path='data/processed/bci_competition_iv_2a',  # override default data dir
        split_config='configs/data_splits.json',            # override default config
        num_workers=0,                                      # passed to DataLoader
        transform=None,                                     # callable applied to each X
    )
"""

import json
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_DATA_PATH = os.path.join(_ROOT, "data", "processed", "bci_competition_iv_2a")
_DEFAULT_CONFIG    = os.path.join(_ROOT, "configs", "data_splits.json")


# ---------------------------------------------------------------------------
# Internal Dataset
# ---------------------------------------------------------------------------

class BCIDataset(Dataset):
    """Holds the NumPy arrays for a single split and exposes them as tensors.

    Labels are shifted from 1-indexed (1–4) to 0-indexed (0–3) so they work
    directly with PyTorch's CrossEntropyLoss.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y - 1, dtype=torch.long)  # 1-4 → 0-3
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npy(data_path: str, subject: str, session: str):
    """Load X and y arrays for one subject/session pair."""
    stem = f"{subject}{session}"
    X = np.load(os.path.join(data_path, f"{stem}_X.npy"))
    y = np.load(os.path.join(data_path, f"{stem}_y.npy"))
    return X, y


def _concat_sessions(data_path: str, subject: str, sessions: list[str]):
    """Concatenate multiple sessions for one subject."""
    Xs, ys = [], []
    for sess in sessions:
        X, y = _load_npy(data_path, subject, sess)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BCIDataLoader:
    """PyTorch DataLoader for BCI Competition IV Dataset 2a.

    Supports two modes:
      - 'subject_dependent': single subject, 4-fold blockwise CV (fold 0–3)
      - 'loso': cross-subject, 90 folds (5 train / 3 val / 1 test subjects)

    See module docstring for full usage examples.
    """

    def __init__(
        self,
        mode: str,
        split: str,
        *,
        # subject_dependent args
        subject: Optional[str] = None,
        fold: Optional[int] = None,          # 0–3 for subject_dependent
        # loso args
        fold_key: Optional[str] = None,      # e.g. 'A01_rep3' for loso
        # shared / optional
        data_path: str = _DEFAULT_DATA_PATH,
        split_config: str = _DEFAULT_CONFIG,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
    ):
        if mode not in ("subject_dependent", "loso"):
            raise ValueError(f"mode must be 'subject_dependent' or 'loso', got '{mode}'")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        with open(split_config) as f:
            config = json.load(f)

        if mode == "subject_dependent":
            X, y = self._build_subject_dependent(config, data_path, subject, fold, split)
        else:
            X, y = self._build_loso(config, data_path, fold_key, split)

        dataset = BCIDataset(X, y, transform=transform)
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # Mode-specific builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subject_dependent(config, data_path, subject, fold, split):
        if subject is None:
            raise ValueError("'subject' is required for mode='subject_dependent'")
        if fold is None:
            raise ValueError("'fold' (0–3) is required for mode='subject_dependent'")

        subj_cfg = config["subject_dependent"][subject]
        fold_cfg = subj_cfg[f"fold_{fold}"]
        indices  = fold_cfg[split]           # list of trial indices
        session  = subj_cfg["session"]       # always 'T'

        X, y = _load_npy(data_path, subject, session)
        return X[indices], y[indices]

    @staticmethod
    def _build_loso(config, data_path, fold_key, split):
        if fold_key is None:
            raise ValueError("'fold_key' (e.g. 'A01_rep3') is required for mode='loso'")

        fold_cfg = config["loso"][fold_key]

        if split == "train":
            subjects = fold_cfg["train_subjects"]
            sessions = fold_cfg["train_sessions"]
        elif split == "val":
            subjects = fold_cfg["val_subjects"]
            sessions = fold_cfg["val_sessions"]
        else:  # test
            subjects = [fold_cfg["test_subject"]]
            sessions = fold_cfg["test_sessions"]

        Xs, ys = [], []
        for subj in subjects:
            X, y = _concat_sessions(data_path, subj, sessions)
            Xs.append(X)
            ys.append(y)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    # ------------------------------------------------------------------
    # DataLoader pass-through
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset