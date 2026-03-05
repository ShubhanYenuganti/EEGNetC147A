"""
Create subject-dependent and LOSO data split configurations.

Usage:
    python src/data/splits.py

Output:
    configs/data_splits.json
"""

import json
import os
import numpy as np


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bci_competition_iv_2a")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


def create_subject_dependent_splits():
    """70/15/15 train/val/test split within each subject's training session."""
    splits = {}
    for subject in SUBJECTS:
        labels = np.load(os.path.join(PROCESSED_DIR, f"{subject}T_y.npy"))
        n = len(labels)

        rng = np.random.RandomState(42 + int(subject[1:]))
        indices = rng.permutation(n)

        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        splits[subject] = {
            "train": indices[:n_train].tolist(),
            "val": indices[n_train:n_train + n_val].tolist(),
            "test": indices[n_train + n_val:].tolist(),
            "session": "T",
        }
    return splits


def create_loso_splits():
    """Leave-one-subject-out: 9 folds, each with 7 train, 1 val, 1 test subject."""
    splits = {}
    for fold_idx, test_subject in enumerate(SUBJECTS):
        remaining = [s for s in SUBJECTS if s != test_subject]
        # val subject is the one after test in circular order
        val_subject = SUBJECTS[(fold_idx + 1) % len(SUBJECTS)]
        train_subjects = [s for s in remaining if s != val_subject]
        splits[f"fold_{fold_idx}"] = {
            "train_subjects": train_subjects,
            "val_subject": val_subject,
            "test_subject": test_subject,
            "train_sessions": ["T", "E"],
            "val_sessions": ["T", "E"],
            "test_sessions": ["T", "E"],
        }
    return splits


def main():
    config = {
        "subject_dependent": create_subject_dependent_splits(),
        "loso": create_loso_splits(),
    }

    os.makedirs(CONFIG_DIR, exist_ok=True)
    out_path = os.path.join(CONFIG_DIR, "data_splits.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("Created data splits configuration")
    print(f"  Subject-dependent: {len(config['subject_dependent'])} subjects")
    for subj, split in config["subject_dependent"].items():
        print(f"    {subj}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    print(f"  LOSO: {len(config['loso'])} folds (7 train, 1 val, 1 test)")
    print(f"  Saved to: {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
