"""
Create subject-dependent and LOSO data split configurations aligned with desc_2a.pdf.

Usage:
    python src/data/splits.py

Output:
    configs/data_splits_TE.json

## Within-subject (subject_dependent)
Random 70/30 train/val split of the T session (seeded per subject).
Test always uses the E session.

## LOSO
90 folds: 9 test subjects × 10 random repetitions.
Per fold: 5 train subjects (T session), 3 val subjects (T session), 1 test subject (E session).
"""

import json
import os
import numpy as np


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bci_competition_iv_2a")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

# Run indices for MI blocks (runs 3–6 = 4 blocks × 48 trials)
MI_RUNS = [3, 4, 5, 6]

# Fold table: (test_run, val_run, train_runs)
_FOLD_TABLE = [
    (3, 4, [5, 6]),
    (4, 5, [3, 6]),
    (5, 6, [3, 4]),
    (6, 3, [4, 5]),
]


def create_subject_dependent_splits():
    """70/30 random train/val split within each subject's T session; E session used for test."""
    splits = {}
    for subject in SUBJECTS:
        labels = np.load(os.path.join(PROCESSED_DIR, f"{subject}T_y.npy"))
        n = len(labels)
        rng = np.random.RandomState(42 + int(subject[1:]))
        indices = rng.permutation(n)
        n_val = int(0.30 * n)

        splits[subject] = {
            "train": indices[n_val:].tolist(),
            "val": indices[:n_val].tolist(),
            "train_session": "T",
            "test_session": "E",
	} 
    return splits


def create_loso_splits():
    """90 folds: 9 test subjects × 10 random repetitions (5 train / 3 val / 1 test)."""
    splits = {}
    rng = np.random.RandomState(42)
    for test_subject in SUBJECTS:
        others = [s for s in SUBJECTS if s != test_subject]
        for rep in range(10):
            perm = rng.permutation(8)
            train_subjects = [others[i] for i in perm[:5]]
            val_subjects   = [others[i] for i in perm[5:]]
            fold_key = f"{test_subject}_rep{rep}"
            splits[fold_key] = {
                "train_subjects": train_subjects,
                "val_subjects":   val_subjects,
                "test_subject":   test_subject,
                "train_sessions": ["T"],
                "val_sessions":   ["T"],
                "test_sessions":  ["E"],
            }
    return splits


def main():
    config = {
        "subject_dependent": create_subject_dependent_splits(),
        "loso": create_loso_splits(),
    }

    os.makedirs(CONFIG_DIR, exist_ok=True)
    out_path = os.path.join(CONFIG_DIR, "data_splits_TE.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    sd = config["subject_dependent"]
    loso = config["loso"]
    print("Created data splits configuration")
    print(f"  Subject-dependent: {len(sd)} subjects (T train/val, E test)")
    for subj, split in sd.items():
        print(f"    {subj}: train={len(split['train'])}, val={len(split['val'])}, test_session=E")
    print(f"  Saved to: {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
