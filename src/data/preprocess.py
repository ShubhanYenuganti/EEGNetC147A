"""
Preprocessing pipeline for BCI Competition IV Dataset 2a.

Usage:
    python src/data/preprocess.py
"""

import os
import sys
import threading
import time
import numpy as np
import scipy.io


class _Spinner:
    def __init__(self, msg: str = "Loading"):
        self._msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        dots = 0
        while not self._stop.is_set():
            display = self._msg + "." * dots + "   "
            sys.stdout.write(f"\r{display}")
            sys.stdout.flush()
            dots = (dots % 3) + 1
            time.sleep(0.4)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self._msg) + 6) + "\r")
        sys.stdout.flush()


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "bci_competition_iv_2a")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
SPLITS = {"T": "Training", "E": "Evaluation"}
CLASS_LABELS = {1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}
EEG_CHANNELS, EOG_CHANNELS = 22, 3


def load_mat(filepath: str) -> dict:
    mat = scipy.io.loadmat(filepath, simplify_cells=True)
    runs = mat["data"]

    X_list, trial_list, y_list, artifact_list = [], [], [], []
    trial_offset = 0

    for run in runs:
        X = run["X"]          # (n_samples, 25)
        trial = run["trial"]  # (n_trials,) — sample indices
        y = run["y"]          # (n_trials,)
        artifacts = run["artifacts"]  # (n_trials,)

        X_list.append(X)
        trial_list.append(trial.astype(np.int64) + trial_offset)
        y_list.append(y)
        artifact_list.append(artifacts)
        trial_offset += X.shape[0]

    first = runs[0]
    return {
        "X": np.vstack(X_list),
        "trial": np.concatenate(trial_list),
        "y": np.concatenate(y_list),
        "artifacts": np.concatenate(artifact_list),
        "fs": first["fs"],
        "gender": first.get("gender", "unknown"),
        "age": first.get("age", "unknown"),
    }


def summarize_mat(mat: dict, subject: str, split_name: str) -> None:
    X = mat["X"]  # (n_samples, 25)
    y = mat["y"]
    artifacts = mat["artifacts"]
    fs = mat["fs"]
    gender = mat["gender"]
    age = mat["age"]

    n_channels = X.shape[1]
    total_trials = len(y)
    n_artifact = int(np.sum(artifacts))

    trial_counts = {label: int(np.sum(y == code)) for code, label in CLASS_LABELS.items()}

    print(f"  Subject {subject} | {split_name}")
    print(f"  {'─' * 46}")
    print(f"  Channels      : {n_channels} total  ({EEG_CHANNELS} EEG + {EOG_CHANNELS} EOG)")
    print(f"  Sampling rate : {fs} Hz")
    print(f"  Signal shape  : {X.T.shape}  (channels × samples)")
    print(f"  Value range   : [{X.min():.2e}, {X.max():.2e}]")
    print(f"  Subject       : {gender}, age {age}")
    print(f"  Total trials  : {total_trials}  ({n_artifact} artifact-flagged)")
    for label, count in trial_counts.items():
        bar = "█" * (count // 4)
        print(f"    {label:<12}: {count:>3}  {bar}")
    print()


def main():
    print("=" * 52)
    print("  BCI Competition IV Dataset 2a — Raw Data Summary")
    print("=" * 52)
    print()

    for subject in SUBJECTS:
        for suffix, split_name in SPLITS.items():
            fname = f"{subject}{suffix}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found\n")
                continue
            spinner = _Spinner(f"  Loading {fname}")
            spinner.start()
            mat = load_mat(fpath)
            spinner.stop()
            summarize_mat(mat, subject, split_name)


if __name__ == "__main__":
    main()
