"""
Preprocessing pipeline for BCI Competition IV Dataset 2a.

Loads raw .mat files, applies a Butterworth bandpass filter (4–40 Hz),
extracts per-trial windows, and saves processed arrays to data/processed/.

Usage:
    python src/data/preprocess.py

Output per subject/split (e.g. A01T):
    data/processed/bci_competition_iv_2a/A01T_X.npy  — (n_trials, 22, 1000)
    data/processed/bci_competition_iv_2a/A01T_y.npy  — (n_trials,)
"""

import os
import sys
import threading
import time
import numpy as np
import scipy.io
import scipy.signal


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "bci_competition_iv_2a")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bci_competition_iv_2a")

SUBJECTS    = [f"A{i:02d}" for i in range(1, 10)]
SPLITS      = {"T": "Training", "E": "Evaluation"}
CLASS_LABELS = {1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}

FS            = 250          # Hz
N_EEG         = 22           # first 22 channels are EEG (channels 23-25 are EOG)
BANDPASS_LOW  = 4.0          # Hz
BANDPASS_HIGH = 40.0         # Hz
FILTER_ORDER  = 5
TRIAL_START   = 0            # seconds after cue onset
TRIAL_END     = 4.0          # seconds after cue onset  →  1000 samples
N_TIMEPOINTS  = int((TRIAL_END - TRIAL_START) * FS)   # 1000


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class _Spinner:
    def __init__(self, msg: str = "Loading"):
        self._msg   = msg
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        dots = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self._msg}{'.' * dots}   ")
            sys.stdout.flush()
            dots = (dots % 3) + 1
            time.sleep(0.4)

    def start(self):  self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self._msg) + 6) + "\r")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(X: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied to each channel.

    Args:
        X: (n_samples, n_channels) raw EEG signal.
        fs: sampling rate in Hz.

    Returns:
        Filtered array of same shape.
    """
    sos = scipy.signal.butter(
        FILTER_ORDER,
        [BANDPASS_LOW, BANDPASS_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return scipy.signal.sosfiltfilt(sos, X, axis=0)


def extract_trials(X_filtered: np.ndarray, trial_onsets: np.ndarray) -> np.ndarray:
    """Slice fixed-length windows around each trial onset.

    Args:
        X_filtered: (n_samples, n_channels) filtered EEG.
        trial_onsets: (n_trials,) integer sample indices (0-based).

    Returns:
        epochs: (n_trials, N_EEG, N_TIMEPOINTS) — EEG channels only.
    """
    start_offset = int(TRIAL_START * FS)
    epochs = []
    for onset in trial_onsets:
        s = int(onset) + start_offset
        e = s + N_TIMEPOINTS
        if e > X_filtered.shape[0]:
            continue                           # skip truncated final trial
        window = X_filtered[s:e, :N_EEG]      # (N_TIMEPOINTS, N_EEG)
        epochs.append(window.T)                # → (N_EEG, N_TIMEPOINTS)
    return np.stack(epochs, axis=0)            # (n_trials, N_EEG, N_TIMEPOINTS)


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------

def load_mat(filepath: str) -> dict:
    """Load a BCI Competition IV 2a .mat file and aggregate across all 9 runs.

    The .mat structure stores data as a (1, 9) array of run structs. Each run
    contains a continuous signal X, per-trial onset indices, labels y, and an
    artifact flag per trial. This function concatenates all runs into a single
    continuous signal and adjusts trial onset indices accordingly.

    Returns a dict with keys: X, trial, y, artifacts, fs, gender, age.
    """
    mat  = scipy.io.loadmat(filepath, simplify_cells=True)
    runs = mat["data"]

    X_list, trial_list, y_list, art_list = [], [], [], []
    offset = 0

    for run in runs:
        X         = run["X"]                            # (n_samples, 25)
        trial     = run["trial"].astype(np.int64)       # (n_trials,)
        y         = np.asarray(run["y"]).ravel()
        artifacts = np.asarray(run["artifacts"]).ravel()

        X_list.append(X)
        trial_list.append(trial + offset)
        y_list.append(y)
        art_list.append(artifacts)
        offset += X.shape[0]

    first = runs[0]
    return {
        "X":         np.vstack(X_list),
        "trial":     np.concatenate(trial_list),
        "y":         np.concatenate(y_list).astype(np.int32),
        "artifacts": np.concatenate(art_list).astype(np.int32),
        "fs":        int(first["fs"]),
        "gender":    first.get("gender", "unknown"),
        "age":       first.get("age",    "unknown"),
    }


# ---------------------------------------------------------------------------
# Processing & saving
# ---------------------------------------------------------------------------

def process(mat: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply bandpass filter and extract trial windows.

    Returns:
        X_epochs: (n_trials, N_EEG, N_TIMEPOINTS)  float64
        y:        (n_trials,)                       int32
    """
    X_filt   = bandpass_filter(mat["X"], mat["fs"])   # (n_samples, 25)
    X_epochs = extract_trials(X_filt, mat["trial"])   # (n_trials, 22, 1000)

    # align labels with any trials dropped by extract_trials
    n_trials_kept = X_epochs.shape[0]
    y = mat["y"][:n_trials_kept]

    return X_epochs, y


def save(X: np.ndarray, y: np.ndarray, fname_stem: str) -> None:
    """Save X and y arrays as .npy files under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    x_path = os.path.join(OUT_DIR, f"{fname_stem}_X.npy")
    y_path = os.path.join(OUT_DIR, f"{fname_stem}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def summarize(mat: dict, X_epochs: np.ndarray, y: np.ndarray,
              subject: str, split_name: str) -> None:
    artifacts  = mat["artifacts"]
    n_artifact = int(np.sum(artifacts))

    print(f"  Subject {subject} | {split_name}")
    print(f"  {'─' * 46}")
    print(f"  Channels      : {mat['X'].shape[1]} total  ({N_EEG} EEG + {25 - N_EEG} EOG)")
    print(f"  Sampling rate : {mat['fs']} Hz")
    print(f"  Bandpass      : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz  (order {FILTER_ORDER}, zero-phase)")
    print(f"  Trial window  : {TRIAL_START}–{TRIAL_END} s  ({N_TIMEPOINTS} samples)")
    print(f"  Epochs shape  : {X_epochs.shape}  (trials × channels × time)")
    print(f"  Subject       : {mat['gender']}, age {mat['age']}")
    print(f"  Total trials  : {len(y)}  ({n_artifact} artifact-flagged)")
    for code, label in CLASS_LABELS.items():
        count = int(np.sum(y == code))
        bar   = "█" * (count // 4)
        print(f"    {label:<12}: {count:>3}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 52)
    print("  BCI Competition IV Dataset 2a — Preprocessing")
    print("=" * 52)
    print()

    for subject in SUBJECTS:
        for suffix, split_name in SPLITS.items():
            fname = f"{subject}{suffix}.mat"
            fpath = os.path.join(DATA_DIR, fname)

            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found\n")
                continue

            spinner = _Spinner(f"  Processing {fname}")
            spinner.start()
            mat = load_mat(fpath)
            X_epochs, y = process(mat)
            save(X_epochs, y, f"{subject}{suffix}")
            spinner.stop()

            summarize(mat, X_epochs, y, subject, split_name)

    print(f"  Saved to: {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
