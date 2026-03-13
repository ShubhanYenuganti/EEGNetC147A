"""
Preprocessing pipeline for BCI Competition IV Dataset 2a.

Loads raw .mat files, applies a Butterworth bandpass filter (4–40 Hz),
extracts per-trial windows, and saves processed arrays to data/processed/.

Usage:
    python src/data/preprocess.py

Output per subject/split (e.g. A01T):
    data/processed/bci_competition_iv_2a/A01T_X.npy  — (n_trials, 22, 1000)
    data/processed/bci_competition_iv_2a/A01T_y.npy  — (n_trials,)

Notes on trial windowing:
    BCI IV-2a trial structure (seconds relative to trial[] onset):
        t = 0.0s  : fixation cross appears          ← trial[] points here
        t = 2.0s  : cue arrow appears (MI onset)    ← TRIAL_START
        t = 2–6s  : motor imagery period             ← the signal we want
        t = 6.0s  : cue disappears                   ← TRIAL_END
    We extract [2, 6) seconds after the trial onset, giving exactly 1000
    samples of motor-imagery EEG at 250 Hz.
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

SUBJECTS     = [f"A{i:02d}" for i in range(1, 10)]
SPLITS       = {"T": "Training", "E": "Evaluation"}
CLASS_LABELS = {1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}

FS            = 250          # Hz
N_EEG         = 22           # first 22 channels are EEG (channels 23–25 are EOG)
BANDPASS_LOW  = 4.0          # Hz
BANDPASS_HIGH = 40.0         # Hz
FILTER_ORDER  = 5

# --- FIX: start at cue onset (t=2s), not fixation cross (t=0s) ---
# trial[] indices point to t=0 (fixation cross). The cue appears at t=2s.
# Motor imagery runs from t=2s to t=6s — 4 seconds = 1000 samples @ 250 Hz.
TRIAL_START  = 2.0           # seconds after trial onset → skip fixation period
TRIAL_END    = 6.0           # seconds after trial onset → end of MI window
N_TIMEPOINTS = int((TRIAL_END - TRIAL_START) * FS)   # 1000


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class _Spinner:
    def __init__(self, msg: str = "Loading"):
        self._msg    = msg
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        dots = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self._msg}{'.' * dots}   ")
            sys.stdout.flush()
            dots = (dots % 3) + 1
            time.sleep(0.4)

    def start(self): self._thread.start()

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
        X:  (n_samples, n_channels) raw EEG signal.
        fs: sampling rate in Hz.

    Returns:
        Filtered array of the same shape.
    """
    sos = scipy.signal.butter(
        FILTER_ORDER,
        [BANDPASS_LOW, BANDPASS_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return scipy.signal.sosfiltfilt(sos, X, axis=0)


def extract_trials(
    X_filtered:   np.ndarray,
    trial_onsets: np.ndarray,
    artifacts:    np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice fixed-length windows around each trial cue onset.

    Windows are extracted starting at TRIAL_START seconds after the stored
    trial onset (which marks the fixation cross), so the extracted signal
    corresponds to the full motor-imagery period.

    Trials flagged as artifacts are excluded entirely.

    Args:
        X_filtered:   (n_samples, n_channels) filtered EEG.
        trial_onsets: (n_trials,) integer sample indices (0-based), pointing
                      to the fixation-cross onset of each trial.
        artifacts:    (n_trials,) array; non-zero means the trial is flagged.

    Returns:
        epochs:          (n_clean_trials, N_EEG, N_TIMEPOINTS)
        clean_trial_idx: (n_clean_trials,) original trial indices kept,
                         useful for aligning labels.
    """
    start_offset = int(TRIAL_START * FS)   # 500 samples = 2 s
    kept_epochs  = []
    kept_indices = []

    for i, onset in enumerate(trial_onsets):
        # Skip artifact-flagged trials
        if artifacts[i] != 0:
            continue

        s = int(onset) + start_offset
        e = s + N_TIMEPOINTS

        # Skip if the window extends beyond the recording
        if e > X_filtered.shape[0]:
            continue

        window = X_filtered[s:e, :N_EEG]   # (N_TIMEPOINTS, N_EEG)
        kept_epochs.append(window.T)        # → (N_EEG, N_TIMEPOINTS)
        kept_indices.append(i)

    if not kept_epochs:
        raise RuntimeError("No valid (non-artifact) trials found.")

    epochs = np.stack(kept_epochs, axis=0)                    # (n, N_EEG, N_TIMEPOINTS)
    clean_trial_idx = np.array(kept_indices, dtype=np.int64)  # (n,)
    return epochs, clean_trial_idx


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

    X_list, trial_list, y_list, art_list, run_list = [], [], [], [], []
    offset = 0

    for run_idx, run in enumerate(runs):
        X         = run["X"]                          # (n_samples, 25)
        trial     = run["trial"].astype(np.int64)     # (n_trials,)
        y         = np.asarray(run["y"]).ravel()
        artifacts = np.asarray(run["artifacts"]).ravel()

        X_list.append(X)
        trial_list.append(trial + offset)
        y_list.append(y)
        art_list.append(artifacts)
        run_list.append(np.full(len(trial), run_idx + 1, dtype=np.int32))
        offset += X.shape[0]

    first = runs[0]
    return {
        "X":         np.vstack(X_list),
        "trial":     np.concatenate(trial_list),
        "y":         np.concatenate(y_list).astype(np.int32),
        "artifacts": np.concatenate(art_list).astype(np.int32),
        "run":       np.concatenate(run_list).astype(np.int32),
        "fs":        int(first["fs"]),
        "gender":    first.get("gender", "unknown"),
        "age":       first.get("age",    "unknown"),
    }


# ---------------------------------------------------------------------------
# Processing & saving
# ---------------------------------------------------------------------------

def process(mat: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply bandpass filter, extract cue-locked trial windows, remove artifacts.

    Returns:
        X_epochs: (n_clean_trials, N_EEG, N_TIMEPOINTS)  float64
        y:        (n_clean_trials,)                       int32, values in {1,2,3,4}
    """
    X_filt = bandpass_filter(mat["X"], mat["fs"])

    # --- FIX: pass artifacts into extract_trials for correct, per-trial removal ---
    # Previously artifacts were collected but never used, and label alignment
    # assumed only trailing trials could be dropped (now any trial can be skipped).
    X_epochs, clean_idx = extract_trials(X_filt, mat["trial"], mat["artifacts"])

    # Align labels and run indices using the indices of kept trials
    y   = mat["y"][clean_idx]
    run = mat["run"][clean_idx]

    return X_epochs, y, run


def save(X: np.ndarray, y: np.ndarray, run: np.ndarray, fname_stem: str) -> None:
    """Save X, y, and run arrays as .npy files under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_X.npy"), X)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_y.npy"), y)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_run.npy"), run)


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def summarize(
    mat:       dict,
    X_epochs:  np.ndarray,
    y:         np.ndarray,
    n_dropped: int,
    subject:   str,
    split_name: str,
) -> None:
    print(f"  Subject {subject} | {split_name}")
    print(f"  {'─' * 50}")
    print(f"  Channels      : {mat['X'].shape[1]} total  ({N_EEG} EEG + {25 - N_EEG} EOG)")
    print(f"  Sampling rate : {mat['fs']} Hz")
    print(f"  Bandpass      : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz  (order {FILTER_ORDER}, zero-phase)")
    print(f"  Trial window  : {TRIAL_START}–{TRIAL_END} s after trial onset  ({N_TIMEPOINTS} samples)")
    print(f"  Epochs shape  : {X_epochs.shape}  (trials × channels × time)")
    print(f"  Subject       : {mat['gender']}, age {mat['age']}")
    print(f"  Trials kept   : {len(y)}  ({n_dropped} artifact-flagged, removed)")
    for code, label in CLASS_LABELS.items():
        count = int(np.sum(y == code))
        bar   = "█" * (count // 4)
        print(f"    {label:<12}: {count:>3}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 54)
    print("  BCI Competition IV Dataset 2a — Preprocessing")
    print("=" * 54)
    print()

    for subject in SUBJECTS:
        for suffix, split_name in SPLITS.items():
            fname = f"{subject}{suffix}.mat"
            fpath = os.path.join(DATA_DIR, fname)

            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found — skipping\n")
                continue

            spinner = _Spinner(f"  Processing {fname}")
            spinner.start()
            mat = load_mat(fpath)
            X_epochs, y, run = process(mat)
            save(X_epochs, y, run, f"{subject}{suffix}")
            spinner.stop()

            n_dropped = int(np.sum(mat["artifacts"]))
            summarize(mat, X_epochs, y, n_dropped, subject, split_name)

    print(f"  Saved to: {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()