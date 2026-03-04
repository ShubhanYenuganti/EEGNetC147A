# Preprocessing Pipeline — BCI Competition IV Dataset 2a

## Overview

`src/data/preprocess.py` loads all 18 `.mat` subject files (9 training + 9 evaluation), applies a Butterworth bandpass filter, extracts per-trial windows, and writes the resulting tensors to `data/processed/bci_competition_iv_2a/`.

---

## Input Data Format

Each `.mat` file contains a `data` struct array of shape `(1, 9)` — one element per recording run. Each run element has:

| Field       | Shape            | Description                                |
|-------------|------------------|--------------------------------------------|
| `X`         | `(n_samples, 25)`| Continuous EEG+EOG signal (µV, float64)    |
| `trial`     | `(n_trials,)`    | Sample index of each trial cue onset       |
| `y`         | `(n_trials,)`    | Class label: 1=Left Hand, 2=Right Hand, 3=Both Feet, 4=Tongue |
| `artifacts` | `(n_trials,)`    | 0 = clean, 1 = artifact-contaminated       |
| `fs`        | scalar           | Sampling rate (250 Hz)                     |
| `gender`    | str              | Subject demographic                        |
| `age`       | int              | Subject demographic                        |

Channels 1–22 are EEG; channels 23–25 are EOG and are discarded after filtering.

---

## Processing Steps

### 1. Aggregation across runs (`load_mat`)

All 9 runs per subject are concatenated into a single continuous signal. The `trial` onset indices from each run are offset by the cumulative sample count of all prior runs so they remain valid after concatenation.

```
X_full   = vstack([run.X for run in runs])          # (N_total_samples, 25)
trial_abs = trial_local + cumulative_sample_offset   # absolute onset positions
```

### 2. Butterworth bandpass filter (`bandpass_filter`)

A 5th-order Butterworth bandpass filter is applied to the full continuous signal (all 25 channels) using `scipy.signal.sosfiltfilt` (zero-phase, forward-backward pass, no phase distortion).

- **Passband:** 4–40 Hz
  - Lower bound (4 Hz) captures the mu (8–13 Hz) and beta (13–30 Hz) motor imagery bands while rejecting slow drifts and DC offsets.
  - Upper bound (40 Hz) rejects high-frequency noise and line interference (50/60 Hz) above the motor bands.
- **Order:** 5 (steeper roll-off than a lower-order filter, standard for EEG)
- **Implementation:** Second-order sections (`sos`) numerically stable form; `sosfiltfilt` applies the filter twice (forward + backward) to achieve zero phase shift.

### 3. Trial window extraction (`extract_trials`)

For each trial onset index, a fixed-length window is sliced from the filtered signal:

```
window = X_filtered[onset : onset + 1000, :22]   # (1000, 22)
epoch  = window.T                                  # (22, 1000)
```

- **Window:** 0 to 4 seconds after cue onset → **1000 samples** at 250 Hz
- **Channels:** Only the first 22 EEG channels are retained (EOG discarded)
- Trials whose window would extend past the end of the signal are silently dropped (rare edge case at the end of evaluation runs)

### 4. Output format

Final tensor shape per subject/split: **(n_trials, 22, 1000)**

- `axis 0` — trials (288 per subject for complete runs)
- `axis 1` — EEG channels (22)
- `axis 2` — time samples (1000 @ 250 Hz = 4 s)

This shape is the canonical EEGNet input format: `(batch, channels, time)`.

---

## Output Files

Written to `data/processed/bci_competition_iv_2a/`:

| File          | Shape             | dtype   | Contents                        |
|---------------|-------------------|---------|---------------------------------|
| `A01T_X.npy`  | `(288, 22, 1000)` | float64 | Filtered, epoched EEG           |
| `A01T_y.npy`  | `(288,)`          | int32   | Class labels (1–4)              |
| `A01E_X.npy`  | `(288, 22, 1000)` | float64 | Evaluation set                  |
| `A01E_y.npy`  | `(288,)`          | int32   | Evaluation labels               |
| ...           | ...               | ...     | (repeated for A02–A09)          |

Labels use the 1-indexed convention from the original dataset:

| Label | Class      |
|-------|------------|
| 1     | Left Hand  |
| 2     | Right Hand |
| 3     | Both Feet  |
| 4     | Tongue     |

---

## Design Decisions

**Why 4–40 Hz?** Motor imagery is reflected primarily in mu (8–13 Hz) and beta (13–30 Hz) rhythms. The 4 Hz lower bound is conservative enough to preserve these bands without including slow cortical potentials. 40 Hz cleanly excludes power-line noise.

**Why 0–4 s window?** The BCI Competition IV 2a protocol presents the motor imagery cue at t=0 and the subject imagines for 4 seconds. This is the full available imagery period and is the standard windowing used in EEGNet literature for this dataset.

**Why zero-phase filtering?** `sosfiltfilt` prevents temporal smearing of event-locked responses. Causal filters introduce group delay that would shift the post-cue onset relative to the expected neural response timing.

**Why keep artifact-flagged trials?** Artifact labels are preserved in `load_mat` but not used to exclude trials during preprocessing. Downstream training code can choose to mask or weight these trials — keeping them here is less destructive than silently dropping data.

---

## Usage

```bash
python src/data/preprocess.py
```

Loading and processing all 18 files takes approximately 10–20 seconds. Progress is shown per file with a spinner.
