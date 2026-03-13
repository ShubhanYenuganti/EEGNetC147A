# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based motor imagery classification using BCI Competition IV Dataset 2a (9 subjects, 4 classes: left hand, right hand, feet, tongue). The project benchmarks CNN and CNN-RNN hybrid architectures for cross-subject generalization.

There are **two independent preprocessing and training pipelines**:

| Pipeline | Freq | Script | Data dir | Split config |
|----------|------|--------|----------|--------------|
| 250 Hz (original) | 250 Hz native | `src/train.py` | `data/processed/bci_competition_iv_2a/` | `configs/data_splits_TE.json` |
| 128 Hz / ERS | resampled 128 Hz | `src/train_128.py` | `data/processed/bci_competition_iv_2a_128/` | `configs/data_splits_128.json` |

The two pipelines produce different data and **cannot be mixed**. Models trained with one pipeline must be evaluated with the corresponding evaluate script.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Pipeline 1: 250 Hz (original)

### Preprocessing

Run once, in order:

```bash
python src/data/preprocess.py   # Extract trials → data/processed/bci_competition_iv_2a/
python src/data/splits.py       # Generate configs/data_splits_TE.json
```

**What it does:**
- Loads raw `.mat` files from `data/raw/bci_competition_iv_2a/`
- Scales signal ×1e6 (raw → µV)
- Zero-phase (forward-backward) Butterworth bandpass 4–40 Hz, order 5
- Epochs trial window 2–6 s after trial onset (full 4-second MI period)
- No resampling — native 250 Hz retained → shape `(n_trials, 22, 1000)`
- Saves `_X.npy`, `_y.npy`, `_run.npy` per subject/session

**Splits (`data_splits_TE.json`):**
- **Subject-dependent**: T session → 70/15 train/val split by index, E session → test
- **LOSO**: 36 folds (9 test subjects × 4 random reps, keys like `A01_rep0`). Per fold: 5 train subjects (T) / 3 val subjects (T) / 1 test subject (E only)

### Training (250 Hz)

```bash
# Subject-dependent — single subject
python -m src.train --model alternative_eegnet_250 --mode subject_dependent --subject A01 --epochs 300 --lr 0.001

# Subject-dependent — all 9 subjects
python run_subject_dependent.py --model alternative_eegnet_250 --epochs 300 --lr 0.001
python run_subject_dependent.py --model cnn_gru --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16

# LOSO — single fold (fold key: {subject}_rep{0-3}, e.g. A01_rep0)
python -m src.train --model alternative_eegnet_250 --mode loso --fold A01_rep0 --epochs 300 --lr 0.001

# LOSO — all 36 folds with ETA timer
python run_loso.py --model alternative_eegnet_250 --epochs 300 --lr 0.001
python run_loso.py --model cnn_gru --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16

# LOSO — fast run (1 rep = 9 folds)
python run_loso.py --model cnn_gru --reps 1 --epochs 300
```

**Available models for `src/train.py`:**
`alternative_eegnet_250`, `eegnet`, `alternative_eegnet`, `cnn_lstm`, `cnn_gru`, `transformer`, `dummy`

Normalization: `Normalizer` (fitted on training set, applied to val/test) is used in this pipeline.

### Evaluation (250 Hz)

```bash
# Subject-dependent
python -m src.evaluate --model alternative_eegnet_250 --mode subject_dependent

# LOSO — aggregates mean ± std across all folds
python -m src.evaluate --model alternative_eegnet_250 --mode loso
python -m src.evaluate --model cnn_gru --mode loso
```

Checkpoints are loaded from `experiments/checkpoints/`, results saved to `experiments/results/`.

---

## Pipeline 2: 128 Hz / ERS

### Why a second pipeline?

The 128 Hz / ERS pipeline differs from the original in three significant ways:

**1. Resampling to 128 Hz**
The native 250 Hz signal is downsampled to 128 Hz via polyphase resampling (scipy `resample_poly`). This halves the number of time samples per trial (1000 → 256) and reduces memory and compute. All models in this pipeline are designed around the 256-sample input shape.

**2. Causal bandpass filter (single-pass)**
The original pipeline uses a zero-phase (forward-backward) Butterworth filter, which is non-causal — it uses future samples to filter past ones. This is acceptable offline but can introduce subtle epoch-boundary artefacts and is inconsistent with real-time BCI deployment. The 128 Hz pipeline uses a causal single-pass (`sosfilt`) Butterworth order 3, 4–38 Hz. The upper cutoff is also tightened to 38 Hz (vs 40 Hz) to avoid aliasing near the Nyquist of 64 Hz.

**3. Exponential Running Standardisation (ERS) on the continuous signal**
Instead of epoch-level normalisation (z-score per trial), ERS normalises the *continuous* signal before epoching using an online causal running estimate of mean and variance (decay α = 1e-3). This means:
- No look-ahead: the normalisation at time t uses only samples up to t
- Stationarity correction is applied at the recording level, not per-trial
- Downstream `Normalizer` and `TrialNormalizer` in `BCIDataLoader` must **not** be applied — the data is already normalised. Both `train_128.py` and `evaluate_128.py` skip normalisation for this reason.

**4. Narrower trial window**
The original pipeline epochs 2–6 s after trial onset (full 4-second MI window). The 128 Hz pipeline epochs 2.5–4.5 s (0.5–2.5 s post-cue), targeting the peak MI response and producing exactly 256 samples at 128 Hz.

### Preprocessing

Run once, in order:

```bash
python src/data/alternative_preprocess.py   # → data/processed/bci_competition_iv_2a_128/
python src/data/alternative_splits.py       # → configs/data_splits_128.json
```

`alternative_preprocess.py` pipeline per subject/session:
1. Load raw `.mat`, aggregate all 9 runs
2. Polyphase resample 250 → 128 Hz (all 25 channels)
3. Scale ×1e6 (→ µV)
4. Causal Butterworth bandpass 4–38 Hz, order 3, single-pass (`sosfilt`)
5. ERS normalisation on the continuous signal (α = 1e-3, ε = 1e-4, init on first 1000 samples)
6. Epoch at 128 Hz: window [2.5, 4.5) s after trial onset → 256 samples
7. Drop artifact-flagged trials
8. Save `{subject}{T|E}_X.npy` `(n, 22, 256)`, `_y.npy`, `_run.npy`

`alternative_splits.py` produces `configs/data_splits_128.json`:
- **Subject-dependent only**: 80/20 stratified random train/val split of T session per subject (seed = subject index 1–9), E session → test. No LOSO for the 128 Hz pipeline.

### Training (128 Hz / ERS)

```bash
# Subject-dependent — single subject
python -m src.train_128 --model <model> --mode subject_dependent --subject A01 \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16 --dropout 0.5

# Subject-dependent — all 9 subjects via runner
python run_subject_dependent_128.py --model <model> --epochs 300 --lr 0.001 \
    --weight_decay 5e-4 --batch_size 16 --dropout 0.5
```

**Available models for `src/train_128.py`:**

| Model name | File | Notes |
|---|---|---|
| `alternative_eegnet` | `models/alternative_eegnet.py` | EEGNet, 128 Hz native, n_timepoints=256 |
| `cnn_gru_alternative` | `models/cnn_gru_alternative.py` | **Recommended.** 128 Hz-native CNN+BiGRU+attention. temporal_kernel=64, track_running_stats=False, dropout forwarded |
| `cnn_lstm_alternative` | `models/cnn_lstm_alternative.py` | **Recommended.** 128 Hz-adapted CNN+LSTM. temporal_kernel=64 (sfreq//2), track_running_stats=False, dropout forwarded |

**Recommended commands per model:**

```bash
# cnn_gru_alternative (best performing, tuned defaults)
python -m src.train_128 --model cnn_gru_alternative --mode subject_dependent --subject A01 \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 8 --dropout 0.4

# All 9 subjects
python run_subject_dependent_128.py --model cnn_gru_alternative \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 8 --dropout 0.4

# cnn_lstm_alternative
python -m src.train_128 --model cnn_lstm_alternative --mode subject_dependent --subject A01 \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 8 --dropout 0.4

# All 9 subjects
python run_subject_dependent_128.py --model cnn_lstm_alternative \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 8 --dropout 0.4

# alternative_eegnet
python -m src.train_128 --model alternative_eegnet --mode subject_dependent --subject A01 \
    --epochs 300 --lr 0.001 --dropout 0.4
```

Checkpoints are saved to `experiments/checkpoints/{model}_{subject}_subject_dependent_128_best.pt`.
Training results (loss/accuracy history) are saved to `experiments/results/{model}_{subject}_subject_dependent_128.json`.

**Important:** Do not apply `--norm` or `TrialNormalizer` with this pipeline — ERS normalisation was already applied during preprocessing. Double-normalising degrades training signal.

### Evaluation (128 Hz / ERS)

```bash
# Subject-dependent — all 9 subjects, aggregates mean ± std
python -m src.evaluate_128 --model cnn_gru_alternative --mode subject_dependent
python -m src.evaluate_128 --model cnn_lstm_alternative --mode subject_dependent
python -m src.evaluate_128 --model alternative_eegnet --mode subject_dependent
```

Outputs a table of train/val/test accuracy per subject plus mean and std across subjects.
Results saved to `experiments/results/{model}_subject_dependent_128_eval.json`.

The evaluator loads checkpoints from `experiments/checkpoints/` using the same naming convention as `train_128.py`. If a checkpoint is missing for a subject, that subject is skipped with a warning.

---

## Architecture

### Data Pipeline
1. Raw `.mat` files in `data/raw/bci_competition_iv_2a/` (9 subjects × T/E sessions)
2. `src/data/preprocess.py`: Load → ×1e6 → zero-phase Butterworth 4–40 Hz → epoch 2–6 s after cue → `(n, 22, 1000)` `.npy` files. **No resampling — native 250 Hz retained.** Also saves `_run.npy` (run index per trial).
3. `src/data/splits.py`: Generates `configs/data_splits_TE.json`:
   - **Subject-dependent**: T session for train/val (index-based 70/15 split), E session for test.
   - **LOSO**: 36 folds (9 test subjects × 4 random reps, keys like `A01_rep0`). Per fold: 5 train subjects (T session) / 3 val subjects (T session) / 1 test subject (E session only).
4. `src/data/dataloader.py`: `BCIDataLoader` wraps `.npy` files with split config, shifts labels 1→0 indexed. Provides `Normalizer` (fit on train, apply to val/test) and `TrialNormalizer` (per-trial z-score, no fitting needed).

### Training Framework
- `src/train.py`: Model registry, Adam optimizer, CosineAnnealingLR scheduler, CrossEntropyLoss (label smoothing 0.1), early stopping (patience 40, min epoch 50), sliding-window + amplitude augmentation in subject-dependent mode. Uses `Normalizer` (fitted on training set). LOSO mode takes `--fold` string key.
- `src/train_128.py`: Identical training loop to `train.py` but defaults to 128 Hz data directory and `data_splits_128.json`. No normalizer applied. Contains the model registry for the 128 Hz pipeline including `cnn_gru_alternative` and `cnn_lstm_alternative`.
- `src/evaluate.py` / `src/evaluate_128.py`: Load checkpoints, run evaluation, export JSON. `evaluate_128.py` imports `get_model` from `train_128.py` to ensure model construction is consistent with training.

### Models (`src/models/`)

**250 Hz pipeline:**
- `alternative_eegnet_250.py`: Baseline EEGNet — 250 Hz native, input `(batch, 1, 22, 1000)`. Temporal kernel = 125 (0.5 s), AvgPool stride 4.
- `cnn_lstm.py`: EEGNet feature extractor + LSTM (hidden=128). Temporal kernel hardcoded to 25 (≈100 ms at 250 Hz). 250 Hz design.
- `cnn_gru.py`: EEGNet extractor + Bidirectional GRU (hidden=48, layers=2) + temporal attention pooling. Native 128 Hz design despite being the "original" file — temporal_kernel = sfreq//2 = 64.

**128 Hz / ERS pipeline:**
- `eegnet.py`: 128 Hz EEGNet, input `(batch, 1, 22, 256)`
- `alternative_eegnet.py`: 128 Hz EEGNet variant, input `(batch, 1, 22, 512)`
- `cnn_gru_alternative.py`: Copy of `cnn_gru.py`, explicitly for the 128 Hz pipeline. temporal_kernel=64, sep_kernel=16, track_running_stats=False, dropout forwarded from CLI.
- `cnn_lstm_alternative.py`: 128 Hz-adapted CNN+LSTM. temporal_kernel=sfreq//2=64 (not hardcoded 25), track_running_stats=False, dropout forwarded from CLI.
- `transformer.py`: EEGNet extractor + Transformer encoder

### Evaluation Modes
- **Subject-dependent**: T session for train/val, E session for test. `BCIDataLoader` takes `subject` (e.g. `'A01'`).
- **LOSO**: 36 folds (string `fold` key like `A01_rep0`, resolved via `configs/data_splits_TE.json`). Test always uses E session; train/val always use T sessions. **250 Hz pipeline only** — not implemented for 128 Hz.

### Key Constants (250 Hz pipeline)
- 22 EEG channels, 250 Hz native, 4-second MI windows → shape `(n_trials, 22, 1000)`
- 4 classes, labels 0–3 in DataLoader (shifted from raw 1–4)
- Trial window: 2–6 s post-cue. Bandpass: 4–40 Hz (zero-phase Butterworth, order 5).
- Random seed: `42` for LOSO splits

### Key Constants (128 Hz / ERS pipeline)
- 22 EEG channels, 128 Hz resampled, 2-second MI windows → shape `(n_trials, 22, 256)`
- Trial window: 2.5–4.5 s after trial onset (0.5–2.5 s post-cue). Bandpass: 4–38 Hz (causal, order 3).
- ERS: α = 1e-3, ε = 1e-4, init on first 1000 samples of continuous signal
- Splits: 80/20 train/val per subject (seed = subject index), no LOSO
