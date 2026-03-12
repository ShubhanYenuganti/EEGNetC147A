# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based motor imagery classification using BCI Competition IV Dataset 2a (9 subjects, 4 classes: left hand, right hand, feet, tongue). The project benchmarks CNN and CNN-RNN hybrid architectures for cross-subject generalization.

## Commands

**Setup:**
```bash
pip install -r requirements.txt
```

**Preprocessing (run once, in this order):**
```bash
python src/data/preprocess.py   # Extract trials → data/processed/ (also writes _run.npy)
python src/data/splits.py       # Generate configs/data_splits.json
```

**Training:**
```bash
# Subject-dependent (single subject)
python -m src.train --model alternative_eegnet_250 --mode subject_dependent --subject A01 --epochs 300 --lr 0.001

# LOSO — single fold (fold key format: {subject}_rep{0-9}, e.g. A01_rep0)
python -m src.train --model alternative_eegnet_250 --mode loso --fold A01_rep0 --epochs 300 --lr 0.001

# Subject-dependent — all 9 subjects for any model
python run_subject_dependent.py --model alternative_eegnet_250 --epochs 300 --lr 0.001
python run_subject_dependent.py --model cnn_gru --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16

# LOSO — 4 reps per subject (36 folds) with ETA timer, for any model
python run_loso.py --model alternative_eegnet_250 --epochs 300 --lr 0.001
python run_loso.py --model cnn_gru --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16
# Override reps (e.g. 1 rep = 9 folds for fastest run):
python run_loso.py --model cnn_gru --reps 1 --epochs 300
```

**Evaluation:**
```bash
# Subject-dependent
python -m src.evaluate --model alternative_eegnet_250 --mode subject_dependent

# LOSO — all folds (aggregates mean ± std across folds)
python -m src.evaluate --model alternative_eegnet_250 --mode loso
python -m src.evaluate --model cnn_gru --mode loso
```

## Architecture

### Data Pipeline
1. Raw `.mat` files in `data/raw/bci_competition_iv_2a/` (9 subjects × T/E sessions)
2. `src/data/preprocess.py`: Load → ×1e6 → zero-phase Butterworth 4–40 Hz → epoch 2–6 s after cue → `(n, 22, 1000)` `.npy` files. **No resampling — native 250 Hz retained.** Also saves `_run.npy` (run index per trial).
3. `src/data/splits.py`: Generates `configs/data_splits_TE.json`:
   - **Subject-dependent**: T session for train/val (index-based 70/15 split), E session for test.
   - **LOSO**: 36 folds (9 test subjects × 4 random reps, keys like `A01_rep0`). Per fold: 5 train subjects (T session) / 3 val subjects (T session) / 1 test subject (E session only).
4. `src/data/dataloader.py`: `BCIDataLoader` wraps `.npy` files with split config, shifts labels 1→0 indexed. Provides `Normalizer` (fit on train, apply to val/test) and `TrialNormalizer` (per-trial z-score, no fitting needed).

### Training Framework
- `src/train.py`: Model registry (`alternative_eegnet_250`, `eegnet`, `alternative_eegnet`, `cnn_lstm`, `cnn_gru`, `transformer`, `dummy`), Adam optimizer, ReduceLROnPlateau scheduler, CrossEntropyLoss, saves best checkpoint by val accuracy to `experiments/checkpoints/`. Uses `Normalizer` (fitted on training set) for normalization. LOSO mode takes `--fold` as integer index.
- `src/evaluate.py`: Loads checkpoints, runs subject-dependent or LOSO eval, exports JSON to `experiments/results/` with mean and std.

### Models (`src/models/`)
- `alternative_eegnet_250.py`: **Baseline EEGNet** — 250 Hz native variant, input `(batch, 1, 22, 1000)`. Temporal conv kernel = 125 (0.5 s), AvgPool stride 4. Use this for EEGNet baseline experiments.
- `eegnet.py`: 128 Hz variant, input `(batch, 1, 22, 256)`
- `alternative_eegnet.py`: 128 Hz variant, input `(batch, 1, 22, 512)`
- `cnn_lstm.py`: EEGNet feature extractor + LSTM (hidden=128)
- `cnn_gru.py`: EEGNet extractor + Bidirectional GRU (hidden=48, layers=2) + temporal attention pooling — best performing, tuned via 3-round grid search
- `transformer.py`: EEGNet extractor + Transformer encoder

### Evaluation Modes
- **Subject-dependent**: T session for train/val, E session for test. `BCIDataLoader` takes `subject` (e.g. `'A01'`).
- **LOSO**: 36 folds (string `fold` key like `A01_rep0`, resolved via `configs/data_splits_TE.json`). Test always uses E session; train/val always use T sessions.

### Key Constants
- 22 EEG channels, **250 Hz (native, no resampling)**, 4-second MI windows → shape `(n_trials, 22, 1000)`
- 4 classes, labels 0–3 in DataLoader (shifted from raw 1–4)
- Trial window: 2–6 s post-cue. Bandpass: 4–40 Hz (zero-phase Butterworth).
- Random seed: `42` for LOSO splits
