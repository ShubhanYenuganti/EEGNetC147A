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
# Subject-dependent (single subject, fold 0–3)
python -m src.train --model eegnet --mode subject_dependent --subject A01 --fold 0 --epochs 300 --lr 0.001

# LOSO (leave-one-subject-out, fold_key e.g. A01_rep0)
python -m src.train --model cnn_gru --mode loso --fold_key A01_rep0 --epochs 300

# Batch experiment scripts
python run_eegnet_experiments.py
python run_cnngru_experiments.py
```

**Evaluation:**
```bash
python -m src.evaluate --model eegnet --mode subject_dependent
python -m src.evaluate --model cnn_gru --mode loso
```

## Architecture

### Data Pipeline
1. Raw `.mat` files in `data/raw/bci_competition_iv_2a/` (9 subjects × T/E sessions)
2. `src/data/preprocess.py`: Load → resample 250→128 Hz → ×1e6 → causal Butterworth 4–38 Hz → exponential running standardization → epoch 0.5–2.5 s → `(n, 22, 256)` `.npy` files. Also saves `_run.npy` (run index per trial).
3. `src/data/splits.py`: Generates `configs/data_splits.json`:
   - **Subject-dependent**: 4-fold blockwise CV using MI runs 3–6 (48 trials/block). Each fold: 2 train blocks / 1 val block / 1 test block.
   - **LOSO**: 90 folds (9 subjects × 10 random reps). Per fold: 5 train subjects (T session) / 3 val subjects (T session) / 1 test subject (E session only).
4. `src/data/dataloader.py`: `BCIDataLoader` wraps `.npy` files with split config, shifts labels 1→0 indexed.

### Training Framework
- `src/train.py`: Model registry (`eegnet`, `cnn_lstm`, `cnn_gru`, `dummy`), Adam optimizer, ReduceLROnPlateau scheduler, CrossEntropyLoss, saves best checkpoint by val accuracy to `experiments/checkpoints/`
- `src/evaluate.py`: Loads checkpoints, runs subject-dependent or LOSO eval, exports JSON to `experiments/results/`

### Models (`src/models/`)
- `eegnet.py`: ~2K param compact CNN (temporal conv → depthwise spatial → separable conv)
- `cnn_lstm.py`: EEGNet feature extractor + LSTM (hidden=128)
- `cnn_gru.py`: EEGNet extractor + Bidirectional GRU (hidden=48, layers=2) + temporal attention pooling — best performing, tuned via 3-round grid search

### Evaluation Modes
- **Subject-dependent**: 4-fold blockwise CV within a single subject's T session. `BCIDataLoader` takes `subject` + `fold` (0–3).
- **LOSO**: 90 folds. `BCIDataLoader` takes `fold_key` like `'A01_rep3'`. Test always uses E session; train/val always use T sessions.

### Key Constants
- 22 EEG channels, 128 Hz (resampled from 250 Hz), 2-second windows → shape `(n_trials, 22, 256)`
- 4 classes, labels 0–3 in DataLoader (shifted from raw 1–4)
- MI runs: indices 3–8 in T session (runs 0–2 are non-MI). Blockwise splits use runs 3–6 (first 4 MI runs).
- Random seed: `42` for LOSO splits; `42 + subject_id` for per-subject reproducibility
