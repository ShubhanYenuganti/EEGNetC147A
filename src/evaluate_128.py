"""
Evaluation script for BCI Competition IV 2a — 128 Hz / ERS pipeline.

Identical to src/evaluate.py except:
  - Checkpoint suffix:  …_subject_dependent_128_best.pt  (matches train_128.py)
  - Default data path:  data/processed/bci_competition_iv_2a_128/
  - Default split config: configs/data_splits_128.json
  - Normalization skipped: data is already ERS-normalised

Usage:
    python -m src.evaluate_128 --model alternative_eegnet --mode subject_dependent
"""

import argparse
import datetime
import json
import os
import re

import numpy as np
import torch

from src.train_128 import get_model
from src.data.dataloader import BCIDataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS  = [f"A{i:02d}" for i in range(1, 10)]
LOSO_REPS = 4

_ROOT             = os.path.join(os.path.dirname(__file__), "..")
_DEFAULT_DATA_PATH = os.path.join(_ROOT, "data", "processed", "bci_competition_iv_2a_128")
_DEFAULT_SPLIT_CFG = os.path.join(_ROOT, "configs", "data_splits_128.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train_128.py first for this model/subject."
        )
    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_one(model: torch.nn.Module, loader: BCIDataLoader, device: torch.device) -> float:
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            preds    = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total


# ---------------------------------------------------------------------------
# Subject-dependent evaluation
# ---------------------------------------------------------------------------

def evaluate_subject_dependent(
    model_name:  str,
    device:      torch.device,
    data_path:   str,
    split_config: str,
) -> dict:
    results = {}

    print(f"\nSubject-dependent evaluation (128 Hz): {model_name}")
    print(f"{'Subject':>9}  {'Train Acc':>9}  {'Val Acc':>7}  {'Test Acc':>8}")
    print("-" * 42)

    loader_kwargs = dict(
        mode="subject_dependent",
        batch_size=64,
        shuffle=False,
        data_path=data_path,
        split_config=split_config,
    )

    for subject in SUBJECTS:
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_{subject}_subject_dependent_128_best.pt",
        )

        try:
            model = load_model(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"{subject:>9}  MISSING CHECKPOINT")
            print(f"           {e}")
            continue

        # No normalizer — ERS already applied in alternative_preprocess.py
        train_loader = BCIDataLoader(subject=subject, split="train", **loader_kwargs)
        val_loader   = BCIDataLoader(subject=subject, split="val",   **loader_kwargs)
        test_loader  = BCIDataLoader(subject=subject, split="test",  **loader_kwargs)

        train_acc = evaluate_one(model, train_loader, device)
        val_acc   = evaluate_one(model, val_loader,   device)
        test_acc  = evaluate_one(model, test_loader,  device)

        results[subject] = {
            "train_acc": train_acc,
            "val_acc":   val_acc,
            "test_acc":  test_acc,
        }
        print(f"{subject:>9} {train_acc:>8.2%}  {val_acc:>6.2%}  {test_acc:>7.2%}")

    if results:
        train_accs = [v["train_acc"] for v in results.values()]
        val_accs   = [v["val_acc"]   for v in results.values()]
        test_accs  = [v["test_acc"]  for v in results.values()]
        results["mean"] = {
            "train_acc": float(np.mean(train_accs)),
            "val_acc":   float(np.mean(val_accs)),
            "test_acc":  float(np.mean(test_accs)),
        }
        results["std"] = {
            "train_acc": float(np.std(train_accs)),
            "val_acc":   float(np.std(val_accs)),
            "test_acc":  float(np.std(test_accs)),
        }
        print("-" * 42)
        print(f"{'Mean':>9} {results['mean']['train_acc']:>8.2%}  "
              f"{results['mean']['val_acc']:>6.2%}  {results['mean']['test_acc']:>7.2%}")
        print(f"{'Std':>9} {results['std']['train_acc']:>8.2%}  "
              f"{results['std']['val_acc']:>6.2%}  {results['std']['test_acc']:>7.2%}")

    return results


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def evaluate_loso(
    model_name:  str,
    device:      torch.device,
    data_path:   str,
    split_config: str,
) -> dict:
    with open(split_config) as f:
        config = json.load(f)

    fold_keys = []
    for key in config["loso"].keys():
        match = re.search(r"_rep(\d+)$", key)
        if match and int(match.group(1)) < LOSO_REPS:
            fold_keys.append(key)
    fold_keys.sort()

    results = {}

    print(f"\nLOSO evaluation (128 Hz): {model_name}")
    print(f"Using {LOSO_REPS} permutations per subject ({len(fold_keys)} folds total)")
    print(f"{'Fold':>12}  {'Test Acc':>8}")
    print("-" * 24)

    loader_kwargs = dict(
        mode="loso",
        batch_size=64,
        shuffle=False,
        data_path=data_path,
        split_config=split_config,
    )

    for fold in fold_keys:
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_{fold}_loso_128_best.pt",
        )

        try:
            model = load_model(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"{fold:>12}  MISSING CHECKPOINT")
            print(f"               {e}")
            continue

        # No normalizer — ERS already applied in alternative_preprocess.py
        test_loader = BCIDataLoader(fold=fold, split="test", **loader_kwargs)

        acc = evaluate_one(model, test_loader, device)
        results[fold] = acc
        print(f"{fold:>12} {acc:>7.2%}")

    if results:
        accs = list(results.values())
        results["mean"] = float(np.mean(accs))
        results["std"]  = float(np.std(accs))
        print("-" * 24)
        print(f"{'Mean':>12} {results['mean']:>7.2%}")
        print(f"{'Std':>12} {results['std']:>7.2%}")

    return results


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: dict, model_name: str, mode: str) -> None:
    out_dir = os.path.join("experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Pull training hyperparams from the first available training results JSON.
    results_dir = out_dir
    training_config = None
    if mode == "subject_dependent":
        candidates = [
            os.path.join(results_dir, f"{model_name}_{s}_subject_dependent_128.json")
            for s in SUBJECTS
        ]
    else:
        candidates = [
            os.path.join(results_dir, f"{model_name}_{s}_rep0_loso_128.json")
            for s in SUBJECTS
        ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                training_config = json.load(f).get("config")
            break

    payload = {"training_config": training_config, **results}

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{model_name}_{mode}_128_eval_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models trained with the 128 Hz / ERS pipeline"
    )

    parser.add_argument("--data_path",    type=str, default=_DEFAULT_DATA_PATH)
    parser.add_argument("--split_config", type=str, default=_DEFAULT_SPLIT_CFG)

    parser.add_argument("--model", type=str, required=True,
                        choices=["dummy", "eegnet", "alternative_eegnet",
                                 "alternative_eegnet_250", "cnn_lstm",
                                 "cnn_lstm_alternative", "tcn", "lstm",
                                 "cnn_gru", "cnn_gru_alternative", "transformer",
                                 "lstm_alternative"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["subject_dependent", "loso"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.data_path}")
    print(f"Config: {args.split_config}")

    if args.mode == "subject_dependent":
        results = evaluate_subject_dependent(
            args.model, device, args.data_path, args.split_config
        )
    else:
        results = evaluate_loso(
            args.model, device, args.data_path, args.split_config
        )

    save_results(results, args.model, args.mode)


if __name__ == "__main__":
    main()
