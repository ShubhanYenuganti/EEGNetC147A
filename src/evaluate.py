"""
Evaluation script for MindReader EEG Motor Imagery Classification.

Usage:
    # Subject-dependent
    python -m src.evaluate --model eegnet --mode subject_dependent

    # LOSO
    python -m src.evaluate --model cnn_lstm --mode loso
"""

import argparse
import json
import os

import numpy as np
import torch

from src.data.dataloader import BCIDataLoader
from src.train import get_model


SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
SUBJECT_DEP_FOLDS = [0, 1, 2, 3]
_SPLIT_CONFIG = os.path.join("configs", "data_splits.json")


def load_loso_fold_keys(split_config_path: str = _SPLIT_CONFIG) -> list[str]:
    with open(split_config_path, "r") as f:
        cfg = json.load(f)

    if "loso" not in cfg or not isinstance(cfg["loso"], dict):
        raise ValueError("configs/data_splits.json does not contain a valid 'loso' section")

    return list(cfg["loso"].keys())


def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train.py first for this model/split."
        )

    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_one(
    model: torch.nn.Module,
    loader: BCIDataLoader,
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            logits = model(X_batch)
            preds = logits.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return correct / total


def evaluate_subject_dependent(model_name: str, device: torch.device) -> dict:
    results = {}
    fold_accs = []

    print(f"\nSubject-dependent evaluation: {model_name}")
    print(f"{'Subject':>9}  {'Fold':>4}  {'Test Acc':>8}")
    print("-" * 30)

    for subject in SUBJECTS:
        subject_fold_accs = []

        for fold in SUBJECT_DEP_FOLDS:
            checkpoint_path = os.path.join(
                "experiments",
                "checkpoints",
                f"{model_name}_{subject}_fold{fold}_subject_dependent_best.pt",
            )

            try:
                model = load_model(model_name, checkpoint_path, device)
            except FileNotFoundError as e:
                print(f"{subject:>9}  {fold:>4}  MISSING")
                print(f"                {e}")
                continue

            test_loader = BCIDataLoader(
                mode="subject_dependent",
                subject=subject,
                fold=fold,
                split="test",
                batch_size=64,
                shuffle=False,
            )

            acc = evaluate_one(model, test_loader, device)
            results[f"{subject}_fold{fold}"] = acc
            subject_fold_accs.append(acc)
            fold_accs.append(acc)

            print(f"{subject:>9}  {fold:>4}  {acc:>7.2%}")

        if subject_fold_accs:
            results[f"{subject}_mean"] = float(np.mean(subject_fold_accs))
            results[f"{subject}_std"] = float(np.std(subject_fold_accs))

    if fold_accs:
        results["mean"] = float(np.mean(fold_accs))
        results["std"] = float(np.std(fold_accs))
        print("-" * 30)
        print(f"{'Overall':>9}  {'-':>4}  {results['mean']:>7.2%}")
        print(f"{'Std':>9}  {'-':>4}  {results['std']:>7.2%}")

    return results


def evaluate_loso(model_name: str, device: torch.device) -> dict:
    results = {}
    loso_fold_keys = load_loso_fold_keys()

    print(f"\nLOSO evaluation: {model_name}")
    print(f"{'Fold Key':>12}  {'Test Acc':>8}")
    print("-" * 24)

    accs = []

    for fold_key in loso_fold_keys:
        checkpoint_path = os.path.join(
            "experiments",
            "checkpoints",
            f"{model_name}_{fold_key}_loso_best.pt",
        )

        try:
            model = load_model(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"{fold_key:>12}  MISSING")
            print(f"              {e}")
            continue

        test_loader = BCIDataLoader(
            mode="loso",
            fold_key=fold_key,
            split="test",
            batch_size=64,
            shuffle=False,
        )

        acc = evaluate_one(model, test_loader, device)
        results[fold_key] = acc
        accs.append(acc)

        print(f"{fold_key:>12}  {acc:>7.2%}")

    if accs:
        results["mean"] = float(np.mean(accs))
        results["std"] = float(np.std(accs))
        print("-" * 24)
        print(f"{'Mean':>12}  {results['mean']:>7.2%}")
        print(f"{'Std':>12}  {results['std']:>7.2%}")

    return results


def save_results(results: dict, model_name: str, mode: str) -> None:
    out_dir = os.path.join("experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_{mode}_eval.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on BCI Competition IV 2a")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["dummy", "eegnet", "cnn_lstm", "tcn", "lstm", "cnn_gru", "transformer"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["subject_dependent", "loso"],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "subject_dependent":
        results = evaluate_subject_dependent(args.model, device)
    else:
        results = evaluate_loso(args.model, device)

    save_results(results, args.model, args.mode)


if __name__ == "__main__":
    main()