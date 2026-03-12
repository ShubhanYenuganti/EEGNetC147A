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

import torch
import numpy as np

from src.train import get_model
from src.data.dataloader import BCIDataLoader, TrialNormalizer, Normalizer  # CHANGED: import Normalizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
N_FOLDS  = 9

# ---------------------------------------------------------------------------
# Load model - from train.py
# ---------------------------------------------------------------------------

def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate model architecture and load saved weights from checkpoint.

    Args:
        model_name:      name of the architecture (must match get_model registry)
        checkpoint_path: path to the .pt file saved by train.py
        device:          device to load the model onto

    Returns:
        model in eval mode with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train.py first for this model/subject/fold."
        )
    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def evaluate_one(
    model: torch.nn.Module,
    loader: BCIDataLoader,
    device: torch.device,
) -> float:
    """One full pass on a single dataloader.
    
    Args:
        model:  trained model in eval mode
        loader: dataloader for the split to evaluate on
        device: device to run inference on

    Returns:
        accuracy as a float between 0 and 1

    """
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
    
    return correct/total

# ---------------------------------------------------------------------------
# Subject-dependent evaluation
# ---------------------------------------------------------------------------

def evaluate_subject_dependent(model_name: str, device: torch.device, split_config: str) -> dict:
    """Evaluate model on test split for all 9 subjects independently.

    For each subject, loads the checkpoint saved by train.py and runs
    inference on that subject's held-out test split.

    Returns:
        dict with per-subject accuracies, mean, and std
    """
    results = {}

    print(f"\nSubject-dependent evaluation: {model_name}")
    print(f"{'Subject':>9}  {'Test Acc':>8}")
    print("-" * 22)

    for subject in SUBJECTS:
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_{subject}_subject_dependent_best.pt"
        )

        try:
            model = load_model(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"{subject:>9}  MISSING CHECKPOINT")
            print(f"           {e}")
            continue

        train_loader = BCIDataLoader(  # CHANGED: rebuild train split to recover training normalization stats
            mode="subject_dependent",  # CHANGED
            subject=subject,  # CHANGED
            split="train",  # CHANGED
            batch_size=64,  # CHANGED
            shuffle=False,  # CHANGED
            split_config=split_config
        )  # CHANGED
        norm = Normalizer()  # CHANGED: use train-fitted normalizer
        norm.fit(train_loader.dataset.X)  # CHANGED: fit on training data only

        test_loader = BCIDataLoader(
            mode="subject_dependent",
            subject=subject,
            split="test",
            batch_size=64,
            shuffle=False,
        )
        norm.apply_(test_loader.dataset)  # CHANGED: apply train-fitted stats to test split

        acc = evaluate_one(model, test_loader, device)
        results[subject] = acc
        print(f"{subject:>9} {acc:>7.2%}")
    
    if results:
        accs = list(results.values())
        results["mean"] = float(np.mean(accs))
        results["stds"] = float(np.std(accs))
        print("-" * 22)
        print(f"{'Mean':>9} {results['mean']:>7.2%}")
        print(f"{'Std':>9} {results['stds']:>7.2%}")  # CHANGED: print std, not mean
    
    return results

# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def evaluate_loso(model_name: str, device: torch.device) -> dict:
    """Evaluate model on test split for all 9 LOSO folds.

    For each fold, loads the checkpoint saved by train.py and runs
    inference on that fold's held-out test subject.

    Returns:
        dict with per-fold accuracies, mean, and std
    """
    results = {}

    print(f"\nLOSO evaluation: {model_name}")
    print(f"{'Fold':>6}  {'Test Acc':>8}")
    print("-" * 18)

    for fold in range(N_FOLDS):
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_fold{fold}_loso_best.pt"
        )

        try:
            model = load_model(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"{fold:>6}  MISSING CHECKPOINT")
            print(f"           {e}")
            continue

        train_loader = BCIDataLoader(  # CHANGED: rebuild LOSO train split to recover training normalization stats
            mode="loso",  # CHANGED
            fold=fold,  # CHANGED
            split="train",  # CHANGED
            batch_size=64,  # CHANGED
            shuffle=False,  # CHANGED
        )  # CHANGED
        norm = Normalizer()  # CHANGED: use train-fitted normalizer
        norm.fit(train_loader.dataset.X)  # CHANGED: fit on training data only

        test_loader = BCIDataLoader(
            mode="loso",
            fold=fold,
            split="test",
            batch_size=64,
            shuffle=False,
        )
        norm.apply_(test_loader.dataset)  # CHANGED: apply train-fitted stats to test split

        acc = evaluate_one(model, test_loader, device)
        results[f"fold_{fold}"] = acc
        print(f"{fold:>6} {acc:>7.2%}")
    
    if results:
        accs = list(results.values())
        results["mean"] = float(np.mean(accs))
        results["stds"] = float(np.std(accs))
        print("-" * 22)
        print(f"{'Mean':>9} {results['mean']:>7.2%}")
        print(f"{'Std':>9} {results['stds']:>7.2%}")
    
    return results

# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: dict, model_name: str, mode: str) -> None:
    """Save evaluation results to JSON."""
    out_dir  = os.path.join("experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_{mode}_eval.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a model on BCI Competition IV 2a")

    parser.add_argument("--split_config", type=str, default="configs/data_splits.json")
# Model and mode
    parser.add_argument("--model",   type=str, required=True,
                        choices=["dummy", "eegnet", "alternative_eegnet", "alternative_eegnet_250",
                                 "cnn_lstm", "tcn", "lstm", "cnn_gru", "transformer"])
    parser.add_argument("--mode",    type=str, required=True,
                        choices=["subject_dependent", "loso"])

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "subject_dependent":
        results = evaluate_subject_dependent(args.model, device, args.split_config)
    else:
        results = evaluate_loso(args.model, device)

    save_results(results, args.model, args.mode)
    

if __name__ == "__main__":
    main()
