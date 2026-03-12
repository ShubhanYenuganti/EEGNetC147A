"""
Training script for MindReader EEG Motor Imagery Classification.

Usage:
    # Subject-dependent
    python -m src.train --model eegnet --mode subject_dependent --subject A01

    # LOSO
    python -m src.train --model cnn_lstm --mode loso --fold 0

    # With custom hyperparams
    python -m src.train --model tcn --mode subject_dependent --subject A01 --epochs 150 --lr 0.0005 --batch_size 64
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data.dataloader import BCIDataLoader, Normalizer, TrialNormalizer

# ---------------------------------------------------------------------------
# Model registry — add new models here as they're implemented
# ---------------------------------------------------------------------------
def sliding_window_augment(
    X: torch.Tensor,
    y: torch.Tensor,
    window: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop — preserves input size so models need no changes."""
    max_start = X.shape[-1] - window
    if max_start == 0:
        return X, y
    start = torch.randint(0, max_start + 1, (1,)).item()
    return X[..., start:start + window], y


def get_model(model_name: str, n_channels: int = 22, n_classes: int = 4) -> nn.Module:
    if model_name == "eegnet":
        from src.models.eegnet import EEGNet
        return EEGNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "alternative_eegnet":
        from src.models.alternative_eegnet import EEGNet as AltEEGNet
        return AltEEGNet(n_channels=n_channels, n_classes=n_classes, n_timepoints=256)  # CHANGED: specify n_timepoints for alternative EEGNet
    elif model_name == "alternative_eegnet_250":
        from src.models.alternative_eegnet_250 import EEGNet as AltEEGNet250
        return AltEEGNet250(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        return CNNLSTM(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "tcn":
        from src.models.tcn import TCN
        return TCN(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "lstm":
        from src.models.lstm import LSTM
        return LSTM(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "cnn_gru":
        from src.models.cnn_gru import CNNGRU
        return CNNGRU(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "transformer":
        from src.models.transformer import EEGTransformer
        return EEGTransformer(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "dummy":
        from src.models.dummy import Dummy
        return Dummy(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Options: eegnet, cnn_lstm, tcn, lstm, cnn_gru, transformer"
        )


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    BCIDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    augment: bool = False,
) -> tuple[float, float]:
    """One full pass over the training data.

    If the model exposes an apply_max_norm_() method (e.g. EEGNet), it is
    called after every optimizer step to enforce the depthwise and classifier
    weight-norm constraints described in Lawhern et al. (2018).

    Returns:
        avg_loss: average cross-entropy loss over all batches
        accuracy: fraction of correctly classified trials
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.float().to(device)
        y_batch = y_batch.long().to(device)

        if augment:
            X_batch, y_batch = sliding_window_augment(X_batch, y_batch)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- FIX: apply max-norm constraints after every weight update ---
        # hasattr check keeps this generic — only EEGNet defines the method,
        # so other architectures are unaffected.
        if hasattr(model, "apply_max_norm_"):
            model.apply_max_norm_()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def validate(
    model:     nn.Module,
    loader:    BCIDataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """One full pass over val/test data with no gradient updates.

    Returns:
        avg_loss: average cross-entropy loss
        accuracy: fraction of correctly classified trials
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model:           nn.Module,
    train_loader:    BCIDataLoader,
    val_loader:      BCIDataLoader,
    config:          dict,
    checkpoint_path: str,
    results_path:    str,
) -> dict:
    """Full training run.

    Trains for config['epochs'] epochs, saves the best model by val accuracy,
    and writes per-epoch metrics to a JSON file.

    Returns:
        history: dict with lists of train/val loss and accuracy per epoch
    """
    device = config["device"]
    model  = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "config":     config,
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 40
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path),    exist_ok=True)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            augment=(config["mode"] == "subject_dependent"),
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.1f}s"
        )

        VAL_SMOOTH = 5  # epochs to average over

        history["val_acc"].append(val_acc)

        # only consider checkpointing after enough epochs to smooth
        if len(history["val_acc"]) >= VAL_SMOOTH:
            smoothed_val = sum(history["val_acc"][-VAL_SMOOTH:]) / VAL_SMOOTH
        else:
            smoothed_val = val_acc

        if smoothed_val > best_val_acc:
            best_val_acc = smoothed_val
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint (smoothed_val={smoothed_val:.2%})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience and epoch > 100:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val_loss improvement for {early_stop_patience} epochs)")
                break

    history["best_val_acc"] = best_val_acc
    print(f"\nBest val acc: {best_val_acc:.2%}")
    print(f"Checkpoint:   {checkpoint_path}")

    history["config"] = {
        k: str(v) if isinstance(v, torch.device) else v
        for k, v in config.items()
    }
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results:      {results_path}\n")

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a model on BCI Competition IV 2a")

    parser.add_argument("--split_config", type=str, default="configs/data_splits_TE.json")

 # Model and mode
    parser.add_argument("--model", type=str, required=True,
                        choices=["dummy", "eegnet", "alternative_eegnet", "alternative_eegnet_250",
                                 "cnn_lstm", "tcn", "lstm", "cnn_gru", "transformer"])
    parser.add_argument("--mode",  type=str, required=True,
                        choices=["subject_dependent", "loso"])

    # Subject-dependent args
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject ID for subject_dependent mode (e.g. A01)")

    # LOSO args
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold key for loso mode (e.g. A01_rep0)")

    # Hyperparameters
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Misc
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Validate args
    if args.mode == "subject_dependent" and args.subject is None:
        parser.error("--subject is required for subject_dependent mode")
    if args.mode == "loso" and args.fold is None:
        parser.error("--fold is required for loso mode")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build dataloaders
    # ------------------------------------------------------------------
    loader_kwargs = dict(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subject=args.subject,
        fold=args.fold,
        split_config=args.split_config,
    )
    train_loader = BCIDataLoader(**loader_kwargs, split="train", shuffle=True)
    val_loader   = BCIDataLoader(**loader_kwargs, split="val",   shuffle=False)

    # --- FIX: normalize per-channel using training statistics only ---
    norm = TrialNormalizer()  # CHANGED: use training-set normalizer instead of TrialNormalizer
    norm.apply_(train_loader.dataset)
    norm.apply_(val_loader.dataset)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model    = get_model(args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    if args.mode == "subject_dependent":
        run_id = f"{args.model}_{args.subject}_subject_dependent"
    else:
        run_id = f"{args.model}_{args.fold}_loso"

    checkpoint_path = os.path.join("experiments", "checkpoints", f"{run_id}_best.pt")
    results_path    = os.path.join("experiments", "results",      f"{run_id}.json")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = {
        "model":        args.model,
        "mode":         args.mode,
        "subject":      args.subject,
        "fold":         args.fold,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "weight_decay": args.weight_decay,
        "device":       device,
    }

    train(model, train_loader, val_loader, config, checkpoint_path, results_path)


if __name__ == "__main__":
    main()
