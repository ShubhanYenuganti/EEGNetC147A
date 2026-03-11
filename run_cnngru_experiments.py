"""
Run all CNN+GRU training and evaluation experiments (final R3 config).
"""

import json
import subprocess

SUBJECTS = [f"A0{i}" for i in range(1, 10)]

LR = "0.001"
WEIGHT_DECAY = "5e-4"
EPOCHS = "300"
BATCH_SIZE = "16"


def load_loso_fold_keys(path="configs/data_splits.json"):
    with open(path, "r") as f:
        splits = json.load(f)

    # Adjust this if your JSON structure is slightly different
    if "loso" in splits:
        loso_obj = splits["loso"]
        if isinstance(loso_obj, dict):
            return list(loso_obj.keys())

    raise ValueError("Could not find LOSO fold keys in configs/data_splits.json")


# ── Subject-dependent training ───────────────────────────────────────────
print("=" * 60)
print(" CNN+GRU (final R3): Subject-Dependent Training")
print(f" lr={LR}  wd={WEIGHT_DECAY}  epochs={EPOCHS}  bs={BATCH_SIZE}")
print("=" * 60)

for subj in SUBJECTS:
    for fold in range(4):
        print(f"\n>>> Training subject {subj}, fold {fold}...")
        subprocess.run([
            "python", "-m", "src.train",
            "--model", "cnn_gru",
            "--mode", "subject_dependent",
            "--subject", subj,
            "--fold", str(fold),
            "--epochs", EPOCHS,
            "--lr", LR,
            "--weight_decay", WEIGHT_DECAY,
            "--batch_size", BATCH_SIZE,
        ])


# ── LOSO training ────────────────────────────────────────────────────────
print("=" * 60)
print(" CNN+GRU (final R3): LOSO Training")
print(f" lr={LR}  wd={WEIGHT_DECAY}  epochs={EPOCHS}  bs={BATCH_SIZE}")
print("=" * 60)

loso_fold_keys = load_loso_fold_keys()

for fold_key in loso_fold_keys:
    print(f"\n>>> Training LOSO fold {fold_key}...")
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "cnn_gru",
        "--mode", "loso",
        "--fold_key", fold_key,
        "--epochs", EPOCHS,
        "--lr", LR,
        "--weight_decay", WEIGHT_DECAY,
        "--batch_size", BATCH_SIZE,
    ])


# ── Evaluation ───────────────────────────────────────────────────────────
print("=" * 60)
print(" CNN+GRU (final R3): Evaluation")
print("=" * 60)

print("\n>>> Evaluating subject-dependent...")
subprocess.run([
    "python", "-m", "src.evaluate",
    "--model", "cnn_gru",
    "--mode", "subject_dependent",
])

print("\n>>> Evaluating LOSO...")
subprocess.run([
    "python", "-m", "src.evaluate",
    "--model", "cnn_gru",
    "--mode", "loso",
])

print("\n" + "=" * 60)
print(" All CNN+GRU experiments complete!")
print("=" * 60)