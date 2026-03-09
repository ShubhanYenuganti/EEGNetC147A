"""
Run all CNN+GRU training and evaluation experiments (final R3 config).

Hyperparameters (from 3-round grid search):
  lr=0.001, wd=5e-4, epochs=300
  gru_hidden=48, gru_layers=2
  cnn_dropout=0.4, fc_dropout=0.5
  batch_size=16

Winner config achieved 64.34% mean val across A02/A03/A07.

Usage:
    python run_cnn_gru.py
"""

import subprocess

SUBJECTS = [f"A0{i}" for i in range(1, 10)]

# ── Tuned hyperparameters (R3 winner) ────────────────────────────────────
LR           = "0.001"
WEIGHT_DECAY = "5e-4"
EPOCHS       = "300"
BATCH_SIZE   = "16"

# ── Subject-dependent training ───────────────────────────────────────────
print("=" * 60)
print(" CNN+GRU (final R3): Subject-Dependent Training")
print(f" lr={LR}  wd={WEIGHT_DECAY}  epochs={EPOCHS}  bs={BATCH_SIZE}")
print("=" * 60)

for subj in SUBJECTS:
    print(f"\n>>> Training subject {subj}...")
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "cnn_gru",
        "--mode", "subject_dependent",
        "--subject", subj,
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

for fold in range(9):
    print(f"\n>>> Training LOSO fold {fold}...")
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "cnn_gru",
        "--mode", "loso",
        "--fold", str(fold),
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