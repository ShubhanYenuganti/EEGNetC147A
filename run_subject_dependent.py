"""
Run subject-dependent training for all 9 subjects.
Usage: python run_subject_dependent.py --model alternative_eegnet_250 [--epochs 300] [--lr 0.001] [--weight_decay 1e-4] [--batch_size 32]
"""
import argparse
import subprocess

SUBJECTS = [f"A0{i}" for i in range(1, 10)]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model name (e.g. alternative_eegnet_250, cnn_gru, eegnet, cnn_lstm, transformer)")
parser.add_argument("--epochs", default="300")
parser.add_argument("--lr", default="0.001")
parser.add_argument("--weight_decay", default="1e-4")
parser.add_argument("--batch_size", default="32")
args = parser.parse_args()

print(f"Subject-dependent training: {args.model} | epochs={args.epochs} lr={args.lr} wd={args.weight_decay} bs={args.batch_size}")

for subj in SUBJECTS:
    print(f"\n[subject_dependent] {args.model} — {subj}")
    subprocess.run([
        "python", "-m", "src.train",
        "--model", args.model,
        "--mode", "subject_dependent",
        "--subject", subj,
        "--epochs", args.epochs,
        "--lr", args.lr,
        "--weight_decay", args.weight_decay,
        "--batch_size", args.batch_size,
    ])
