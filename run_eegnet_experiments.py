import json
import subprocess

SUBJECTS = [f"A0{i}" for i in range(1, 10)]

# Subject-dependent
for subj in SUBJECTS:
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "eegnet",
        "--mode", "subject_dependent",
        "--subject", subj,
        "--epochs", "300",
    ])

# LOSO
with open("configs/data_splits_TE.json") as f:
    _config = json.load(f)
loso_fold_keys = list(_config["loso"].keys())

for fold_key in loso_fold_keys:
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "eegnet",
        "--mode", "loso",
        "--fold", fold_key,
        "--epochs", "300",
    ])