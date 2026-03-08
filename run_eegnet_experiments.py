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
for fold in range(9):
    subprocess.run([
        "python", "-m", "src.train",
        "--model", "eegnet",
        "--mode", "loso",
        "--fold", str(fold),
        "--epochs", "300",
    ])