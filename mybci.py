#!/usr/bin/env python3
"""
mybci.py — Main CLI entry point for the BCI pipeline.

Usage:
    python mybci.py <subject> <run> train    # train & save model
    python mybci.py <subject> <run> predict  # stream predict on held-out test
    python mybci.py                          # evaluate all subjects x experiments
"""
import sys
import numpy as np
from pathlib import Path

from preprocess import load_subject, LRW_RUNS, WF_RUNS
from pipeline import train, load, predict_stream, evaluate_subject, MODELS_DIR

DATA_DIR = Path("data/physionet.org/files/eegmmidb/1.0.0")

# Maps experiment index (0-5) to the run numbers for that experiment type
# Mirrors the physionet dataset structure:
#   exp 0,1 = left/right hand real (R03/04)     exp 2,3 = left/right hand imagined (R07/08, R11/12)
#   exp 4,5 = both hands/feet real (R05/06)     ...
EXPERIMENT_RUNS: dict[int, set[int]] = {
    0: {3, 4},    # real left/right hand
    1: {7, 8},    # imagined left/right hand (first)
    2: {11, 12},  # imagined left/right hand (second)
    3: {5, 6},    # real both hands / feet
    4: {9, 10},   # imagined both hands / feet (first)
    5: {13, 14},  # imagined both hands / feet (second)
}


def model_path(subject: int, run: int) -> Path:
    return MODELS_DIR / f"S{subject:03d}_R{run:02d}.pkl"


def cmd_train(subject: int, run: int) -> None:
    subj_dir = DATA_DIR / f"S{subject:03d}"
    runs = {run}

    print(f"Loading S{subject:03d} run {run:02d}...")
    X, y = load_subject(subj_dir, runs=runs, require_balance=False)
    if X is None or y is None:
        print(f"[ERROR] No data found for subject {subject} run {run}")
        sys.exit(1)

    print(f"Epochs: {len(y)}, shape: {X.shape}")

    save_path = model_path(subject, run)
    pipeline, cv_scores, test_score = train(X, y, save_path)

    print(f"{np.round(cv_scores, 4).tolist()}")
    print(f"cross_val_score: {cv_scores.mean():.4f}")
    print(f"Test score (held-out 20%): {test_score:.4f}")
    print(f"Model saved to: {save_path}")


def cmd_predict(subject: int, run: int) -> None:
    save_path = model_path(subject, run)
    if not save_path.exists():
        print(f"[ERROR] No trained model found at {save_path}")
        print(f"  Run: python mybci.py {subject} {run} train")
        sys.exit(1)

    pipeline, X_test, y_test = load(save_path)
    predict_stream(pipeline, X_test, y_test)


def cmd_evaluate_all() -> None:
    """
    Evaluate all 109 subjects across all 6 experiment types.
    Prints per-subject accuracy and mean per experiment.
    """
    subject_dirs = sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("S")
    )

    exp_accuracies: dict[int, list[float]] = {i: [] for i in range(6)}

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        for exp_idx, runs in EXPERIMENT_RUNS.items():
            X, y = load_subject(subj_dir, runs=runs, require_balance=False)
            if X is None or y is None:
                continue
            if len(np.unique(y)) < 2:
                continue

            acc = evaluate_subject(X, y)
            exp_accuracies[exp_idx].append(acc)
            print(f"experiment {exp_idx}: subject {subject_id}: accuracy = {acc:.4f}")

    print("\nMean accuracy of the six different experiments for all subjects:")
    mean_all = []
    for exp_idx, accs in exp_accuracies.items():
        if accs:
            mean_exp = float(np.mean(accs))
            mean_all.append(mean_exp)
            print(f"experiment {exp_idx}: accuracy = {mean_exp:.4f}")

    if mean_all:
        print(f"\nMean accuracy of 6 experiments: {np.mean(mean_all):.4f}")


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 0:
        cmd_evaluate_all()

    elif len(args) == 3:
        subject = int(args[0])
        run     = int(args[1])
        mode    = args[2].lower()

        if mode == "train":
            cmd_train(subject, run)
        elif mode == "predict":
            cmd_predict(subject, run)
        else:
            print(f"[ERROR] Unknown mode '{mode}'. Use 'train' or 'predict'.")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python mybci.py <subject> <run> train")
        print("  python mybci.py <subject> <run> predict")
        print("  python mybci.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
