#!/usr/bin/env python3
import sys
import mne
import numpy as np
from pathlib import Path
from mne.datasets import eegbci
from numpy import ndarray

# Channels over motor cortex
CHANNELS = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
LRW_RUNS = {3, 4, 7, 8, 11, 12}   # left/right hand
WF_RUNS  = {5, 6, 9, 10, 13, 14}  # both hands/feet

processed_folder = Path("./processed_data")


def preprocess_raw(raw: mne.io.BaseRaw, l_freq: float = 8., h_freq: float = 30.) -> mne.io.BaseRaw:
    """Full preprocessing of raw EEG: standardize names, filter, reference."""
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw.pick("eeg")
    raw.set_eeg_reference("average", projection=True)
    raw.filter(l_freq, h_freq, fir_design="firwin")
    return raw


def load_file(path: Path, runs: set = LRW_RUNS) -> tuple[ndarray | None, ndarray | None]:
    """Load a single EDF file, preprocess, epoch, return (X, y) or (None, None)."""
    filename = path.stem
    try:
        run_num = int(filename.split("R")[-1])
    except ValueError:
        return None, None

    if run_num not in runs:
        return None, None

    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw = preprocess_raw(raw)

        events, _ = mne.events_from_annotations(raw, verbose=False)
        event_map = {"T1": 1, "T2": 2}

        epochs = mne.Epochs(
            raw, events,
            event_id=event_map,
            tmin=0.0, tmax=2.0,
            baseline=None,
            preload=True,
            picks=CHANNELS,
            verbose=False
        )
        if len(epochs) == 0:
            return None, None

        processed_folder.mkdir(exist_ok=True)
        epochs.save(processed_folder / f"{filename}-epo.fif", overwrite=True, verbose=False)

        X = epochs.get_data()        # (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]     # labels 1 or 2
        return X, y

    except Exception as e:
        print(f"  [WARN] Skipping {path.name}: {e}")
        return None, None


def load_subject(subject_dir: Path, runs: set = LRW_RUNS) -> tuple[ndarray | None, ndarray | None]:
    """
    Load and concatenate all valid EDF runs for a single subject directory.

    Parameters
    ----------
    subject_dir : Path   e.g. data/.../S001/
    runs        : set    which run numbers to include (default: LRW_RUNS)

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)  or None
    y : ndarray, shape (n_epochs,)                      or None
    """
    X_list, y_list = [], []
    for edf in sorted(subject_dir.glob("*.edf")):
        X, y = load_file(edf, runs=runs)
        if X is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return None, None

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def load_all_subjects(
    data_dir: Path,
    runs: set = LRW_RUNS,
    verbose: bool = True
) -> tuple[ndarray, ndarray, list[str]]:
    """
    Iterate over all S001..S109 subject folders under data_dir.

    Parameters
    ----------
    data_dir : Path   root folder containing S001/, S002/, ... S109/
    runs     : set    run numbers to include
    verbose  : bool   print per-subject summary

    Returns
    -------
    X        : ndarray, shape (total_epochs, n_channels, n_times)
    y        : ndarray, shape (total_epochs,)
    subjects : list[str]  subject IDs successfully loaded (e.g. ["S001", "S003", ...])
    """
    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("S"))

    X_all, y_all, subject_ids = [], [], []

    for subj_dir in subject_dirs:
        X, y = load_subject(subj_dir, runs=runs)
        if X is None:
            if verbose:
                print(f"  [SKIP] {subj_dir.name}: no valid epochs")
            continue

        X_all.append(X)
        y_all.append(y)
        subject_ids.append(subj_dir.name)

        if verbose:
            counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
            print(f"  {subj_dir.name}: {len(y):>4} epochs  shape={X.shape}  labels={counts}")

    if not X_all:
        raise RuntimeError(f"No valid data found under {data_dir}")

    X_out = np.concatenate(X_all, axis=0)
    y_out = np.concatenate(y_all, axis=0)
    print(f"\nTotal: {len(subject_ids)} subjects, {len(y_out)} epochs, shape={X_out.shape}")
    return X_out, y_out, subject_ids


def check_csp_quality(X: ndarray, y: ndarray, reg: float = 1e-4) -> None:
    """
    Print eigenvalue spread as a quick CSP quality diagnostic.
    Good data: eigenvalues spread from ~0.1 to ~0.9
    Bad data:  eigenvalues all cluster near 0.5
    """
    from scipy.linalg import eigh
    classes = np.unique(y)

    def norm_cov(epochs: ndarray) -> ndarray:
        covs = [e @ e.T for e in epochs]
        return np.mean([c / np.trace(c) for c in covs], axis=0)

    cov1 = norm_cov(X[y == classes[0]])
    cov2 = norm_cov(X[y == classes[1]])
    cov_total = cov1 + cov2 + reg * np.eye(X.shape[1])

    ev, _ = eigh(cov1, cov_total)
    spread = ev.max() - ev.min()
    print(f"CSP eigenvalue spread: {spread:.4f}  (want > 0.3)")
    print(f"Eigenvalues: {np.round(ev, 4)}")


# ---------------------------------------------------------------------------
# Legacy single-folder entry point (backwards compatible)
# ---------------------------------------------------------------------------

def main(path: Path, runs: set = LRW_RUNS) -> tuple[ndarray | None, ndarray | None]:
    """Load all EDF files in a single folder (one subject). Kept for compatibility."""
    X, y = load_subject(path, runs=runs)
    if X is None:
        print("No valid epochs found in the folder")
        return None, None
    print(f"Total epochs: {len(y)}, X shape: {X.shape}")
    return X, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG preprocessing for Total Perspective Vortex")
    parser.add_argument("path", type=Path, help="Subject folder (single) or dataset root (--all)")
    parser.add_argument("--all", action="store_true", help="Process all subjects under path")
    parser.add_argument("--check", action="store_true", help="Run CSP quality diagnostic after loading")
    parser.add_argument("--runs", choices=["lrw", "wf"], default="lrw",
                        help="Run type: lrw=left/right hand (default), wf=both hands/feet")
    args = parser.parse_args()

    runs = LRW_RUNS if args.runs == "lrw" else WF_RUNS

    if args.all:
        X, y, subjects = load_all_subjects(args.path, runs=runs)
    else:
        X, y = main(args.path, runs=runs)

    if args.check and X is not None:
        print()
        check_csp_quality(X, y)
