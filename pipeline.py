#!/usr/bin/env python3
"""
pipeline.py — Core BCI pipeline: build, train, evaluate, predict stream.
"""
import time
import joblib
import numpy as np
from pathlib import Path
from numpy import ndarray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from csp import CSP
from preprocess import EPOCH_TMIN, EPOCH_TMAX, SFREQ, CHANNELS, preprocess_raw

MODELS_DIR  = Path("models")
TEST_SIZE   = 0.2
CV_FOLDS    = 5
RANDOM_SEED = 42
MIN_EPOCHS  = 20
MAX_LATENCY = 2.0   # seconds — hard limit per subject specification


def build_pipeline(n_components: int = 4) -> Pipeline:
    """
    Build the CSP → StandardScaler → LDA pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP spatial filters (default 4: 2 bottom + 2 top eigenvalues)

    Returns
    -------
    sklearn Pipeline ready for .fit() / .predict()
    """
    return Pipeline([
        ("csp",    CSP(n_components=n_components)),
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis()),
    ])


def _check_data(X: ndarray, y: ndarray, label: str = "") -> bool:
    tag = f" {label}" if label else ""
    if len(y) < MIN_EPOCHS:
        print(f"  [SKIP]{tag}: only {len(y)} epochs (need >= {MIN_EPOCHS})")
        return False
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        print(f"  [SKIP]{tag}: only 1 class in data")
        return False
    if counts.min() < CV_FOLDS:
        print(f"  [SKIP]{tag}: minority class has {counts.min()} epochs (need >= {CV_FOLDS})")
        return False
    return True


def train(
    X: ndarray,
    y: ndarray,
    save_path: Path,
    n_components: int = 4,
) -> tuple[Pipeline, ndarray, float] | None:
    """
    Split data, cross-validate, fit on full train set, save model.

    Parameters
    ----------
    X          : (n_epochs, n_channels, n_times)
    y          : (n_epochs,)
    save_path  : where to serialize the fitted pipeline
    n_components: CSP filter count

    Returns
    -------
    pipeline   : fitted Pipeline (trained on 80% of data)
    cv_scores  : array of per-fold accuracy scores
    test_score : accuracy on the held-out 20% test set
    """
    if not _check_data(X, y):
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_pipeline(n_components=n_components)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    pipeline.fit(X_train, y_train)
    test_score = float(pipeline.score(X_test, y_test))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "X_test": X_test, "y_test": y_test}, save_path)

    return pipeline, cv_scores, test_score


def load(save_path: Path) -> tuple[Pipeline, ndarray, ndarray]:
    """
    Load a saved pipeline and its held-out test set.

    Returns
    -------
    pipeline : fitted Pipeline
    X_test   : held-out epochs  (n_epochs, n_channels, n_times)
    y_test   : held-out labels  (n_epochs,)
    """
    data = joblib.load(save_path)
    return data["pipeline"], data["X_test"], data["y_test"]


def predict_stream(
    pipeline: Pipeline,
    X_test: ndarray,
    y_test: ndarray,
    delay: float = 0.25,
) -> float:
    """
    Simulate real-time epoch classification from pre-epoched data.
    Enforces that total per-epoch time stays within MAX_LATENCY.

    Parameters
    ----------
    pipeline : fitted Pipeline
    X_test   : (n_epochs, n_channels, n_times)
    y_test   : (n_epochs,) ground-truth labels
    delay    : seconds to wait between epochs (clamped to [0, MAX_LATENCY])

    Returns
    -------
    accuracy : float
    """
    delay = min(max(delay, 0.0), MAX_LATENCY)
    print("epoch nb: [prediction] [truth] equal?")
    correct = 0

    for i, (epoch, truth) in enumerate(zip(X_test, y_test)):
        t0 = time.perf_counter()
        pred = int(pipeline.predict(epoch[np.newaxis, ...])[0])
        elapsed = time.perf_counter() - t0

        match = pred == int(truth)
        correct += match
        print(f"epoch {i:02d}: [{pred}] [{int(truth)}] {match}  (inference: {elapsed*1000:.2f}ms)")

        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    accuracy = correct / len(y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    return accuracy


def predict_raw_stream(
    pipeline: Pipeline,
    raw_path: Path,
    delay: float = 0.25,
) -> float | None:
    """
    Real-time simulation directly from a raw EDF file.
    Mimics a live BCI loop: for each event onset, cut a window of EPOCH_TMAX seconds,
    preprocess on-the-fly, and predict. Measures and enforces MAX_LATENCY.

    Parameters
    ----------
    pipeline : fitted Pipeline (must have been trained with same EPOCH_TMAX)
    raw_path : path to an EDF file
    delay    : seconds to wait between epochs (clamped to [0, MAX_LATENCY])

    Returns
    -------
    accuracy : float, or None if no valid events found
    """
    import mne

    if not raw_path.exists():
        print(f"[ERROR] File not found: {raw_path}")
        return None

    delay = min(max(delay, 0.0), MAX_LATENCY)
    n_times = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ) + 1  # expected timepoints per window

    try:
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        raw = preprocess_raw(raw)
        raw.pick(CHANNELS)
    except Exception as e:
        print(f"[ERROR] Could not load {raw_path.name}: {e}")
        return None

    events, _ = mne.events_from_annotations(raw, verbose=False)
    valid_events = [(sample, code) for sample, _, code in events if code in (1, 2)]

    if not valid_events:
        print(f"[WARN] No T1/T2 events found in {raw_path.name}")
        return None

    data = raw.get_data()  # (n_channels, n_total_times)
    total_samples = data.shape[1]

    print(f"Streaming {len(valid_events)} epochs from {raw_path.name}")
    print(f"Window: {EPOCH_TMIN}-{EPOCH_TMAX}s  ({n_times} samples @ {SFREQ}Hz)")
    print(f"epoch nb: [prediction] [truth] equal?  (latency budget: {MAX_LATENCY}s)")

    correct = 0
    skipped = 0

    for i, (onset_sample, code) in enumerate(valid_events):
        start = onset_sample + int(EPOCH_TMIN * SFREQ)
        end   = start + n_times

        if end > total_samples:
            print(f"epoch {i:02d}: [SKIP] window exceeds recording length")
            skipped += 1
            continue

        t0 = time.perf_counter()
        window = data[:, start:end]        # (n_channels, n_times)
        X = window[np.newaxis, ...]        # (1, n_channels, n_times)
        pred = int(pipeline.predict(X)[0])
        elapsed = time.perf_counter() - t0

        match = pred == code
        correct += match
        status = "✓" if elapsed <= MAX_LATENCY else f"LATENCY EXCEEDED ({elapsed:.3f}s)"
        print(f"epoch {i:02d}: [{pred}] [{code}] {match}  "
              f"(processing: {elapsed*1000:.2f}ms) {status}")

        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    counted = len(valid_events) - skipped
    if counted == 0:
        print("No epochs could be processed.")
        return None

    accuracy = correct / counted
    print(f"\nAccuracy: {accuracy:.4f}  ({correct}/{counted} correct, {skipped} skipped)")
    return accuracy


def evaluate_subject(X, y, label: str = "", n_components: int = 4) -> float | None:
    """
    Quick single-subject evaluation: fit on 80%, score on 20%.
    Used in the full all-subjects loop — no model is saved.

    Returns
    -------
    test accuracy : float
    """
    if not _check_data(X, y, label=label):
        return None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )
        pipeline = build_pipeline(n_components=n_components)
        pipeline.fit(X_train, y_train)
        return float(pipeline.score(X_test, y_test))
    except Exception as e:
        print(f"  [SKIP] {label}: {e}")
        return None
