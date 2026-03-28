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

MODELS_DIR = Path("models")
TEST_SIZE   = 0.2   # 20% held out for predict
CV_FOLDS    = 5
RANDOM_SEED = 42
MIN_EPOCHS  = 20    # minimum epochs needed for a meaningful train/test split


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
    # 80/20 stratified split — test set is never touched during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_pipeline(n_components=n_components)

    # Cross-validate on train set only
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    # Fit on full train set
    pipeline.fit(X_train, y_train)
    test_score = float(pipeline.score(X_test, y_test))

    # Save pipeline + test split together so predict can reuse the same held-out set
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
    Simulate real-time epoch classification with a delay between each epoch.
    Prints per-epoch result and final accuracy.

    Parameters
    ----------
    pipeline : fitted Pipeline
    X_test   : (n_epochs, n_channels, n_times)
    y_test   : (n_epochs,) ground-truth labels
    delay    : seconds to wait between epochs (default 0.25s, max allowed 2s)

    Returns
    -------
    accuracy : float
    """
    print("epoch nb: [prediction] [truth] equal?")
    correct = 0

    for i, (epoch, truth) in enumerate(zip(X_test, y_test)):
        pred = int(pipeline.predict(epoch[np.newaxis, ...])[0])
        match = pred == int(truth)
        correct += match
        print(f"epoch {i:02d}: [{pred}] [{int(truth)}] {match}")
        time.sleep(delay)

    accuracy = correct / len(y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
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
