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

MODELS_DIR  = Path("models")
TEST_SIZE   = 0.2
CV_FOLDS    = 5
RANDOM_SEED = 42
MIN_EPOCHS  = 20   # need at least this many epochs to get a meaningful split


def build_pipeline(n_components: int = 4) -> Pipeline:
    return Pipeline([
        ("csp",    CSP(n_components=n_components)),
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis()),
    ])


def _check_data(X: ndarray, y: ndarray, label: str = "") -> bool:
    """
    Return False (and print a warning) if the data is too small to train on.
    Checks:
      - total epoch count >= MIN_EPOCHS
      - both classes present in train split (stratify guarantees this if counts >= CV_FOLDS)
    """
    if len(y) < MIN_EPOCHS:
        print(f"  [SKIP]{' ' + label if label else ''}: only {len(y)} epochs (need >= {MIN_EPOCHS})")
        return False
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        print(f"  [SKIP]{' ' + label if label else ''}: only 1 class in data")
        return False
    if counts.min() < CV_FOLDS:
        print(f"  [SKIP]{' ' + label if label else ''}: minority class has {counts.min()} epochs (need >= {CV_FOLDS})")
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
    Returns None if data does not meet minimum requirements.
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
    data = joblib.load(save_path)
    return data["pipeline"], data["X_test"], data["y_test"]


def predict_stream(
    pipeline: Pipeline,
    X_test: ndarray,
    y_test: ndarray,
    delay: float = 0.25,
) -> float:
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


def evaluate_subject(
    X: ndarray,
    y: ndarray,
    label: str = "",
    n_components: int = 4,
) -> float | None:
    """
    Quick single-subject evaluation: fit on 80%, score on 20%.
    Returns None if data does not meet minimum requirements.
    """
    if not _check_data(X, y, label=label):
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    pipeline = build_pipeline(n_components=n_components)
    pipeline.fit(X_train, y_train)
    return float(pipeline.score(X_test, y_test))
