from preprocess import load_subject
from csp import CSP
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA = Path("data/physionet.org/files/eegmmidb/1.0.0")

pipeline = Pipeline([
    ('csp',    CSP(n_components=4)),
    ('scaler', StandardScaler()),
    ('lda',    LinearDiscriminantAnalysis())
])


X, y = load_subject(DATA / "S004")
print(np.unique(y, return_counts=True))
# If output is (array([1, 2]), array([90, 45])) → class imbalance

