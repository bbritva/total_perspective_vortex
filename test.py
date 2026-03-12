from preprocess import main
from csp import CSP
from pathlib import Path
import numpy as np
from scipy.linalg import eigh


X, y = main(Path("one_sample/"))   # uses your existing sample data

X1 = X[y == 1]
X2 = X[y == 2]
cov1 = np.mean([e @ e.T / np.trace(e @ e.T) for e in X1], axis=0)
cov2 = np.mean([e @ e.T / np.trace(e @ e.T) for e in X2], axis=0)

reg = 1e-4  # stronger than 1e-6 for rank-deficient case
cov_total = cov1 + cov2 + reg * np.eye(cov1.shape[0])
ev, _ = eigh(cov1, cov_total)
print("All eigenvalues:", ev)


csp = CSP()          # n_components=None → auto (2 with C3/C4)
csp.fit(X, y)
csp.get_filter_info()  # prints eigenvalues + weights per filter

features = csp.transform(X)
print(features.shape)  # → (n_epochs, 2)
