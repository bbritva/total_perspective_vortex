#!/usr/bin/env python3
"""
Common Spatial Patterns (CSP) — custom sklearn-compatible transformer.

Design goals:
  - Works with ANY number of channels (auto-detected from X.shape)
  - n_components defaults to min(2*2, n_channels) so it's always safe
  - Inherits BaseEstimator + TransformerMixin → usable in sklearn Pipeline,
    cross_val_score, GridSearchCV out of the box
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns transformer for binary EEG classification.

    Parameters
    ----------
    n_components : int or None
        Number of CSP filters to use (must be even: k top + k bottom).
        If None, defaults to min(4, n_channels) at fit time — always safe.
    reg : float
        Regularization added to the composite covariance matrix diagonal.
        Prevents numerical issues when channels are nearly collinear.
        Small value like 1e-6 is fine; increase if you see singular matrix errors.

    Attributes (set after .fit())
    --------------------------------
    filters_ : ndarray, shape (n_components, n_channels)
        The selected spatial filters (rows = filters, columns = channels).
    eigenvalues_ : ndarray, shape (n_components,)
        Eigenvalues of the selected filters (useful for inspection/plotting).
    n_channels_ : int
        Number of channels detected from training data.
    classes_ : ndarray
        The two unique class labels found in y.
    """

    def __init__(self, n_components=None, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _covariance(self, epochs):
        """
        Compute the average normalized covariance matrix across epochs.

        For each epoch (n_channels, n_times):
          1. Compute covariance matrix  C = X @ X.T  (shape: n_ch x n_ch)
          2. Normalize by trace so amplitude differences don't dominate
          3. Average across all epochs of this class

        Parameters
        ----------
        epochs : ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        cov : ndarray, shape (n_channels, n_channels)
        """
        covs = []
        for epoch in epochs:
            # X @ X.T is faster and numerically equivalent to np.cov
            # when the signal is zero-mean (which it is after band-pass filtering)
            C = epoch @ epoch.T
            covs.append(C / np.trace(C))
        return np.mean(covs, axis=0)

    def _resolve_n_components(self, n_channels):
        """
        Determine the actual number of components to use.

        Rules:
          - Must be even (we take k from the top AND k from the bottom)
          - Cannot exceed n_channels (hard limit from linear algebra)
          - Default: min(4, n_channels) rounded down to nearest even number
        """
        if self.n_components is None:
            n = min(4, n_channels)
        else:
            n = self.n_components

        # Clamp to channel count
        n = min(n, n_channels)

        # Force even number (top-k + bottom-k symmetry)
        if n % 2 != 0:
            n -= 1

        if n < 2:
            raise ValueError(
                f"Need at least 2 components (1 per class), got {n}. "
                f"Check n_components and n_channels ({n_channels})."
            )
        return n

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Learn the CSP spatial filters from training data.

        Steps:
          1. Separate epochs by class
          2. Compute per-class normalized covariance matrices (Σ1, Σ2)
          3. Solve the generalized eigenvalue problem:
               Σ1 @ w = λ * (Σ1 + Σ2) @ w
             → eigenvalues λ ∈ [0, 1]:
               λ ≈ 1  means this filter captures mostly class-1 variance
               λ ≈ 0  means this filter captures mostly class-2 variance
          4. Select top-k (λ largest) and bottom-k (λ smallest) filters
             → these maximally separate the two classes

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        y : ndarray, shape (n_epochs,)  — class labels (any two distinct values)

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(self.classes_)}: {self.classes_}"
            )

        _, self.n_channels_, _ = X.shape
        n_components = self._resolve_n_components(self.n_channels_)

        # Separate epochs by class
        X1 = X[y == self.classes_[0]]  # class 0 (e.g. label=1, left hand)
        X2 = X[y == self.classes_[1]]  # class 1 (e.g. label=2, right hand)

        # Per-class covariance matrices
        cov1 = self._covariance(X1)
        cov2 = self._covariance(X2)

        # Composite covariance (total spread = sum of both classes)
        cov_total = cov1 + cov2

        # Regularization: nudge diagonal to avoid singular matrix
        # (especially important with only 2 channels or short recordings)
        cov_total += self.reg * np.eye(self.n_channels_)

        # Generalized eigenvalue problem: cov1 @ W = λ * cov_total @ W
        # eigh() returns eigenvalues sorted ASCENDING and guarantees real output
        # (valid because covariance matrices are symmetric positive semi-definite)
        eigenvalues, eigenvectors = eigh(cov1, cov_total)

        # eigenvalues shape: (n_channels,)   — sorted low → high
        # eigenvectors shape: (n_channels, n_channels) — columns are eigenvectors

        # Select: last k (highest λ → best for class 1)
        #         first k (lowest λ → best for class 2)
        k = n_components // 2
        top_idx    = np.arange(self.n_channels_ - k, self.n_channels_)  # last k
        bottom_idx = np.arange(k)                                        # first k
        selected   = np.concatenate([bottom_idx, top_idx])               # 2k total

        # Store as row filters: shape (n_components, n_channels)
        self.filters_     = eigenvectors[:, selected].T
        self.eigenvalues_ = eigenvalues[selected]

        return self

    def transform(self, X):
        """
        Project epochs onto CSP filters and return log-variance features.

        For each epoch:
          1. Project:  Z = filters_ @ epoch   → shape (n_components, n_times)
          2. Variance: var per component       → shape (n_components,)
          3. Log:      log(var) to normalize scale and approximate Gaussianity

        The log-variance is the standard CSP feature for EEG.
        It compresses large variance ranges and makes distributions more
        Gaussian, which benefits LDA and other linear classifiers.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : ndarray, shape (n_epochs, n_components)
        """
        # Project all epochs at once: (n_epochs, n_components, n_times)
        projected = np.tensordot(self.filters_, X, axes=[[1], [1]])
        # tensordot result shape: (n_components, n_epochs, n_times)
        # → transpose to (n_epochs, n_components, n_times)
        projected = projected.transpose(1, 0, 2)

        # Variance over time axis, then log
        # Clip at a tiny positive value to avoid log(0) on flat signals
        var = np.var(projected, axis=2)
        log_var = np.log(np.clip(var, 1e-10, None))

        return log_var  # shape: (n_epochs, n_components)

    def fit_transform(self, X, y=None, **fit_params):
        """Convenience: fit then transform in one call."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Inspection helpers (not required by sklearn, useful for debugging)
    # ------------------------------------------------------------------

    def get_filter_info(self):
        """
        Print a summary of learned filters — useful during development.

        Example output with 2 channels:
          Filter 0: eigenvalue=0.0821  weights=[C3: -0.72, C4:  0.91]
          Filter 1: eigenvalue=0.9179  weights=[C3:  0.85, C4:  0.63]
        """
        if not hasattr(self, 'filters_'):
            print("CSP not fitted yet. Call .fit(X, y) first.")
            return
        print(f"CSP: {len(self.filters_)} filters, {self.n_channels_} channels")
        print(f"Classes: {self.classes_[0]} vs {self.classes_[1]}")
        for i, (filt, ev) in enumerate(zip(self.filters_, self.eigenvalues_)):
            weights = "  ".join(f"{w:+.4f}" for w in filt)
            print(f"  Filter {i}: λ={ev:.4f}  weights=[{weights}]")
