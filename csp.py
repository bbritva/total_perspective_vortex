#!/usr/bin/env python3
"""
Common Spatial Patterns (CSP) — custom sklearn-compatible transformer.

Design goals:
  - Works with ANY number of channels (auto-detected from X.shape)
  - n_components defaults to min(4, n_channels) so it's always safe
  - Inherits BaseEstimator + TransformerMixin → usable in sklearn Pipeline,
    cross_val_score, GridSearchCV out of the box
"""

import numpy as np
from numpy import ndarray
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

    def __init__(self, n_components: int | None = None, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _covariance(self, epochs: ndarray) -> ndarray:
        """
        Compute the average normalized covariance matrix across epochs.

        For each epoch (n_channels, n_times):
          1. C = X @ X.T  (shape: n_ch x n_ch)
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
            # when the signal is zero-mean (after band-pass filtering)
            C: ndarray = epoch @ epoch.T
            covs.append(C / np.trace(C))
        return np.mean(covs, axis=0)

    def _resolve_n_components(self, n_channels: int) -> int:
        """
        Determine the actual number of components to use.

        Rules:
          - Must be even (we take k from the top AND k from the bottom)
          - Cannot exceed n_channels (hard limit from linear algebra)
          - Default: min(4, n_channels) rounded down to nearest even number
        """
        n: int = min(4, n_channels) if self.n_components is None else self.n_components
        n = min(n, n_channels)   # clamp to channel count
        if n % 2 != 0:           # force even
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

    def fit(self, X: ndarray, y: ndarray) -> "CSP":
        """
        Learn the CSP spatial filters from training data.

        Steps:
          1. Separate epochs by class
          2. Compute per-class normalized covariance matrices (Σ1, Σ2)
          3. Solve generalized eigenvalue problem: Σ1 @ w = λ * (Σ1 + Σ2) @ w
             → eigenvalues λ ∈ [0, 1]:
               λ ≈ 1  means this filter captures mostly class-1 variance
               λ ≈ 0  means this filter captures mostly class-2 variance
          4. Select top-k (λ largest) and bottom-k (λ smallest) filters

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        y : ndarray, shape (n_epochs,) — class labels (any two distinct values)

        Returns
        -------
        self
        """
        self.classes_: ndarray = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(self.classes_)}: {self.classes_}"
            )

        _, self.n_channels_, _ = X.shape
        n_components = self._resolve_n_components(self.n_channels_)

        X1: ndarray = X[y == self.classes_[0]]  # e.g. label=1, left hand
        X2: ndarray = X[y == self.classes_[1]]  # e.g. label=2, right hand

        cov1: ndarray = self._covariance(X1)
        cov2: ndarray = self._covariance(X2)

        cov_total: ndarray = cov1 + cov2
        # Regularization: nudge diagonal to avoid singular matrix
        # (especially important with only 2 channels or short recordings)
        cov_total += self.reg * np.eye(self.n_channels_)

        # eigh() returns eigenvalues sorted ASCENDING, guarantees real output
        # (valid because covariance matrices are symmetric positive semi-definite)
        eigenvalues: ndarray
        eigenvectors: ndarray
        eigenvalues, eigenvectors = eigh(cov1, cov_total)

        # eigenvalues shape:  (n_channels,)            — sorted low → high
        # eigenvectors shape: (n_channels, n_channels) — columns are eigenvectors

        k: int = n_components // 2
        top_idx: ndarray    = np.arange(self.n_channels_ - k, self.n_channels_)  # highest λ → class 1
        bottom_idx: ndarray = np.arange(k)                                        # lowest  λ → class 2
        selected: ndarray   = np.concatenate([bottom_idx, top_idx])

        # Store as row vectors: shape (n_components, n_channels)
        self.filters_: ndarray     = eigenvectors[:, selected].T
        self.eigenvalues_: ndarray = eigenvalues[selected]

        return self

    def transform(self, X: ndarray) -> ndarray:
        """
        Project epochs onto CSP filters and return log-variance features.

        einsum notation 'fc,ect->eft':
          f = n_components  (filters)
          e = n_epochs
          c = n_channels    (contracted / summed over)
          t = n_times

        Produces (n_epochs, n_components, n_times), then var+log over time.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : ndarray, shape (n_epochs, n_components)
        """
        # filters_: (n_components, n_channels) — 'fc'
        # X:        (n_epochs, n_channels, n_times) — 'ect'
        # result:   (n_epochs, n_components, n_times) — 'eft'
        projected: ndarray = np.einsum("fc,ect->eft", self.filters_, X)

        # Variance over time axis (2), then log
        # Clip to avoid log(0) on pathological flat signals
        var: ndarray     = np.var(projected, axis=2)
        log_var: ndarray = np.log(np.clip(var, 1e-10, None))

        return log_var  # shape: (n_epochs, n_components)

    def fit_transform(self, X: ndarray, y: ndarray, **fit_params) -> ndarray:  # type: ignore[override]
        """Convenience: fit then transform in one call."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Inspection helpers (not required by sklearn, useful for debugging)
    # ------------------------------------------------------------------

    def get_filter_info(self) -> None:
        """
        Print a summary of learned filters — useful during development.

        Example output with 2 channels (C3, C4):
          CSP: 2 filters, 2 channels
          Classes: 1 vs 2
          Filter 0: λ=0.0821  weights=[-0.7201  +0.9134]
          Filter 1: λ=0.9179  weights=[+0.8502  +0.6311]
        """
        if not hasattr(self, "filters_"):
            print("CSP not fitted yet. Call .fit(X, y) first.")
            return
        print(f"CSP: {len(self.filters_)} filters, {self.n_channels_} channels")
        print(f"Classes: {self.classes_[0]} vs {self.classes_[1]}")
        for i, (filt, ev) in enumerate(zip(self.filters_, self.eigenvalues_)):
            weights = "  ".join(f"{w:+.4f}" for w in filt)
            print(f"  Filter {i}: λ={ev:.4f}  weights=[{weights}]")
