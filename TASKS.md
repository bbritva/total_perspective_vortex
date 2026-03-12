# 📌 Total Perspective Vortex — Task Tracker

Status legend: ✅ Done · 🔧 In Progress · ⬜ To Do · 🌟 Bonus

---

## Phase 1 — Preprocessing & Visualization

| # | Task | File | Status |
|---|---|---|---|
| 1.1 | Load `.edf` files with MNE, standardize 10-20 channel names | `preprocess.py` | ✅ |
| 1.2 | Set average reference, pick EEG channels only | `preprocess.py` | ✅ |
| 1.3 | Band-pass filter 8–30 Hz (mu + beta) | `preprocess.py` | ✅ |
| 1.4 | Epoch extraction around T1/T2 events (0–2s) | `preprocess.py` | ✅ |
| 1.5 | Filter by relevant run types (LRW: R03,04,07,08,11,12) | `preprocess.py` | ✅ |
| 1.6 | Save processed epochs to `.fif` files | `preprocess.py` | ✅ |
| 1.7 | Visualize raw vs filtered signal (C3, C4) | `plot.py` | ✅ |
| 1.8 | ERD/ERS contrast plot (Left − Right hand) | `plot.py` | ✅ |
| 1.9 | Feature boxplots (mean, var, PTP, mu/beta power) | `plot.py` | ✅ |
| 1.10 | **Expand to WF runs** (both hands/feet, R05,06,09,10,13,14) | `preprocess.py` | ⬜ |
| 1.11 | Support downloading data for any subject via `eegbci.load_data()` | `preprocess.py` | ⬜ |

---

## Phase 2 — Custom CSP Transformer

| # | Task | File | Status |
|---|---|---|---|
| 2.1 | Create `CSP` class inheriting `BaseEstimator, TransformerMixin` | `csp.py` | ⬜ |
| 2.2 | `.fit(X, y)`: compute per-class covariance matrices | `csp.py` | ⬜ |
| 2.3 | Solve generalized eigenvalue problem: `C1 w = λ C2 w` | `csp.py` | ⬜ |
| 2.4 | Sort eigenvectors, keep top-k and bottom-k filters | `csp.py` | ⬜ |
| 2.5 | `.transform(X)`: project epochs onto CSP filters, return log-variance | `csp.py` | ⬜ |
| 2.6 | Validate CSP transformer in isolation on one subject | `csp.py` | ⬜ |
| 2.7 | 🌟 Custom eigenvalue decomposition (no `scipy.linalg.eig`) | `csp.py` | 🌟 |

---

## Phase 3 — Pipeline & Classification

| # | Task | File | Status |
|---|---|---|---|
| 3.1 | Build sklearn `Pipeline([CSP, StandardScaler, Classifier])` | `pipeline.py` | ⬜ |
| 3.2 | Evaluate with `cross_val_score` + `StratifiedKFold` on one subject | `pipeline.py` | ⬜ |
| 3.3 | Test with `LDA`, `SVC`, `RandomForest` — compare scores | `pipeline.py` | ⬜ |
| 3.4 | Tune hyperparameters (n_components for CSP, C for SVC…) | `pipeline.py` | ⬜ |
| 3.5 | 🌟 Try PCA or ICA as alternative to CSP | `pipeline.py` | 🌟 |

---

## Phase 4 — Train Script

| # | Task | File | Status |
|---|---|---|---|
| 4.1 | Accept CLI args: `mybci.py <subject> <run> train` | `mybci.py` | ⬜ |
| 4.2 | Load & preprocess data for given subject+run | `train.py` | ⬜ |
| 4.3 | Split into Train / Validation / Test (stratified, no overlap) | `train.py` | ⬜ |
| 4.4 | Fit pipeline, print `cross_val_score` array + mean | `train.py` | ⬜ |
| 4.5 | Serialize trained pipeline to disk (`joblib.dump`) | `train.py` | ⬜ |

---

## Phase 5 — Predict / Streaming Script

| # | Task | File | Status |
|---|---|---|---|
| 5.1 | Accept CLI args: `mybci.py <subject> <run> predict` | `mybci.py` | ⬜ |
| 5.2 | Load serialized pipeline from disk | `predict.py` | ⬜ |
| 5.3 | Simulate data stream: iterate over epochs with ≤2s delay per epoch | `predict.py` | ⬜ |
| 5.4 | Print per-epoch: `epoch nb: [prediction] [truth] equal?` | `predict.py` | ⬜ |
| 5.5 | Print final accuracy | `predict.py` | ⬜ |
| 5.6 | **Do NOT use `mne-realtime`** (use manual iteration) | `predict.py` | ⬜ |

---

## Phase 6 — Full Evaluation (All Subjects)

| # | Task | File | Status |
|---|---|---|---|
| 6.1 | `mybci.py` with no args: iterate over all 109 subjects | `mybci.py` | ⬜ |
| 6.2 | Test all 6 experiment types per subject | `mybci.py` | ⬜ |
| 6.3 | Report accuracy per experiment and mean across all | `mybci.py` | ⬜ |
| 6.4 | Ensure ≥60% mean accuracy on never-seen test data | `mybci.py` | ⬜ |

---

## Bonus

| # | Task | Status |
|---|---|---|
| B.1 | 🌟 Wavelet transform preprocessing (replace band-pass) | 🌟 |
| B.2 | 🌟 Custom classifier implementation | 🌟 |
| B.3 | 🌟 Work on other EEG datasets | 🌟 |
| B.4 | 🌟 Custom eigenvalue/SVD decomposition | 🌟 |

---

## 🔜 Suggested Next Steps (priority order)

1. **Create `csp.py`** — implement the CSP transformer (Phase 2)
2. **Create `pipeline.py`** — wire CSP into a full sklearn pipeline and test with `cross_val_score`
3. **Expand `preprocess.py`** — support all 6 experiment types (WF runs)
4. **Create `train.py` + `predict.py`** — CLI scripts as required by the subject
5. **Create `mybci.py`** — main entry point routing to train/predict/full eval
6. Reach ≥60% accuracy across all subjects
