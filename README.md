# 🧠 Total Perspective Vortex

> *"If life is going to exist in a Universe of this size, then the one thing it cannot afford to have is a sense of proportion."*
> — Douglas Adams

A **Brain-Computer Interface (BCI)** built on electroencephalographic (EEG) data and machine learning. Given a subject's EEG recording, the system infers which motor imagery task (left hand, right hand, both hands, or feet) the person was performing — in near real-time.

---

## 📋 Project Overview

| Property | Value |
|---|---|
| Language | Python 3 |
| EEG Library | [MNE-Python](https://mne.tools/) |
| ML Library | [scikit-learn](https://scikit-learn.org/) |
| Dataset | [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) |
| Target accuracy | ≥ 60% mean across 6 experiment types, 109 subjects |

---

## 🗂️ Repository Structure

```
total_perspective_vortex/
│
├── one_sample/            # Sample EDF files for subject S001 (runs R01–R14)
├── processed_data/        # Saved .fif epoch files (auto-generated, gitignored)
├── subject/               # Subject PDF documentation
│
├── preprocess.py          # Phase 1 – EEG loading, filtering, epoching, feature prep
├── plot.py                # Visualization helpers (raw vs filtered, ERD/ERS, features)
├── visualization.py       # Additional visualization utilities
│
├── csp.py                 # [TODO] Custom CSP transformer (sklearn-compatible)
├── pipeline.py            # [TODO] Full sklearn Pipeline (CSP → Classifier)
├── train.py               # [TODO] Training script: mybci.py <subject> <run> train
├── predict.py             # [TODO] Prediction/streaming script: mybci.py <subject> <run> predict
├── mybci.py               # [TODO] Main entry point (train | predict | full eval)
│
├── requirements.txt       # Python dependencies
├── init_env.sh            # Environment setup script
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/bbritva/total_perspective_vortex.git
cd total_perspective_vortex

# Create and activate virtual environment
bash init_env.sh
source venv/bin/activate   # or: . venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
```
mne
scikit-learn
numpy
matplotlib
```

---

## 📊 Dataset

The project uses the **PhysioNet EEG Motor Movement/Imagery Dataset** (BCI2000), recorded at 160 Hz with 64 EEG channels from 109 subjects.

Run types relevant to this project:

| Run numbers | Task |
|---|---|
| 3, 4, 7, 8, 11, 12 | Left/right hand (motor execution + imagery) |
| 5, 6, 9, 10, 13, 14 | Both hands / feet |
| 1, 2 | Baselines (not used) |

Annotations:
- `T1` → label `1` (e.g. left hand)
- `T2` → label `2` (e.g. right hand)

Download data via MNE:
```python
from mne.datasets import eegbci
eegbci.load_data(subject=1, runs=[3, 4, 7, 8, 11, 12])
```

---

## 🔬 Phase 1 — Preprocessing (`preprocess.py`)

**What is done:**
- Loads `.edf` files, standardizes channel names to 10–20 system
- Picks EEG channels only, sets average reference
- Band-pass filter: **8–30 Hz** (mu + beta rhythms)
- Channels of interest: **C3, C4** (primary motor cortex)
- Epochs time-locked to events: `tmin=0s`, `tmax=2s`
- Saves epochs as `.fif` files to `processed_data/`
- Returns `X` (n_epochs, n_channels, n_times) and `y` (labels)

**Visualization (`plot.py`):**
- Raw vs filtered signal side-by-side
- ERD/ERS contrast plot (Left − Right)
- Feature boxplots (mean, variance, PTP, mu power, beta power)

---

## 🧮 Phase 2 — Dimensionality Reduction (CSP)

**Goal:** Implement **Common Spatial Patterns (CSP)** as a custom sklearn-compatible transformer.

The CSP finds a projection matrix \( W \) such that:
$$W^T X = X_{\text{CSP}}$$

where the transformed data maximizes variance separation between the two motor imagery classes.

**Implementation** (`csp.py`) must:
- Inherit from `BaseEstimator` and `TransformerMixin`
- Implement `.fit(X, y)` — computes covariance matrices per class and solves the generalized eigenvalue problem
- Implement `.transform(X)` — projects data onto top/bottom `n_components` filters
- Use `numpy` / `scipy` for eigenvalue decomposition

---

## 🤖 Phase 3 — Pipeline & Classification

The full sklearn `Pipeline` chains:
```
RawData → [CSP] → [StandardScaler] → [Classifier] → Label
```

Suggested classifiers: `LDA`, `SVC`, `RandomForestClassifier`.

Evaluation uses `cross_val_score` with `StratifiedKFold`.

---

## 🖥️ Usage

### Train on a specific subject and run:
```bash
python mybci.py <subject_id> <run_id> train
# Example:
python mybci.py 4 14 train
```

### Predict (simulated real-time stream, <2s delay per epoch):
```bash
python mybci.py <subject_id> <run_id> predict
```

### Full evaluation across all 109 subjects and 6 experiments:
```bash
python mybci.py
# Expected output:
# experiment 0: accuracy = 0.60+
# ...
# Mean accuracy of 6 experiments: 0.62+
```

---

## 🎯 Evaluation Criteria

- **≥ 60% mean accuracy** across all 6 experiment types on test data (never-seen subjects/runs)
- `cross_val_score` used on the whole pipeline
- Train/Validation/Test splits must differ each run (no overfitting)
- Prediction must happen within **2 seconds** per epoch in stream mode

---

## 🌟 Bonus Ideas

- Wavelet transform for preprocessing (instead of simple band-pass)
- Implement your own classifier
- Custom eigenvalue / SVD decomposition without scipy shortcuts
- Extend to other PhysioNet datasets

---

## 📚 References

- [MNE EEG Motor Imagery tutorial](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [PhysioNet EEG Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [CSP Algorithm explained](https://en.wikipedia.org/wiki/Common_spatial_pattern)
- [sklearn Pipeline docs](https://scikit-learn.org/stable/modules/pipeline.html)
