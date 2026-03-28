# Total Perspective Vortex

A Brain-Computer Interface (BCI) pipeline that classifies motor imagery EEG signals using a custom implementation of Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA).

> **Result:** 65.59% mean accuracy across 109 subjects × 6 experiment types on the PhysioNet EEG Motor Movement/Imagery dataset — above the required 60% threshold.

---

## Table of Contents

- [The Problem](#the-problem)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
  - [Step 1 — Preprocessing](#step-1--preprocessing)
  - [Step 2 — Common Spatial Patterns (CSP)](#step-2--common-spatial-patterns-csp)
  - [Step 3 — Classification with LDA](#step-3--classification-with-lda)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Results](#results)

---

## The Problem

When a person imagines moving their left hand vs. their right hand, different regions of the motor cortex become active. This manifests as changes in EEG signal power in the **mu (8–12 Hz)** and **beta (13–30 Hz)** frequency bands — a phenomenon called **Event-Related Desynchronization (ERD)**.

The challenge: EEG signals are extremely noisy, non-stationary, and highly subject-specific. A model trained on one person's brain signals does not generalize to another person. This pipeline trains one model per subject, per experiment type.

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Database** (109 subjects)

Each subject performed 14 experimental runs:
- **Rest** (R01, R02)
- **Real movement** — left/right hand, both hands/feet (R03–R06)
- **Imagined movement** — same tasks repeated twice (R07–R14)

This project uses 6 experiment types, each pairing 2 runs per subject:

| Experiment | Runs | Task |
|---|---|---|
| 0 | R03, R04 | Real left/right hand |
| 1 | R07, R08 | Imagined left/right hand (1st) |
| 2 | R11, R12 | Imagined left/right hand (2nd) |
| 3 | R05, R06 | Real both hands / feet |
| 4 | R09, R10 | Imagined both hands / feet (1st) |
| 5 | R13, R14 | Imagined both hands / feet (2nd) |

Each run contains ~45 epochs of 2-second trials annotated as **T1** (class 1) or **T2** (class 2).

---

## How It Works

### Step 1 — Preprocessing

**File:** `preprocess.py`

Raw EEG data goes through the following pipeline for each `.edf` file:

```
Raw EDF → Standardize channel names → Set montage (standard_1020)
       → Pick EEG channels only
       → Average reference (apply_proj)
       → Band-pass filter 8–30 Hz
       → Epoch around T1/T2 events [0s, 2s]
       → Select 7 motor cortex channels: FC3, FC4, C3, Cz, C4, CP3, CP4
```

**Why these steps:**
- **Average reference** removes the common-mode noise shared by all electrodes, making spatial differences between channels more meaningful.
- **Band-pass 8–30 Hz** isolates the mu and beta rhythms where motor imagery signal lives. Everything below (eye blinks, slow drifts) and above (muscle artifacts, high-frequency noise) is discarded.
- **7 motor cortex channels** — the full 64-channel cap contains many channels irrelevant to hand/foot motor imagery. Selecting FC3/FC4/C3/Cz/C4/CP3/CP4 reduces dimensionality and focuses the spatial filter on the relevant cortex region.
- **Epochs [0, 2s]** — the motor imagery signal peaks in the first 2 seconds after cue onset.

---

### Step 2 — Common Spatial Patterns (CSP)

**File:** `csp.py`

CSP is a spatial filtering algorithm that finds linear combinations of EEG channels that **maximize variance for one class while minimizing it for the other**. It is the core of this project and was implemented from scratch as a scikit-learn compatible transformer.

#### The Math

Given two classes of epochs (e.g. left hand vs. right hand), CSP computes:

1. **Per-class normalized covariance matrix:**

   For each epoch `X` of shape `(n_channels, n_times)`:
   ```
   C = (X @ X.T) / trace(X @ X.T)
   ```
   Then average across all epochs of that class → `Σ1`, `Σ2`.

   Normalizing by the trace removes amplitude differences between subjects and sessions, keeping only the spatial covariance structure.

2. **Generalized eigenvalue problem:**

   ```
   Σ1 @ w = λ * (Σ1 + Σ2) @ w
   ```

   Solved with `scipy.linalg.eigh` (symmetric positive semi-definite version — faster and numerically stable).

   The eigenvalues `λ ∈ [0, 1]` have a direct interpretation:
   - `λ ≈ 1` → this spatial filter captures mostly **class 1** variance
   - `λ ≈ 0` → this spatial filter captures mostly **class 2** variance
   - `λ ≈ 0.5` → this filter is uninformative (equal variance in both classes)

3. **Filter selection:**

   From all `n_channels` eigenvectors, we keep:
   - The `k` filters with **highest** λ → class 1 detectors
   - The `k` filters with **lowest** λ → class 2 detectors

   Default: `n_components = 4` → k=2 from each end.

4. **Feature extraction:**

   Each epoch is projected through the selected filters:
   ```
   projected = filters @ epoch          # (n_components, n_times)
   features  = log(var(projected))      # (n_components,)
   ```

   Log-variance compresses the dynamic range and makes features approximately Gaussian — ideal for LDA.

#### Why CSP Works for Motor Imagery

During left-hand imagery, the left motor cortex (C3 area) desynchronizes — its power in 8–30 Hz drops. The right motor cortex (C4 area) may synchronize. CSP automatically discovers this spatial asymmetry by finding filters that emphasize C3 for one class and C4 for the other, even when the signal is mixed across many electrodes.

#### Implementation Details

```python
class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, reg=1e-4): ...
    def fit(self, X, y): ...       # learns filters from training data
    def transform(self, X): ...    # returns log-variance features
    def fit_transform(self, X, y): ...
```

- Inherits `BaseEstimator` + `TransformerMixin` → works inside sklearn `Pipeline`, `cross_val_score`, `GridSearchCV` with zero extra code.
- `reg=1e-4` adds a small diagonal term to `Σ1 + Σ2` to prevent singular matrix errors when channels are nearly collinear.
- `n_components=None` auto-selects `min(4, n_channels)` at fit time, always rounded down to an even number.

---

### Step 3 — Classification with LDA

**File:** `pipeline.py`

The full sklearn pipeline:

```
CSP → StandardScaler → LinearDiscriminantAnalysis
```

- **CSP** extracts `n_components` log-variance features per epoch.
- **StandardScaler** normalizes features to zero mean and unit variance. This prevents LDA from being dominated by features with larger absolute values.
- **LDA** finds the linear decision boundary that maximizes the ratio of between-class variance to within-class variance. With CSP features, the two classes are already well-separated along the feature axes, so LDA converges quickly even with small training sets.

**Why LDA over SVM or Random Forest?**

LDA is optimal when features are normally distributed — which log-variance CSP features approximately are. It is also very fast and robust with small sample sizes (30–100 epochs per subject), where SVM and RF tend to overfit.

---

## Project Structure

```
total_perspective_vortex/
│
├── mybci.py          # Main CLI — train, predict, full evaluation
├── pipeline.py       # Pipeline: build, train, evaluate, predict_stream
├── csp.py            # Custom CSP transformer (sklearn-compatible)
├── preprocess.py     # MNE-based EEG loading and preprocessing
├── plot.py           # Visualization: raw signal, ERD/ERS, feature plots
├── visualization.py  # Helper visualizations
├── test.py           # Quick sanity checks
├── requirements.txt  # Python dependencies
├── init_env.sh       # Virtualenv setup script
│
├── data/             # PhysioNet dataset (not committed)
├── models/           # Saved .pkl pipelines (not committed)
└── processed_data/   # Cached .fif epoch files (not committed)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bbritva/total_perspective_vortex.git
cd total_perspective_vortex

# Create virtualenv and install dependencies
bash init_env.sh
source venv/bin/activate

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:**
```
mne
numpy
scipy
scikit-learn
joblib
matplotlib
```

**Dataset:** Download from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) and place under:
```
data/physionet.org/files/eegmmidb/1.0.0/S001/
data/physionet.org/files/eegmmidb/1.0.0/S002/
...
```

---

## Usage

### Train a model on a single subject/run

```bash
python mybci.py <subject> <run> train
```

Example:
```bash
python mybci.py 1 3 train
```

Output:
```
Loading S001 run 03...
Epochs: 23, shape: (23, 7, 321)
[0.25, 0.75, 0.75, 0.3333, 0.6667]
cross_val_score: 0.5500
Test score (held-out 20%): 0.6000
Model saved to: models/S001_R03.pkl
```

### Predict on held-out test set (streaming simulation)

```bash
python mybci.py <subject> <run> predict
```

Example:
```bash
python mybci.py 1 3 predict
```

Output:
```
epoch nb: [prediction] [truth] equal?
epoch 00: [1] [2] False
epoch 01: [2] [1] False
epoch 02: [1] [1] True
epoch 03: [1] [1] True
epoch 04: [2] [2] True

Accuracy: 0.6000
```

Each epoch is classified with a 0.25s delay to simulate real-time streaming (max allowed: 2s).

### Full evaluation — all subjects × all experiments

```bash
python mybci.py
```

Output (excerpt):
```
experiment 0: subject S001: accuracy = 0.7000
experiment 1: subject S001: accuracy = 0.6000
...
experiment 5: subject S109: accuracy = 0.6667

Mean accuracy of the six different experiments for all subjects:
experiment 0: accuracy = 0.6689
experiment 1: accuracy = 0.6202
experiment 2: accuracy = 0.6576
experiment 3: accuracy = 0.6596
experiment 4: accuracy = 0.6778
experiment 5: accuracy = 0.6515

Mean accuracy of 6 experiments: 0.6559
```

### Visualize EEG data

```bash
python plot.py
```

Shows: raw vs. filtered signal, ERD/ERS contrast (C3 vs C4), CSP feature boxplots.

### CSP quality diagnostic

```bash
python preprocess.py data/.../S001/ --check
```

Output:
```
CSP eigenvalue spread: 0.6234  (want > 0.3)
Eigenvalues: [0.1883 0.2341 0.7659 0.8117]
```

A spread > 0.3 indicates the subject has decodable spatial patterns. Values near 0.5 throughout mean the subject's motor imagery is not reliably reflected in the EEG.

---

## Architecture & Design Decisions

### Per-subject models

Each model is trained exclusively on data from the subject it will predict. Brain signals are highly individual — scalp geometry, skull thickness, and cortical organization differ between people. Cross-subject models consistently underperform per-subject models for motor imagery.

### 80/20 stratified split

The 80% train / 20% test split is stratified by class label, ensuring both classes are represented proportionally in each partition. The test set is saved alongside the model so `predict` mode always uses the exact same held-out epochs, guaranteeing reproducibility.

### Graceful degradation

Some subjects (~5%) have degenerate EEG for specific experiments — flat signals, zero-variance channels, or complete absence of motor imagery signal. The pipeline detects these cases and skips them rather than crashing:
- `_check_data()`: validates epoch count and class balance before fitting
- `try/except` in `evaluate_subject()`: catches numerical failures from LDA (e.g. zero singular values after degenerate CSP projection)

### Regularization

`reg=1e-4` Tikhonov regularization is added to the composite covariance matrix `Σ1 + Σ2` before solving the eigenvalue problem. This prevents singular matrix errors when channels are nearly collinear — common with 7-channel recordings and small epoch counts.

---

## Results

| Experiment | Task | Mean Accuracy |
|---|---|---|
| 0 | Real left/right hand | 66.89% |
| 1 | Imagined left/right hand (1st) | 62.02% |
| 2 | Imagined left/right hand (2nd) | 65.76% |
| 3 | Real both hands / feet | 65.96% |
| 4 | Imagined both hands / feet (1st) | 67.78% |
| 5 | Imagined both hands / feet (2nd) | 65.15% |
| **Overall** | | **65.59%** |

Chance level for binary classification is 50%. The pipeline achieves ~15 percentage points above chance consistently across all task types, including purely imagined movements.
