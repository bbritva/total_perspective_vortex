# 🛡️ Total Perspective Vortex — Defense / Evaluation Guide

This document maps **every checkbox and slider** from the 42 Intra evaluation checklist
to the exact code, commands, and talking points you need during defense.

---

## 📋 Checklist Overview

| #   | Section                 | Item                  | Type     |
| --- | ----------------------- | --------------------- | -------- |
| 1   | Preprocessing           | Watch it for the plot | Yes / No |
| 2   | Preprocessing           | Feature extraction    | Yes / No |
| 3   | Classification Pipeline | Train                 | Yes / No |
| 4   | Classification Pipeline | Predict               | Yes / No |
| 5   | Classification Pipeline | Realtime              | Yes / No |
| 6   | Implementation          | Integration           | Yes / No |
| 7   | Implementation          | Implementation        | Yes / No |
| 8   | Implementation          | Score (≥60%)          | Yes / No |
| 9   | Implementation          | Score (slider 0-5)    | Slider   |

---

## 1. Preprocessing — "Watch it for the plot" ✅

> **Checklist says:** _Check if the data were parsed then visualized with a script, showing
> raw and filtered data. The plots should look like what is shown in the video, the filtered
> signal being "cleaner"._

### How to demonstrate

```bash
# Left/Right hand task (LRW runs: R03, R04, R07, R08, R11, R12):
python plot.py one_sample/S001R03.edf

# Both hands/Feet task (WF runs: R05, R06, R09, R10, R13, R14):
python plot.py one_sample/S001R05.edf
```

### What happens

- `plot.py → visualize_preprocessing()` loads the raw EDF file with MNE
- Plots **raw signal** (blue) vs **band-pass filtered signal 8–30 Hz** (red) for channels C3 and C4
- The filtered signal is visibly "cleaner" — high-frequency noise and slow drifts are removed
- Also calls `plot_erd_contrast()` showing ERD/ERS contrast between T1 and T2 conditions

**⚠️ Important — two different task types need different channel picks:**

| Task Type                 | Runs               | T1         | T2         | Best Channels for Contrast                       |
| ------------------------- | ------------------ | ---------- | ---------- | ------------------------------------------------ |
| **Left/Right hand (LRW)** | R03,04,07,08,11,12 | Left hand  | Right hand | **C3 vs C4** (contralateral motor cortex)        |
| **Both hands/Feet (WF)**  | R05,06,09,10,13,14 | Both hands | Both feet  | **C3/C4 vs Cz** (lateral vs vertex motor cortex) |

- For **LRW runs**: C3/C4 contrast is correct — left hand activates right motor cortex (C4),
  right hand activates left motor cortex (C3). The lateralization is what CSP exploits.
- For **WF runs**: C3 vs C4 is **NOT the right contrast** — both hands activate C3 and C4
  symmetrically, while feet activate the vertex (Cz). The meaningful contrast would be
  **Cz vs C3** (or C4), showing midline vs lateral motor cortex activation.
- The `visualize_preprocessing()` (raw vs filtered) is **task-agnostic** — it works on any EDF file.
- The `plot_erd_contrast()` **auto-detects the task type** from the filename:
    - LRW runs → picks **C3 vs C4**, title "Left vs Right Hand", labels "L-R"
    - WF runs → picks **Cz vs C3**, title "Hands vs Feet", labels "Hands-Feet"

**For defense:** Run `python plot.py` on both run types to show you handle both tasks:

```bash
python plot.py one_sample/S001R03.edf   # Left/Right hand → C3 vs C4
python plot.py one_sample/S001R05.edf   # Hands vs Feet   → Cz vs C3
```

### Key code references

| File            | Function                    | What it does                                                     |
| --------------- | --------------------------- | ---------------------------------------------------------------- |
| `plot.py`       | `visualize_preprocessing()` | Side-by-side raw vs filtered plot                                |
| `plot.py`       | `plot_erd_contrast()`       | ERD/ERS contrast (Left − Right hand)                             |
| `preprocess.py` | `preprocess_raw()`          | Band-pass filter 8–30 Hz, average reference, standardize montage |

### What the plot shows

- **Raw signal (blue):** Large amplitude swings (±100–150 µV), slow drifts visible especially in C3.
  These low-frequency components (< 8 Hz) are **noise/artifacts** — electrode drift, eye blinks,
  movement artifacts, DC offset changes. They are NOT useful for motor imagery classification.
- **Filtered signal (red, 8–30 Hz):** Much smaller amplitude (±10–20 µV), visibly "cleaner".
  This contains only the mu (8–12 Hz) and beta (13–30 Hz) oscillations — the frequency bands
  that show Event-Related Desynchronization (ERD) during motor imagery tasks.

### Talking points

- "We use MNE to load PhysioNet EDF files and standardize channel names to the 10-20 system"
- "Band-pass filtering at 8–30 Hz keeps the mu (8–12 Hz) and beta (13–30 Hz) rhythms, which are the frequency bands relevant to motor imagery"
- "The raw signal shows large slow drifts — these are low-frequency artifacts below 8 Hz (electrode drift, eye blinks, movement). Our filter removes all of that noise"
- "The filtered signal is cleaner and smaller in amplitude because we kept only the relevant frequency bands"
- "We use a FIR filter (`fir_design='firwin'`) which is the standard for EEG preprocessing"

### ERD/ERS Contrast Plot (second figure)

This plot is generated by `plot.py → plot_erd_contrast()` and shows the **Event-Related
Desynchronization/Synchronization (ERD/ERS) contrast** between T1 and T2 conditions.
The function auto-detects the task type from the filename and adjusts channels/labels accordingly.

**What the axes mean:**

- **X-axis (Time):** Time in seconds relative to event onset (t=0). The window is −0.2s to +0.5s.
  Negative times are the baseline period (before the cue appeared).
- **Y-axis (% change):** The percentage change in signal power for T1 minus T2.
  Positive values = more power during T1 trials; Negative values = more power during T2 trials.

**For Left/Right Hand runs (LRW):**

- **Purple (C3, L−R):** C3 is over the **right motor cortex** (contralateral to left hand).
  When the subject imagines left hand movement, C3 should show ERD (power decrease).
- **Green (C4, L−R):** C4 is over the **left motor cortex** (contralateral to right hand).
  When the subject imagines right hand movement, C4 should show ERD.
- The key insight: C3 and C4 show **different patterns** — this lateralization is what CSP exploits.

**For Hands/Feet runs (WF):**

- **Purple (Cz, Hands−Feet):** Cz is over the **vertex/midline motor cortex** — feet imagery
  produces strong ERD here (foot area is in the medial longitudinal fissure, represented at Cz).
- **Green (C3, Hands−Feet):** C3 is over the **lateral motor cortex** — hand imagery
  produces ERD here (hand area is on the lateral surface of the motor cortex).
- The key insight: Cz and C3 respond differently to hands vs feet — this is the spatial
  difference CSP learns to exploit.

**How to interpret the noisy pattern:**

- The oscillations between positive and negative values are expected — this is single-subject,
  single-run data with very few trials, so the contrast is noisy.
- The key insight is that the two channels show **different patterns** — they don't track each other
  perfectly. This spatial difference is exactly what CSP exploits for classification.
- With more subjects and more trials, these curves would smooth out and show clearer ERD patterns.

**Talking points for the evaluator:**

- "This plot shows the ERD/ERS contrast: T1 power minus T2 power, as a percentage change from baseline"
- "For left/right hand: C3 (purple) is over the right motor cortex, C4 (green) is over the left — we see lateralization"
- "For hands/feet: Cz (purple) is over the vertex where feet are represented, C3 (green) is lateral where hands are represented"
- "The plot auto-detects the task type from the filename and picks the appropriate channels"
- "The noise is expected with few trials from a single run — CSP handles this by learning optimal spatial filters across all channels"

---

## 2. Preprocessing — "Feature extraction" ✅

> **Checklist says:** _It's nice to filter a signal, but it needs to mean something in the context
> of your data. Check that the significative frequencies for a motor imagery task are kept (~8-40Hz).
> If the program learns to select the relevant frequencies for classification, it's better, cf bonus questions._

### How to demonstrate

```bash
# Show the band-pass filter range in the code:
grep -n "l_freq\|h_freq\|8\.\|30\." preprocess.py
```

### Key code references

| File            | Function/Line                                               | What it does                                                        |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `preprocess.py` | `preprocess_raw(l_freq=8., h_freq=30.)`                     | Band-pass filter keeping 8–30 Hz (mu + beta bands)                  |
| `preprocess.py` | `CHANNELS = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]` | Motor cortex channels selected                                      |
| `plot.py`       | `extract_features()`                                        | Extracts mu power (8–12 Hz) and beta power (13–30 Hz) via Welch PSD |
| `csp.py`        | `CSP.transform()`                                           | Log-variance features from CSP-projected signals                    |

### Talking points

- "We keep 8–30 Hz which covers both mu (8–12 Hz) and beta (13–30 Hz) rhythms — these are the frequency bands that show Event-Related Desynchronization (ERD) during motor imagery"
- "The checklist says ~8-40 Hz; our 8–30 Hz range is within that and is actually more focused on the relevant bands"
- "We select 7 channels over the motor cortex (C3, C4, Cz, FC3, FC4, CP3, CP4) — these are the electrodes that best capture hand/feet motor imagery signals"
- "The CSP algorithm then learns spatial filters that maximize the variance difference between the two classes — this is our main feature extraction step"
- "We also extract mu/beta band power features using Welch's method for visualization and analysis"

---

## 3. Classification Pipeline — "Train" ✅

> **Checklist says:** _The program has a train mode, sklearn score validation tools are used.
> The score for the training is displayed._

### How to demonstrate

```bash
python mybci.py 1 3 train
```

### Expected output

```
Loading S001 run 03...
Epochs: 21, shape: (21, 7, 321)
[0.6667 0.5000 0.7500 ...]
cross_val_score: 0.XXXX
Test score (held-out 20%): 0.XXXX
Model saved to: models/S001_R03.pkl
```

### Key code references

| File          | Function           | What it does                                                                               |
| ------------- | ------------------ | ------------------------------------------------------------------------------------------ |
| `mybci.py`    | `cmd_train()`      | CLI entry: loads data, calls `train()`, prints scores                                      |
| `pipeline.py` | `train()`          | Splits 80/20, runs `cross_val_score` with `StratifiedKFold(5)`, fits pipeline, saves model |
| `pipeline.py` | `build_pipeline()` | Creates `Pipeline([CSP, StandardScaler, LDA])`                                             |

### Talking points

- "Running `python mybci.py 1 3 train` loads subject 1, run 3, preprocesses the EEG data, and trains the pipeline"
- "We use `cross_val_score` from sklearn with `StratifiedKFold(5)` — this ensures each fold has balanced class proportions"
- "The data is split 80% train / 20% test using `train_test_split` with stratification"
- "We print both the per-fold CV scores array AND the mean, plus the held-out test score"
- "The trained pipeline is serialized to disk with `joblib.dump` for later prediction"

---

## 4. Classification Pipeline — "Predict" ✅

> **Checklist says:** _There is a predict mode, which also uses validation tools.
> The prediction output is displayed (the id of the output class is enough)._

### How to demonstrate

```bash
# First train (if not already done):
python mybci.py 1 3 train

# Then predict:
python mybci.py 1 3 predict
```

### Expected output

```
epoch nb: [prediction] [truth] equal?
epoch 00: [1] [1] True  (inference: 0.45ms) ✓
epoch 01: [2] [1] False (inference: 0.38ms) ✓
epoch 02: [1] [2] False (inference: 0.41ms) ✓
...
Accuracy: 0.XXXX
```

### Key code references

| File          | Function           | What it does                                                             |
| ------------- | ------------------ | ------------------------------------------------------------------------ |
| `mybci.py`    | `cmd_predict()`    | Loads saved model, calls `predict_stream()`                              |
| `pipeline.py` | `load()`           | Deserializes pipeline + held-out test set from `.pkl`                    |
| `pipeline.py` | `predict_stream()` | Iterates epochs, prints `[prediction] [truth] equal?`, computes accuracy |

### Talking points

- "The predict mode loads the serialized pipeline from disk using `joblib.load`"
- "It iterates over the held-out test epochs (never seen during training)"
- "For each epoch, it prints the prediction, ground truth, and whether they match — exactly as shown in the subject PDF examples"
- "Final accuracy is computed and displayed"

---

## 5. Classification Pipeline — "Realtime" ✅

> **Checklist says:** _The prediction is made as the data is streamed to the processing pipeline.
> The program outputs the result between 0 and 2 seconds after the event was triggered._

### How to demonstrate

```bash
# Train first:
python mybci.py 1 3 train

# Stream predict (simulated real-time):
python mybci.py 1 3 predict
```

Or for raw EDF streaming:

```bash
python mybci.py 1 3 predict_raw
```

### Key code references

| File          | Function               | What it does                                                             |
| ------------- | ---------------------- | ------------------------------------------------------------------------ |
| `pipeline.py` | `predict_stream()`     | Simulates streaming with per-epoch timing, enforces `MAX_LATENCY = 2.0s` |
| `pipeline.py` | `predict_raw_stream()` | Reads raw EDF, cuts windows at event onsets, predicts on-the-fly         |
| `pipeline.py` | `MAX_LATENCY = 2.0`    | Hard limit — prints "LATENCY EXCEEDED" if inference > 2s                 |

### Talking points

- "We simulate real-time streaming by iterating over epochs one at a time"
- "Each epoch is timed with `time.perf_counter()` — inference typically takes <1ms, well under the 2-second limit"
- "The output shows the timing for each epoch: `(inference: 0.45ms) ✓`"
- "We have a `MAX_LATENCY = 2.0` constant — if any epoch exceeds 2 seconds, it's flagged"
- "We do NOT use `mne-realtime` — we use manual iteration as required by the subject"
- "The `predict_raw` mode goes even further: it reads the raw EDF file, cuts windows at event onsets, and predicts on-the-fly — simulating a true BCI loop"

---

## 6. Implementation — "Integration" ✅

> **Checklist says:** _Implementation was integrated to sklearn pipeline, inheriting from the
> baseEstimator and transformerMixin classes of sklearn._

### How to demonstrate

```bash
grep -n "BaseEstimator\|TransformerMixin\|class CSP" csp.py
```

Output:

```
6:from sklearn.base import BaseEstimator, TransformerMixin
8:class CSP(BaseEstimator, TransformerMixin):
```

```bash
grep -n "Pipeline" pipeline.py
```

Output:

```
from sklearn.pipeline import Pipeline
Pipeline([("csp", CSP(...)), ("scaler", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])
```

### Key code references

| File          | Line                                         | What it does                                   |
| ------------- | -------------------------------------------- | ---------------------------------------------- |
| `csp.py`      | `class CSP(BaseEstimator, TransformerMixin)` | Custom CSP inherits both sklearn base classes  |
| `csp.py`      | `def fit(self, X, y)`                        | Standard sklearn `.fit()` API                  |
| `csp.py`      | `def transform(self, X)`                     | Standard sklearn `.transform()` API            |
| `pipeline.py` | `build_pipeline()`                           | CSP is used inside `sklearn.pipeline.Pipeline` |

### Talking points

- "Our CSP class inherits from `BaseEstimator` and `TransformerMixin` from sklearn"
- "`BaseEstimator` gives us `get_params()` and `set_params()` — needed for `GridSearchCV` and `cross_val_score`"
- "`TransformerMixin` gives us `fit_transform()` automatically (though we also override it explicitly)"
- "This means our CSP works seamlessly in any sklearn `Pipeline`, `cross_val_score`, `GridSearchCV`, etc."
- "The pipeline is: `CSP → StandardScaler → LDA`"

---

## 7. Implementation — "Implementation" ✅

> **Checklist says:** _A dimensionality reduction algorithm is implemented, the subject talks about
> PCA and CSP but other algorithms performing a dimensionality reduction are feasible.
> Check that the student has a general understanding of the algorithm.
> It is allowed to use functions from libs like numpy or scipy for some tasks:
> the eigenvalues decomposition, singular values decomposition and covariance matrix estimation._

### How to demonstrate

```bash
# Show the CSP implementation:
cat csp.py
```

### Key code references — CSP algorithm step by step

| Step                              | Code in `csp.py`                                                    | Math                         |
| --------------------------------- | ------------------------------------------------------------------- | ---------------------------- |
| 1. Separate epochs by class       | `X1 = X[y == classes[0]]`, `X2 = X[y == classes[1]]`                | Split data by label          |
| 2. Compute per-class covariance   | `_covariance()`: `C = epoch @ epoch.T`, normalize by trace, average | Σ₁, Σ₂                       |
| 3. Composite covariance           | `cov_total = cov1 + cov2 + reg * I`                                 | Σ_total = Σ₁ + Σ₂ + εI       |
| 4. Generalized eigenvalue problem | `eigh(cov1, cov_total)`                                             | Σ₁w = λ(Σ₁+Σ₂)w              |
| 5. Select top-k and bottom-k      | `top_idx`, `bottom_idx`                                             | λ≈1 → class 1, λ≈0 → class 2 |
| 6. Project & log-variance         | `einsum('fc,ect->eft')`, then `log(var)`                            | W^T X → X_CSP                |

### Talking points — understanding the algorithm

- "CSP = Common Spatial Patterns. It finds spatial filters that maximize variance for one class while minimizing it for the other"
- "We compute normalized covariance matrices for each class: `C = X @ X.T / trace(X @ X.T)`"
- "Then we solve the generalized eigenvalue problem: `Σ₁w = λ(Σ₁+Σ₂)w` using `scipy.linalg.eigh`"
- "Eigenvalues near 1 mean the filter captures mostly class-1 variance; near 0 means class-2"
- "We take the top-k and bottom-k eigenvectors as our spatial filters — this gives us features that discriminate between the two classes"
- "The transform step projects epochs onto these filters using `einsum`, then takes log-variance as features"
- "We use Tikhonov regularization (`reg=1e-4`) to handle rank-deficient covariance matrices"
- "We use `scipy.linalg.eigh` for eigenvalue decomposition — this is explicitly allowed by the subject"
- "The covariance is computed manually (`X @ X.T / trace`) — not using `np.cov`"

### If asked about PCA vs CSP

- "PCA finds directions of maximum variance in the data regardless of class labels — it's unsupervised"
- "CSP finds directions that maximize the variance RATIO between classes — it's supervised and specifically designed for BCI"
- "CSP is more appropriate for EEG motor imagery because it exploits the spatial structure of the signal and the class labels"

---

## 8. Implementation — "Score" (Yes/No checkbox) ✅

> **Checklist says:** _There has to be a script executing training over each subject and computing
> the mean of scores over each subjects, by type of experiment runs.
> The mean of the resulting six means (corresponding to the six types of experiment runs)
> has to be superior or equal to 60%._

### How to demonstrate

```bash
python mybci.py
```

### Expected output

```
experiment 0: subject S001: accuracy = 0.XXXX
experiment 0: subject S002: accuracy = 0.XXXX
...
Mean accuracy of the six different experiments for all subjects:
experiment 0: accuracy = 0.XXXX
experiment 1: accuracy = 0.XXXX
experiment 2: accuracy = 0.XXXX
experiment 3: accuracy = 0.XXXX
experiment 4: accuracy = 0.XXXX
experiment 5: accuracy = 0.XXXX

Mean accuracy of 6 experiments: 0.6XXX   ← must be ≥ 0.60
```

### Key code references

| File          | Function             | What it does                                   |
| ------------- | -------------------- | ---------------------------------------------- |
| `mybci.py`    | `cmd_evaluate_all()` | Iterates all 109 subjects × 6 experiment types |
| `mybci.py`    | `EXPERIMENT_RUNS`    | Maps experiment index 0-5 to run pairs         |
| `pipeline.py` | `evaluate_subject()` | Quick 80/20 split evaluation per subject       |

### The 6 experiment types

| Exp | Runs   | Description                              |
| --- | ------ | ---------------------------------------- |
| 0   | 3, 4   | Real left/right hand movement            |
| 1   | 7, 8   | Imagined left/right hand (1st session)   |
| 2   | 11, 12 | Imagined left/right hand (2nd session)   |
| 3   | 5, 6   | Real both hands / feet movement          |
| 4   | 9, 10  | Imagined both hands / feet (1st session) |
| 5   | 13, 14 | Imagined both hands / feet (2nd session) |

### Talking points

- "Running `python mybci.py` with no arguments evaluates ALL 109 subjects across ALL 6 experiment types"
- "For each subject × experiment, we do an 80/20 stratified split, train the pipeline, and report test accuracy"
- "We compute the mean accuracy per experiment type, then the grand mean across all 6 experiments"
- "The grand mean must be ≥ 60% — our implementation achieves this"
- "Data is never-seen test data (held-out 20%) — no overfitting"

---

## 9. Implementation — "Score" (Slider 0–5) 📊

> **Checklist says:** _Over 60% add a point for every 1%.
> Rate it from 0 (failed) through 5 (excellent)._

### Scoring scale

| Mean Accuracy | Points        |
| ------------- | ------------- |
| < 60%         | 0 (fail)      |
| 60%           | 0 (pass)      |
| 61%           | 1             |
| 62%           | 2             |
| 63%           | 3             |
| 64%           | 4             |
| ≥ 65%         | 5 (excellent) |

### How to maximize score

- Run `python mybci.py` and note the final mean accuracy
- Each percentage point above 60% = +1 point on the slider
- Our pipeline uses CSP + LDA which is well-suited for motor imagery BCI

---

## 🎯 Quick Demo Script for Defense

Run these commands in order during the evaluation:

```bash
# 1. Show visualization (Preprocessing - Plot)
# Left/Right hand:
python plot.py one_sample/S001R03.edf
# Both hands/Feet:
python plot.py one_sample/S001R05.edf

# 2. Show the filter range (Preprocessing - Feature extraction)
grep -n "l_freq\|h_freq" preprocess.py
grep -n "CHANNELS" preprocess.py

# 3. Train a model (Classification Pipeline - Train)
python mybci.py 1 3 train

# 4. Predict with streaming (Classification Pipeline - Predict + Realtime)
python mybci.py 1 3 predict

# 5. Show CSP integration (Implementation - Integration)
grep -n "BaseEstimator\|TransformerMixin\|class CSP" csp.py
grep -n "Pipeline" pipeline.py

# 6. Walk through CSP code (Implementation - Implementation)
cat csp.py

# 7. Full evaluation (Implementation - Score)
python mybci.py
```

---

## 📁 File Map

| File            | Purpose                                                            |
| --------------- | ------------------------------------------------------------------ |
| `mybci.py`      | Main CLI entry point — train, predict, evaluate all                |
| `preprocess.py` | EEG loading, filtering, epoching, channel selection                |
| `pipeline.py`   | sklearn Pipeline construction, training, streaming prediction      |
| `csp.py`        | Custom CSP transformer (BaseEstimator + TransformerMixin)          |
| `plot.py`       | Visualization: raw vs filtered, ERD/ERS contrast, feature boxplots |

---

## ⚠️ Common Evaluator Questions & Answers

**Q: Why CSP and not PCA?**

> CSP is supervised — it uses class labels to find spatial filters that maximize discrimination between motor imagery classes. PCA is unsupervised and only finds directions of maximum variance regardless of class. CSP is the standard algorithm for EEG-based BCIs.

**Q: Why 8–30 Hz and not 8–40 Hz?**

> 8–30 Hz covers both mu (8–12 Hz) and beta (13–30 Hz) bands, which are the primary frequency bands showing ERD/ERS during motor imagery. The 30–40 Hz range (low gamma) adds mostly noise for this task. Our range is within the checklist's "~8-40Hz" guideline.

**Q: What is the regularization for?**

> Tikhonov regularization (`reg=1e-4 * I` added to the composite covariance) prevents singular matrix errors when channels are nearly collinear or when there are very few epochs. It's a standard numerical stability technique.

**Q: Why LDA as classifier?**

> LDA (Linear Discriminant Analysis) is the standard classifier for CSP-based BCIs. It's fast, works well with the low-dimensional log-variance features from CSP, and doesn't overfit on small datasets. We tested SVC and RandomForest as well (see TASKS.md).

**Q: How do you ensure no overfitting?**

> Three safeguards: (1) Stratified 80/20 train/test split — test data is never seen during training. (2) 5-fold stratified cross-validation on the training set. (3) The full evaluation (`mybci.py` with no args) reports accuracy on held-out test data across all 109 subjects.

**Q: Why these 7 channels?**

> FC3, FC4, C3, Cz, C4, CP3, CP4 are positioned over the primary motor cortex and supplementary motor areas. These are the electrodes that best capture the mu/beta desynchronization patterns during hand and feet motor imagery.

**Q: Do you use mne-realtime?**

> No. The subject explicitly says not to use mne-realtime. We simulate streaming by iterating over epochs manually with timing enforcement (< 2 seconds per epoch).
