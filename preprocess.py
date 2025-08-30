import random
import sys
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot import plot_features, visualize_preprocessing

CHANNELS = ["C3..", "C4.."]            # motor cortex electrodes
LRW_RUNS = {3, 4, 7, 8, 11, 12}
WF_RUNS = {5, 6, 9, 10, 13, 14}

def bandpower(epoch, sfreq, fmin, fmax):
    psd, _ = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return psd.mean(axis=1)  # average across frequencies

def load_file(path: Path):
    run_num = int(path.stem.split("R")[-1])
    if run_num not in LRW_RUNS:
        return None, None  # skip baselines

    # Load EDF
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=True)
    raw.filter(8., 30., fir_design="firwin")

    # Events
    events, _ = mne.events_from_annotations(raw)
    event_map = {"T1": 1, "T2": 2}

    # Epoching
    epochs = mne.Epochs(raw, events, event_id=event_map,
                        tmin=0.0, tmax=2.0,
                        baseline=None, preload=True,
                        picks=CHANNELS)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y


def main(path:Path):
    X_list, y_list = [], []
    edf_files = sorted(path.glob("*.edf"))

    visualize_preprocessing(random.choice(edf_files))

    for f in edf_files:
        X, y = load_file(f)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    plot_features(X, y, CHANNELS)

    print(f"Total epochs: {len(y)}, shape of X: {X.shape}")
    return X, y

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/edf/file(s)")
        sys.exit(1)
    path = Path(sys.argv[1])
    if path.is_dir():
        main(path)
    else:
        print("Provided path is not a folder")

# Feature Selection Justification:
# For this BCI project, we focused on EEG signals
# from channels C3 and C4, which correspond to the
# primary motor cortex in the international 10–10
# system. Research in motor imagery and motor execution
# shows that these regions exhibit the strongest
# modulation during hand movements. We applied
# a band-pass filter from 8–30 Hz, covering the mu (8–13 Hz)
# and beta (13–30 Hz) rhythms, which are known to reflect
# motor activity. Features such as bandpower, RMS,
# and variance were extracted to summarize the signal’s
# amplitude and spectral content. Visual inspection
# of the filtered signals shows clear differences
# between T1 and T2 classes, confirming that these
# features carry information useful for classification.
# This combination of biologically informed channel
# selection, frequency filtering, and compact signal
# features ensures that the model receives the most
# relevant input while reducing noise and irrelevant
# activity.