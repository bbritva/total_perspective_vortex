import sys
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def bandpower(epoch, sfreq, fmin, fmax):
    psd, _ = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return psd.mean(axis=1)  # average across frequencies

def load_file(path: str):
    run_num = int(path.stem.split("R")[-1])
    if run_num not in {3,4,7,8,11,12}:
        return None, None  # skip baselines

    # Load EDF
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference("average", projection=True)
    raw.filter(8., 30., fir_design="firwin")

    # Events
    events, _ = mne.events_from_annotations(raw)
    event_map = {"T1": 1, "T2": 2}

    # Epoching
    epochs = mne.Epochs(raw, events, event_id=event_map,
                        tmin=0.0, tmax=2.0,
                        baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y


def plot_sample(file: Path):

    channels = ["C3..", "C4.."]   # motor cortex electrodes
    t_start, t_stop = 5, 10               # seconds of data to visualize

    # === LOAD RAW ===
    raw = mne.io.read_raw_edf(file, preload=True)

    raw.pick_types(eeg=True)  # keep only EEG
    raw.set_eeg_reference("average")

    # Make a copy for filtering
    raw_filtered = raw.copy().filter(l_freq=8., h_freq=30.)

    # === TIME SERIES SNIPPET ===
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 6), sharex=True)

    for i, ch in enumerate(channels):
        # raw
        data_raw, times = raw.copy().pick(ch).get_data(return_times=True)
        # filtered
        data_filt, _ = raw_filtered.copy().pick(ch).get_data(return_times=True)

        # select snippet
        mask = (times >= t_start) & (times <= t_stop)
        axes[i].plot(times[mask], data_raw[0, mask]*1e6, label="Raw", alpha=0.6)
        axes[i].plot(times[mask], data_filt[0, mask]*1e6, label="Filtered (8-30 Hz)", alpha=0.8)
        axes[i].set_ylabel(f"{ch} (µV)")
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EEG Snippet Before vs After Filtering")
    plt.tight_layout()
    plt.show()

    # === PSD BEFORE & AFTER ===
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    raw.plot_psd(fmax=60, picks=channels, ax=ax, show=False, color="blue", average=True)
    raw_filtered.plot_psd(fmax=60, picks=channels, ax=ax, show=False, color="red", average=True)
    ax.set_title("Power Spectral Density: Raw (blue) vs Filtered (red)")
    plt.show()

    # === FEATURE EXTRACTION ===
    # Events for left vs right hand
    events, _ = mne.events_from_annotations(raw)
    event_map = {"T1": 1, "T2": 2}
    epochs = mne.Epochs(raw_filtered, events, event_id=event_map,
                        tmin=0.0, tmax=2.0, baseline=None, preload=True)
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    features = []
    sfreq = raw.info["sfreq"]
    for epoch in X:
        mu = bandpower(epoch, sfreq, 8, 13)
        beta = bandpower(epoch, sfreq, 13, 30)
        features.append(np.hstack([mu, beta]))
    features = np.array(features)  # (n_epochs, n_channels*2)

    # === SCATTER PLOT: C3 vs C4 mu power ===
    plt.figure(figsize=(6, 5))
    for label, color in zip([1, 2], ["blue", "red"]):
        mask = y == label
        plt.scatter(features[mask, 0], features[mask, 1], c=color, label=f"Class {label}", alpha=0.7)
    plt.xlabel("C3 mu power")
    plt.ylabel("C4 mu power")
    plt.title("Left vs Right hand - Mu power")
    plt.legend()
    plt.show()


def plot_features_summary(file: Path):
    import itertools

    # Motor electrodes
    channels = ["C3..", "C4..", "C1..", "C2.."]

    # Load and preprocess
    raw = mne.io.read_raw_edf(file, preload=True)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=True)
    raw_filtered = raw.copy().filter(8., 30.)

    # Epoching for left vs right hand
    run_num = int(file.stem.split("R")[-1])
    if run_num not in {3,4,7,8,11,12}:
        print("Skipping baselines or non-hand runs")
        return
    events, _ = mne.events_from_annotations(raw)
    event_map = {"T1": 1, "T2": 2}
    epochs = mne.Epochs(raw_filtered, events, event_id=event_map,
                        tmin=0.0, tmax=2.0, baseline=None, preload=True)
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]
    sfreq = raw.info["sfreq"]

    # Feature functions
    def bandpower(epoch, fmin, fmax):
        psd, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq,
                                                         fmin=fmin, fmax=fmax, verbose=False)
        return psd.mean(axis=1)

    def signal_energy(epoch):
        return np.sum(epoch**2, axis=1)

    def signal_variance(epoch):
        return np.var(epoch, axis=1)

    # Compute features
    features = []
    for epoch in X:
        mu = bandpower(epoch, 8, 13)
        beta = bandpower(epoch, 13, 30)
        energy = signal_energy(epoch)
        var = signal_variance(epoch)
        features.append(np.hstack([mu, beta, energy, var]))
    features = np.array(features)  # shape: n_epochs x (n_channels*4)
    
    feature_names = ["mu", "beta", "energy", "variance"]

    # --- Scatter summary plots ---
    for i, ch in enumerate(channels):
        ch_features = features[:, i*4:(i+1)*4]  # features for this channel
        pairs = list(itertools.combinations(range(4), 2))
        for (f1, f2) in pairs:
            plt.figure(figsize=(6, 5))
            for cls, color in zip([1, 2], ["blue", "red"]):
                mask = y == cls
                plt.scatter(ch_features[mask, f1], ch_features[mask, f2],
                            alpha=0.7, label=f"Class {cls}", edgecolors='k')
            plt.xlabel(feature_names[f1])
            plt.ylabel(feature_names[f2])
            plt.title(f"{ch}: {feature_names[f1]} vs {feature_names[f2]}")
            plt.legend()
            plt.show()

    # --- PSD using modern API ---
    print("Plotting PSD for selected channels...")

    epochs_T1 = epochs['T1']
    epochs_T2 = epochs['T2']

    # Compute PSD (returns shape: n_epochs x n_channels x n_freqs)
    psd_T1 = epochs_T1.compute_psd(fmin=8, fmax=30, method='welch').get_data()
    psd_T2 = epochs_T2.compute_psd(fmin=8, fmax=30, method='welch').get_data()

    # Average across epochs
    psd_T1_mean = psd_T1.mean(axis=0)  # shape: n_channels x n_freqs
    psd_T2_mean = psd_T2.mean(axis=0)

    # Plot manually
    freqs = epochs_T1.compute_psd(fmin=8, fmax=30, method='welch').freqs

    plt.figure(figsize=(8,5))
    plt.plot(freqs, psd_T1_mean[0], label='T1', color='blue')  # first channel
    plt.plot(freqs, psd_T2_mean[0], label='T2', color='red')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (uV^2/Hz)")
    plt.title("PSD per Class for Channel C3")
    plt.legend()
    plt.show()

    # psd = raw_filtered.compute_psd(picks=channels, fmin=0, fmax=60, average="mean")
    # psd.plot()
    plt.show()



def main(path:Path):
    X_list, y_list = [], []
    edf_files = sorted(path.glob("*.edf"))

    plot_features_summary(edf_files[2]) 

    # for f in edf_files:
    #     X, y = load_file(f)
    #     if X is not None:
    #         X_list.append(X)
    #         y_list.append(y)
    # X = np.concatenate(X_list, axis=0)
    # y = np.concatenate(y_list, axis=0)

    # print(f"Total epochs: {len(y)}, shape of X: {X.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/edf_file")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_dir():
        print("Provided path is not a folder")
        sys.exit(1)
    main(path)


def main_old(path: str):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference('average', projection=True)
    raw.filter(8., 30., fir_design="firwin")

    n_channels = len(raw.ch_names)
    sfreq = raw.info['sfreq']
    print(n_channels)

    # 2. Get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    print("Event IDs:", event_id)


    # 3. Create epochs (2s window, no baseline correction)
    # Adjust event_id mapping according to your dataset's labels
    # Example: {'wrist': 1, 'eyes_closed': 2}
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0, tmax=2,
        baseline=None, preload=True
    )

    # 4. Compute PSD per epoch
    psd = epochs.compute_psd(fmin=1, fmax=50, method='welch', n_fft=int(sfreq * 2))
    psd_data, freqs = psd.get_data(return_freqs=True)
    # psd shape = (n_epochs, n_channels, n_freqs)
    print("psd_data shape:", psd_data.shape)   # (n_epochs, n_channels, n_freqs)
    print("freqs shape:", freqs.shape)         # (n_freqs,)

    # 5. Extract band powers
    alpha = bandpower(psd_data, freqs, 8, 13)   # shape: (n_epochs, n_channels)
    beta  = bandpower(psd_data, freqs, 13, 30)

    # 6. Prepare DataFrame
    ch_names = epochs.ch_names
    df = pd.DataFrame()

    for i_epoch in range(len(epochs)):
        row = {
            "epoch": i_epoch,
            "label": epochs.events[i_epoch, 2]  # numeric event code
        }
        for i_ch, ch in enumerate(ch_names):
            row[f"{ch}_alpha"] = alpha[i_epoch, i_ch]
            row[f"{ch}_beta"]  = beta[i_epoch, i_ch]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # 7. Save to CSV
    df.to_csv("eeg_bandpowers.csv", index=False)
    print("Saved features to eeg_bandpowers.csv")




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