from pathlib import Path
import sys
import mne
import matplotlib.pyplot as plt
import numpy as np

def visualize_preprocessing(file: Path,
                                    channels=("C3..", "C4.."),
                                    freq_band=(8., 30.),
                                    t_start=0, t_stop=10):
    # --- Load raw data ---
    raw = mne.io.read_raw_edf(file, preload=True)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=True)

    # --- Apply band-pass filter ---
    raw_filtered = raw.copy().filter(freq_band[0], freq_band[1], fir_design="firwin")

    # --- Plot raw and filtered together ---
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 6), sharex=True)
    for i, ch in enumerate(channels):
        data_raw, times = raw.copy().pick(ch).get_data(return_times=True)
        data_filt, _ = raw_filtered.copy().pick(ch).get_data(return_times=True)
        mask = (times >= t_start) & (times <= t_stop)
        axes[i].plot(times[mask], data_raw[0, mask]*1e6, label="Raw", alpha=0.6, color="blue")
        axes[i].plot(times[mask], data_filt[0, mask]*1e6, label=f"Filtered {freq_band[0]}-{freq_band[1]} Hz", alpha=0.8, color="red")
        axes[i].set_ylabel(f"{ch} (µV)")
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EEG Raw vs Filtered: {file.name}")
    plt.tight_layout()
    plt.show()


def extract_features(X):
    """Extract multiple features from epochs (n_epochs, n_channels, n_times)."""
    features = {}

    # Mean per epoch per channel
    features["mean"] = X.mean(axis=2)
    # Variance per epoch per channel
    features["var"] = X.var(axis=2)
    # Peak-to-peak
    features["ptp"] = np.ptp(X, axis=2)
    # Bandpower in mu and beta
    sfreq = 160  # BCI2000 dataset
    for band, (fmin, fmax) in {"mu": (8,12), "beta": (13,30)}.items():
        bp = []
        for epoch in X:
            epoch_bp = []
            for ch_data in epoch:
                psd, freqs = plt.psd(ch_data, NFFT=256, Fs=sfreq, noverlap=128)
                idx = (freqs >= fmin) & (freqs <= fmax)
                epoch_bp.append(psd[idx].mean())
            bp.append(epoch_bp)
        features[f"{band}_power"] = np.array(bp)
    return features

def plot_features(X, y, channels):
    features = extract_features(X)
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4), sharey=True)
    if n_features == 1:
        axes = [axes]
    for ax, (feat_name, feat_data) in zip(axes, features.items()):
        for ch_idx, ch in enumerate(channels):
            ax.boxplot([feat_data[y==1,ch_idx], feat_data[y==2,ch_idx]],
                       labels=["T1","T2"])
            ax.set_title(f"{feat_name} - {ch}")
    plt.tight_layout()
    plt.show()

def plot_erd_contrast(file, picks=("C3..", "C4.."), fmin=8, fmax=30,
                      tmin=-0.2, tmax=0.5, baseline=(-0.5, 0.0), n_points=8):
    raw = mne.io.read_raw_edf(file, preload=True)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=True)
    raw.filter(fmin, fmax, fir_design="firwin")

    events, _ = mne.events_from_annotations(raw)
    event_map = {"T1": 1, "T2": 2}  # left vs right hand
    epochs = mne.Epochs(raw, events, event_id=event_map,
                        tmin=tmin, tmax=tmax, picks="eeg",
                        baseline=None, preload=True)

    # Hilbert envelope
    ep = epochs.copy().pick(picks).apply_hilbert(envelope=True)

    # Baseline normalization
    ep_bsl = ep.copy().crop(*baseline).get_data()
    bsl = ep_bsl.mean(axis=2, keepdims=True)
    data = ep.get_data()
    erd = (data - bsl) / (bsl + 1e-12) * 100.0

    times = ep.times
    y = ep.events[:, -1]

    contrast = {}
    for i, ch in enumerate(picks):
        mean_T1 = erd[y==1, i].mean(axis=0)
        mean_T2 = erd[y==2, i].mean(axis=0)
        diff = mean_T1 - mean_T2
        contrast[ch] = diff

        print(f"\nChannel {ch} contrast (Left - Right) at sample points:")
        for t_idx in np.linspace(0, len(times)-1, n_points, dtype=int):
            t = times[t_idx]
            d = diff[t_idx]
            print(f"  t={t:5.2f}s | L-R=%7.2f" % d)

    # Plot contrast
    plt.figure(figsize=(8,5))
    for ch, color in zip(picks, ["purple","green"]):
        plt.plot(times, contrast[ch], label=f"{ch} (L-R)", color=color)
    plt.axhline(0, color="black", ls="--", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("% change (L-R)")
    plt.title("ERD/ERS Contrast: Left vs Right Hand")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = Path("./one_sample/S001R03.edf")
    plot_erd_contrast(file)
    plot_lateralization_index(file)
