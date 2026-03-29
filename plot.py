#!/usr/bin/env python3
from pathlib import Path
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne.datasets import eegbci


def visualize_preprocessing(
    file: Path,
    channels: tuple = ("C3..", "C4.."),
    freq_band: tuple = (8., 30.),
    t_start: float = 0,
    t_stop: float = 10
) -> None:
    """Plot raw vs band-pass filtered EEG signal for selected channels."""
    if not file.exists():
        print(f"[ERROR] File not found: {file}")
        return

    try:
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
        raw.pick("eeg")
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()
    except Exception as e:
        print(f"[ERROR] Could not load {file.name}: {e}")
        return

    raw_filtered = raw.copy().filter(freq_band[0], freq_band[1], fir_design="firwin", verbose=False)

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 6), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for i, ch in enumerate(channels):
        data_raw, times = raw.copy().pick(ch).get_data(return_times=True)
        data_filt, _ = raw_filtered.copy().pick(ch).get_data(return_times=True)
        mask = (times >= t_start) & (times <= t_stop)
        axes[i].plot(times[mask], data_raw[0, mask] * 1e6, label="Raw", alpha=0.6, color="blue")
        axes[i].plot(times[mask], data_filt[0, mask] * 1e6,
                     label=f"Filtered {freq_band[0]}-{freq_band[1]} Hz", alpha=0.8, color="red")
        axes[i].set_ylabel(f"{ch} (µV)")
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EEG Raw vs Filtered: {file.name}")
    plt.tight_layout()
    plt.show()


def extract_features(X: np.ndarray, sfreq: float = 160.0) -> dict:
    """
    Extract multiple features from epochs (n_epochs, n_channels, n_times).

    Parameters
    ----------
    X     : ndarray, shape (n_epochs, n_channels, n_times)
    sfreq : sampling frequency in Hz (default 160 Hz for BCI2000 dataset)

    Returns
    -------
    dict mapping feature name → ndarray of shape (n_epochs, n_channels)
    """
    features = {}
    features["mean"] = X.mean(axis=2)
    features["var"]  = X.var(axis=2)
    features["ptp"]  = np.ptp(X, axis=2)

    for band, (fmin, fmax) in {"mu": (8, 12), "beta": (13, 30)}.items():
        bp = []
        for epoch in X:
            epoch_bp = []
            for ch_data in epoch:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=256, noverlap=128)
                idx = (freqs >= fmin) & (freqs <= fmax)
                epoch_bp.append(psd[idx].mean())
            bp.append(epoch_bp)
        features[f"{band}_power"] = np.array(bp)
    return features


def plot_features(X: np.ndarray, y: np.ndarray, channels: list) -> None:
    """Boxplot of each feature per class, per channel."""
    features = extract_features(X)
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4), sharey=True)
    if n_features == 1:
        axes = [axes]
    for ax, (feat_name, feat_data) in zip(axes, features.items()):
        for ch_idx, ch in enumerate(channels):
            ax.boxplot(
                [feat_data[y == 1, ch_idx], feat_data[y == 2, ch_idx]],
                labels=["T1", "T2"]
            )
            ax.set_title(f"{feat_name} - {ch}")
    plt.tight_layout()
    plt.show()


# Run-type detection
LRW_RUNS = {3, 4, 7, 8, 11, 12}
WF_RUNS  = {5, 6, 9, 10, 13, 14}


def _detect_run_type(file: Path) -> str:
    """Detect task type from EDF filename. Returns 'lrw', 'wf', or 'unknown'."""
    try:
        run_num = int(file.stem.split("R")[-1])
    except ValueError:
        return "unknown"
    if run_num in LRW_RUNS:
        return "lrw"
    elif run_num in WF_RUNS:
        return "wf"
    return "unknown"


def plot_erd_contrast(
    file: Path,
    picks: tuple | None = None,
    fmin: float = 8,
    fmax: float = 30,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: tuple = (-0.5, 0.0),
    n_points: int = 8
) -> None:
    """Plot ERD/ERS contrast between T1 and T2 conditions for selected channels.
    
    Auto-detects task type from filename:
      - LRW runs (left/right hand): picks C3 vs C4
      - WF runs (hands/feet):       picks Cz vs C3
    """
    task = _detect_run_type(file)
    if picks is None:
        if task == "wf":
            picks = ("Cz..", "C3..")
        else:
            picks = ("C3..", "C4..")

    if task == "wf":
        title = "ERD/ERS Contrast: Hands vs Feet"
        contrast_label = "Hands-Feet"
    else:
        title = "ERD/ERS Contrast: Left vs Right Hand"
        contrast_label = "L-R"

    if not file.exists():
        print(f"[ERROR] File not found: {file}")
        return

    try:
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
        raw.pick("eeg")
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()
        raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
    except Exception as e:
        print(f"[ERROR] Could not load {file.name}: {e}")
        return

    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_map = {"T1": 1, "T2": 2}
    epochs = mne.Epochs(
        raw, events, event_id=event_map,
        tmin=tmin, tmax=tmax, picks="eeg",
        baseline=None, preload=True, verbose=False
    )

    ep = epochs.copy().pick(picks).apply_hilbert(envelope=True)

    ep_bsl = ep.copy().crop(*baseline).get_data()
    bsl = ep_bsl.mean(axis=2, keepdims=True)
    data = ep.get_data()
    erd = (data - bsl) / (bsl + 1e-12) * 100.0

    times = ep.times
    y = ep.events[:, -1]

    contrast = {}
    for i, ch in enumerate(picks):
        mean_T1 = erd[y == 1, i].mean(axis=0)
        mean_T2 = erd[y == 2, i].mean(axis=0)
        diff = mean_T1 - mean_T2
        contrast[ch] = diff

        print(f"\nChannel {ch} contrast ({contrast_label}) at sample points:")
        for t_idx in np.linspace(0, len(times) - 1, n_points, dtype=int):
            t = times[t_idx]
            d = diff[t_idx]
            print(f"  t={t:5.2f}s | {contrast_label}={d:7.2f}")

    plt.figure(figsize=(8, 5))
    for ch, color in zip(picks, ["purple", "green"]):
        plt.plot(times, contrast[ch], label=f"{ch} ({contrast_label})", color=color)
    plt.axhline(0, color="black", ls="--", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel(f"% change ({contrast_label})")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path/to/file.edf>")
        sys.exit(1)

    file = Path(sys.argv[1])
    visualize_preprocessing(file)
    plot_erd_contrast(file)
