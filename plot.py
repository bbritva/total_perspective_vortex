from pathlib import Path
import sys
import mne
import matplotlib.pyplot as plt

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




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py /path/to/edf_file")
        sys.exit(1)
    path = Path(sys.argv[1])
    if path.is_file():
        visualize_preprocessing(path)
    else:
        print("Provided path is not a correct file/folder")