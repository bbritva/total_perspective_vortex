import sys
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bandpower(psd, freqs, fmin, fmax):
    """Integrate the PSD in a specific frequency band."""
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapezoid(psd[:,:, idx_band], freqs[idx_band], axis=2)  # integrate along freq axis


def main(path: str):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference('average', projection=True)
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




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/edf_file")
        sys.exit(1)
    main(sys.argv[1])