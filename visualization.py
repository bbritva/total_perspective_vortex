#!/usr/bin/env python3
import argparse
import mne
import matplotlib.pyplot as plt
from mne.datasets import eegbci

psd_params = {
    "fmax" : 40,
    "method": "welch",
    "n_fft": 256,
    "n_overlap": 128
}


def load_and_preprocess(path):
    # Load EDF file
    raw = mne.io.read_raw_edf(path, preload=True, stim_channel=None)
    eegbci.standardize(raw)  # sets channel names to standard 10-20 labels
    # attach standard 10-20 montage for channel positions
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    print(raw.info)
    filtered = raw.copy().filter(l_freq=7., h_freq=30.)

    return raw, filtered


def show_psd(raw, filtered):
    raw.compute_psd(**psd_params).plot(spatial_colors=True)
    filtered.compute_psd(**psd_params).plot(spatial_colors=True)
    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(
        description="Parse, visualize, and preprocess PhysioNet motor imagery EEG data."
    )
    parser.add_argument("edf_path", type=str, help="Path to an EDF recording")
    args = parser.parse_args()

    raw, filtered = load_and_preprocess(args.edf_path)
    show_psd(raw, filtered)


if __name__ == "__main__":
    main()
