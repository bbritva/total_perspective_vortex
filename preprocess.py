import sys
import mne
import matplotlib.pyplot as plt  # Import matplotlib for plotting control


def main(path: str):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference('average', projection=True)
    n_channels = len(raw.ch_names)
    print(n_channels)
    raw.plot(n_channels=n_channels, scalings='auto', duration=5, block=True)
    raw.filter(8., 30., fir_design='firwin')
    raw.plot(n_channels=n_channels, scalings='auto', duration=5, block=True)
    raw.compute_psd(fmin=1, fmax=50).plot()
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/train")
        sys.exit(1)
    main(sys.argv[1])