import sys
import mne

def main(path: str):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference('average', projection=True)
    raw.plot(n_channels=64, scalings='auto', duration=5, block=True)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/train")
        sys.exit(1)
    main(sys.argv[1])