import mne
import matplotlib.pyplot as plt
import numpy as np

# Path to your EDF file (replace with your actual path)
edf_path = "/home/grvelva/TPV/one_sample/S001R01.edf"

# Load data
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw.set_eeg_reference("average", projection=True)

# Get events
events, event_id = mne.events_from_annotations(raw)

# Pick first non-rest event if exists, else first event
first_event_idx = 0
event_sample = events[first_event_idx, 0]
event_time = event_sample / raw.info["sfreq"]

# Define window size and overlap
epoch_length = 2.0  # seconds
overlap = 1.0       # seconds
sfreq = raw.info["sfreq"]
n_samples_epoch = int(epoch_length * sfreq)
n_samples_overlap = int(overlap * sfreq)

# Get 10 seconds of data starting from the event onset
duration = 10.0
start_sample = int(event_time * sfreq)
stop_sample = start_sample + int(duration * sfreq)
data, times = raw.get_data(start=start_sample, stop=stop_sample, picks=[0], return_times=True)

# Create epochs start points (with overlap)
epoch_starts = np.arange(0, len(data[0]) - n_samples_epoch + 1, n_samples_epoch - n_samples_overlap)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(times - times[0], data[0] * 1e6, label="EEG (µV)")
for s in epoch_starts:
    t0 = times[s] - times[0]
    t1 = times[s + n_samples_epoch - 1] - times[0]
    plt.axvspan(t0, t1, color="orange", alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title(f"Segmentation of a {duration}s window starting at event {list(event_id.keys())[0]}")
plt.legend()
plt.tight_layout()
plt.show()

ann = raw.annotations
print("Number of annotations:", len(ann))
for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
    print(f"{desc}: onset={onset:.3f}s  duration={duration:.3f}s")
