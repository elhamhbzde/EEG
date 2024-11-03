import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import butter, filtfilt, windows, spectrogram, find_peaks
from scipy.stats import entropy
import mne

# Load EEG data
file_path = './sampleEEGdata.mat'  # Update with the correct file path
mat_data = scipy.io.loadmat(file_path)
eeg_data = mat_data['EEG']
eeg_signal_data = eeg_data['data'][0, 0]
sampling_rate = int(eeg_data['srate'][0, 0][0, 0])
num_channels = int(eeg_data['nbchan'][0, 0][0, 0])
num_trials = int(eeg_data['trials'][0, 0][0, 0])
times = eeg_data['times'][0, 0].flatten()
channel_names = [eeg_data['chanlocs'][0, 0]['labels'][0, i][0] for i in range(num_channels)]

# Define Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data)

# Select and process trials
# Specify representative channels for Central, Parietal, Frontal, and Occipital regions
representative_channels = {'Central': 'Cz', 'Parietal': 'Pz', 'Frontal': 'Fz', 'Occipital': 'Oz'}
channel_indices = {region: channel_names.index(channel) for region, channel in representative_channels.items()}
lowcut, highcut, order = 1, 30, 4

# Time-frequency analysis
def compute_stft(data, fs, window='hann', nperseg=128):
    f, t, Zxx = spectrogram(data, fs, window=window, nperseg=nperseg, noverlap=nperseg//2, detrend=False)
    return f, t, np.abs(Zxx)

# Define the ONSET of stimulating section
# Assuming a hypothetical threshold for stimulus detection
def find_stimulus_onset(data, threshold=0.5):
    peaks, _ = find_peaks(data, height=threshold)
    return peaks[0] if peaks.size > 0 else None

# Calculate reaction times
# Hypothetically assuming stimulus and response events are known
stimulus_times = np.array([100, 150, 200])  # Replace with actual stimulus times
response_times = np.array([120, 170, 220])  # Replace with actual response times
reaction_times = response_times - stimulus_times

# Plot Topo plots
# Hypothetically assuming channel locations are known
# channel_locs = ...  # Provide the channel location data

# Create an MNE Info object for topographic plots
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channel_names, channel_locs)))
info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types='eeg')
info.set_montage(montage)

# Compute the mean data across trials for each channel
mean_data = np.mean(eeg_signal_data, axis=2)

# Create an MNE Evoked object
evoked = mne.EvokedArray(mean_data, info)

# Plot the topomap
evoked.plot_topomap(times=[0.1, 0.2, 0.3], time_unit='s')  # Adjust times as needed

# Calculate and plot entropy
def calculate_entropy(signal):
    hist, _ = np.histogram(signal, bins=20, density=True)
    return entropy(hist)

# Calculate entropy for first 30 and last 30 trials
entropy_first_30 = [calculate_entropy(trial) for trial in eeg_signal_data[:, :, :30].reshape(-1, eeg_signal_data.shape[1])]
entropy_last_30 = [calculate_entropy(trial) for trial in eeg_signal_data[:, :, -30:].reshape(-1, eeg_signal_data.shape[1])]

# Plot the entropies
plt.figure(figsize=(10, 5))
plt.plot(entropy_first_30, label='First 30 Trials')
plt.plot(entropy_last_30, label='Last 30 Trials')
plt.xlabel('Channel')
plt.ylabel('Entropy')
plt.title('Entropy of EEG Channels')
plt.legend()
plt.show()