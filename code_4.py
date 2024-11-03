# make sure to install libraries on your machine before running


import scipy.io
import matplotlib.pyplot as plt

# Load EEG data from a .mat file
mat_data = scipy.io.loadmat('EEG_P2090.mat')
eeg_data = mat_data['eeg']  # Replace 'eeg' with the actual key in your .mat file

# Data Overview
num_channels, num_samples = eeg_data.shape
sampling_frequency = 500  # Hz
duration_seconds = num_samples / sampling_frequency
duration_minutes = duration_seconds / 60

print(f"Number of Channels: {num_channels}")
print(f"Sample Time (Number of Samples): {num_samples}")
print(f"Sampling Frequency (Fs): {sampling_frequency} Hz")
print(f"Duration of the Recording (in Seconds): {duration_seconds} seconds")
print(f"Duration of the Recording (in Minutes): {duration_minutes} minutes")

# Task 1: Subplot of Specific Ranges of EEG Signals
channel_index = 0  # Example channel index
range1_start, range1_end = 30 * sampling_frequency, 40 * sampling_frequency
range2_start, range2_end = 150 * sampling_frequency, 160 * sampling_frequency
range3_start, range3_end = -5 * sampling_frequency, None  # Last 5 seconds

plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.plot(eeg_data[channel_index, range1_start:range1_end])
plt.title("EEG Signal from 30 to 40 seconds")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(eeg_data[channel_index, range2_start:range2_end])
plt.title("EEG Signal from 150 to 160 seconds")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(eeg_data[channel_index, range3_start:range3_end])
plt.title("Last 5 seconds of EEG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Task 2: Manual Statistical Calculations
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std(data, mean=None):
    if mean is None:
        mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

def calculate_range(data):
    return max(data) - min(data)

# Example usage of Task 2 functions
data_segment = eeg_data[channel_index, 0:1000]  # Example data segment
mean_value = calculate_mean(data_segment)
std_value = calculate_std(data_segment, mean_value)
median_value = calculate_median(data_segment)
range_value = calculate_range(data_segment)

print(f"Mean: {mean_value}, Standard Deviation: {std_value}, Median: {median_value}, Range: {range_value}")
