#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.fft import fft, rfft
from scipy.signal import spectrogram
import os
import requests
from numpy import where, arange
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter
from matplotlib.pyplot import *
from scipy import signal
from scipy.io import loadmat


# In[14]:


EEG = loadmat('/Users/home/Desktop/computational N/sessions/ssvep_data_assignment.mat')


# In[15]:


fs = 250
time_step = 1 / 250
n_trial = np.arange(0, 3, time_step)
target_frequencies = [12, 8]

def find_character_index (freq) :
    return int((freq - 8) / 0.2 + 1) - 1


# In[16]:


target_channel_index = np.where(eeg_channels == 'OZ')[0][0]
eeg_3d = eeg_wave[target_channel_index, :, :, :]
print("shape of channel OZ :\n",eeg_3d.shape)



def cal_mean (EEG, freq) :
    eeg_mean_block = []
    for f in freq :
        # extract phase : 
        eeg_selected_phase = EEG[:, :, find_character_index(f)]
        # print("shape of selected phase :\n",np.shape(eeg_selected_phase))
        # calculate mean :
        eeg_mean_block.append(np.mean(eeg_selected_phase, axis=1))
        # print("shape of mean : \n",np.shape(eeg_mean_block))
        
    return np.array(eeg_mean_block)


# In[17]:


Vertical_space = 40
# function to plot eeg signals
def plot_subplot(EEG_signal, time_axis) :
    for targt in range(EEG_signal.shape[0]):
      phase_signal = EEG_signal.iloc[targt, :] + targt * Vertical_space
      plt.plot(time_axis, phase_signal)
      EEG_label = ["EEG 8Hz character", "EEG 12Hz character"]
      eeg_pos = np.arange(EEG_signal.shape[0]) * Vertical_space
      plt.yticks(eeg_pos, EEG_label)
      plt.xlabel('Time(s)')
      plt.ylabel('Target Amplitude [$\mu V^2$]')
      plt.grid(True)
      plt.xlim([0,3])
      print(phase_signal.shape)
      plt.ylim([-phase_signal[:].min(),phase_signal[:].max()])
      plt.plot([0.5, 0.5],[-phase_signal[:].min(),phase_signal[:].max()], 'k', lw=1)
      plt.plot([2.5, 2.5],[-phase_signal[:].min(),phase_signal[:].max()], 'k', lw=1)


      # plt.title(f'EEG signal range {time_axis[0]} to {time_axis[-1]}')
      # savefig(f"./Plot/subplot_between {time_axis[0]} to {time_axis[-1]}", format="PNG")

    plt.show()


# In[18]:


plt.figure(figsize=(20, 8))

eeg_mean = cal_mean(eeg_3d, target_frequencies)
print(np.shape(eeg_mean))

# convert to dataframe :
eeg_mean_df = pd.DataFrame(eeg_mean)
plot_subplot(eeg_mean_df, n_trial)


# In[19]:


freqs, times, Sxx = spectrogram(eeg_mean[0,:],250, nperseg=250,noverlap=0)

print(np.shape(freqs))
print(np.shape(times))
print(np.shape(Sxx))

plt.figure(figsize=(10, 10))
plt.pcolormesh(times, freqs, 10*np.log10(Sxx))
plt.xlabel('time (sec)')
plt.ylabel('freq (db/Hz)')
plt.ylim([0,100])
plt.xlim([0,3])
plt.show()


# In[20]:


freqs, times, Sxx = spectrogram(eeg_mean[1,:],250, nperseg=250,noverlap=0)

print(np.shape(freqs))
print(np.shape(times))
print(np.shape(Sxx))

plt.figure(figsize=(10, 10))
plt.pcolormesh(times, freqs, 10*np.log10(Sxx))
plt.xlabel('time (sec)')
plt.ylabel('freq (db/Hz)')
plt.ylim([0,100])
plt.xlim([0,3])
plt.show()


# In[21]:


eeg_mean = cal_mean(eeg_3d, target_frequencies)
# print(np.shape(eeg_mean))

# calculate welch (power) of 2 signals
f1, power_spectrum_12Hz = signal.welch(eeg_mean[0,:], fs, nperseg=750)
f2, power_spectrum_8Hz = signal.welch(eeg_mean[1,:], fs, nperseg=750)


# In[22]:


plt.figure(figsize=(20, 5))

plt.plot(f2, power_spectrum_8Hz)
plt.title("8 Hz")
plt.xlim([0,50])
plt.grid(True)
plt.xlabel("freq (Hz)")
plt.axvline(7.5,color ='k')
plt.axvline(9.5,color ='k')
plt.axvline(16,color ='k')
plt.axvline(18,color ='k')


# In[23]:


plt.figure(figsize=(20, 5))

plt.plot(f1, power_spectrum_12Hz)
plt.title("12 Hz")
plt.xlim([0,50])
plt.grid(True)
plt.xlabel("freq (Hz)")
plt.axvline(11.5,color ='k')
plt.axvline(13.5,color ='k')
plt.axvline(24,color ='k')
plt.axvline(26,color ='k')


# In[ ]:




