import numpy as np
import os
import torch
from scipy.interpolate import CubicSpline
from collections import deque

def get_last_i(x,y,z,i):
    x_cut = list(deque(x, maxlen=i)) if i < 150 else list(x)
    y_cut = list(deque(y, maxlen=i)) if i < 150 else list(y)
    z_cut = list(deque(z, maxlen=i)) if i < 150 else list(z)

    return x_cut,y_cut,z_cut

def normalize(signal):
    return signal - np.mean(signal)

def find_max_segment_power(signal, length):
    max_i = signal.shape[0]-length
    max_pow = -1
    for i in reversed(range(signal.shape[0] - length)):
        if np.square(signal[i:i+length]).sum() > max_pow:
            max_i = i
            max_pow = np.square(signal[i:i+length]).sum()
    return signal[max_i:max_i+length], max_pow, max_i

def resample_array(array, new_length):
    old_length = len(array)
    x = np.linspace(0, 1, old_length)  # Normalized x-coordinates
    x_new = np.linspace(0, 1, new_length)  # Normalized new x-coordinates
    cs = CubicSpline(x, array)  # Cubic spline interpolation
    resampled_array = cs(x_new)  # Perform interpolation
    return resampled_array

def process_signal(signal):
    data = np.array([
        signal['x'],
        signal['y'], 
        signal['z'],
    ])

    length = 80

    process_data = np.zeros((data.shape[0], length))
    power = np.zeros(data.shape[0])
    max_i = np.zeros(data.shape[0])

    for i in range(len(data)):
        data[i] = normalize(data[i])
        if data.shape[1] > 80:
            process_data[i], power[i], max_i[i] = find_max_segment_power(data[i], length)
        else: # data.shape[1] <= 80:
            process_data[i] = resample_array(data[i], length)
            power[i] = np.square(process_data[i]).sum()
    
    return process_data, power, max_i