import numpy as np
import os
import torch
# from scipy import interpolate
from scipy.interpolate import CubicSpline
from collections import deque

import time

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

def preprocess_data(file_path):
    X = []
    y = []
    num_of_bad_data = 0
    label2digit = {"GestureDown":0, "GestureLeft":1, "GestureN":2,
                   "GestureO": 3, "GestureRight":4, "GestureUp":5, 
                   "GestureV": 6, "GestureZ": 7, "Noise":8
                   }
    for folder in os.listdir(file_path):
        if os.path.isdir(f"{file_path}/{folder}"):
            for file in os.listdir(f"{file_path}/{folder}"):
                if os.path.isfile(f"{file_path}/{folder}/{file}"):
                    if file.split('.')[1] == 'npz' and file.split('.')[0].split('_')[0]!='hung':
                        signal = np.load(f"{file_path}/{folder}/{file}")
                        is_good_data = True
                        if is_good_data:
                            signal = process_signal(signal)
                            if not np.isnan(signal).any() :
                                X.append(np.expand_dims(signal, 0).tolist())
                                y.append(label2digit[folder])
                            else:
                                num_of_bad_data += 1
                                print(f"{file_path}/{folder}/{file}")
    print(f'Total {num_of_bad_data} bad data!')
    return torch.tensor(X), torch.tensor(y)