from utils import process_signal, normalize, find_max_segment_power
from utils import get_last_i

import torch
import numpy as np
from collections import Counter

import time

POWER_THRESHOLD = 0

def get_result(model,x,y,z):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # signal = {}
    # signal['x'] = x
    # signal['y'] = y
    # signal['z'] = z
    # signal['rx'] = rx
    # signal['ry'] = ry
    # signal['rz'] = rz

    # processed_data = process_signal(signal)

    length = [150]
    model_input = np.zeros((len(length),3,80))
    power = np.zeros((len(length),3))
    max_i = np.zeros((len(length),3))
    
    start_time = time.time()
    for i in range(len(length)):
        # print(i)
        
        x_cut,y_cut,z_cut = get_last_i(x,y,z,length[i])
        signal = {}
        signal['x'] = x_cut
        signal['y'] = y_cut
        signal['z'] = z_cut
        # signal['rx'] = rx_cut
        # signal['ry'] = ry_cut
        # signal['rz'] = rz_cut

        
        processed_data, power[i], max_i[i] = process_signal(signal)
        model_input[i] = processed_data
    
        # print(f'max_i = {max_i.reshape((1,3))}')
        # print(f'(max_i == 69).all() =  {(max_i == 69).all()}')
        if (max_i[i] == 149-length[i]).any() == True:
            print(f'(max_i[i] == 149-length[i]).any() == True')
        #     return None
    print(f'process time = {time.time()-start_time}')
    # test if >= POWER_THRESHOLD
    # print(f'power = {power}')
    # if power.all() < POWER_THRESHOLD:
    #     return None

    # signal = torch.tensor(signal, dtype=torch.float32)
    # signal = torch.unsqueeze(torch.unsqueeze(signal,0),0)
    # print(f'signal.dtype = {signal.dtype}')
    
    # pred = model(signal.to(device))
    model_input = torch.tensor(model_input, dtype=torch.float32)
    model_input = torch.unsqueeze(model_input,1)
    # print(f'model_input.dtype = {model_input.dtype}')
    
    start_time = time.time()
    pred = model(model_input.to(device))
    print(f'model_time = {time.time()-start_time}')

    # print(f'pred.shape = {pred.shape}')
    # print(f'pred = {pred}')

    results = pred.argmax(1)
    # print(f'results = {results}')

    counter = Counter([element for element in results if element is not 8])
    most_common = counter.most_common(1)
    result = most_common[0][0] if most_common else 8
    # print(f'result = {result}')

    digit2label = ["GestureDown", "GestureLeft", "GestureN",
                   "GestureO", "GestureRight", "GestureUp", 
                   "GestureV", "GestureZ", "Noise"
                   ]
    
    label2result = {"GestureDown": "DOWN", "GestureLeft": "LEFT", "GestureN": "N",
                   "GestureO": "O", "GestureRight": "RIGHT", "GestureUp": "UP", 
                   "GestureV": "V", "GestureZ": "Z", "Noise": None
                   }
    
    if label2result[digit2label[result]] != None:
        print(f'max_i = {max_i}')
    
    return label2result[digit2label[result]]