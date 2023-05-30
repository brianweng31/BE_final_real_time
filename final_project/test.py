from utils import process_signal, normalize, find_max_segment_power
from utils import get_last_i

import torch
import numpy as np
from collections import Counter

POWER_THRESHOLD = 30

digit2label = ["GestureDown", "GestureLeft", "GestureN",
                   "GestureO", "GestureRight", "GestureUp", 
                   "GestureV", "GestureZ", "Noise"
                   ]
    
label2result = {"GestureDown": "DOWN", "GestureLeft": "LEFT", "GestureN": "N",
                "GestureO": "O", "GestureRight": "RIGHT", "GestureUp": "UP", 
                "GestureV": "V", "GestureZ": "Z", "Noise": None
                }

def get_result(model,x,y,z):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    length = [150]
    model_input = np.zeros((len(length),3,80))
    power = np.zeros((len(length),3))
    max_i = np.zeros((len(length),3))
    
    for i in range(len(length)): 
        x_cut,y_cut,z_cut = get_last_i(x,y,z,length[i])
        signal = {}
        signal['x'] = x_cut
        signal['y'] = y_cut
        signal['z'] = z_cut

        processed_data, power[i], max_i[i] = process_signal(signal)
        model_input[i] = processed_data
    
        if (max_i[i] > length[i]-1-80-5).any() == True:
            return None
        if (power[i] < POWER_THRESHOLD).all() == True:
            return None
    
    model_input = torch.tensor(model_input, dtype=torch.float32)
    model_input = torch.unsqueeze(model_input,1)
    
    # feed to model
    pred = model(model_input.to(device))

    # get final result from result via voting
    results = pred.argmax(1).numpy()
    counter = Counter([element for element in results if element != 8])
    most_common = counter.most_common(1)
    result = most_common[0][0] if most_common else 8
    
    return label2result[digit2label[result]]