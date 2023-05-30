from model import model
from test import get_result
from shortcut import read_json, shortcut
from utils import get_last_i

import serial
import numpy as np
import torch
import random
from collections import deque
from collections import Counter
from statistics import mean

##### PARAMETERS #####
FILE = 'Setting.json'
MODEL_PATH = './model/best_model_gauss_noise.pth'

# COM_PORT = '/dev/cu.usbserial-14140'
COM_PORT = '/dev/ttyUSB0'
BAUD_RATES = 115200
######################

x = deque([0] * 150, maxlen=150)
y = deque([0] * 150, maxlen=150)
z = deque([0] * 150, maxlen=150)

x_mean = 0
y_mean = 0
z_mean = 0
sigma = 0.01
buf_x = []
buf_y = []
buf_z = []
SKIP = True

if __name__ == '__main__':

    # load json
    shortcut_dict = read_json(FILE)

    # load model
    model = model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # connect to IMU
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    time.sleep(0.5)
    
    try:
        while True:
            # clear buffer every time
            ser.flushInput()
            data = ser.readline().decode().split('/')

            while len(data) != 3:
                print(data)
                data = ser.readline().decode().split('/')
            
            if SKIP:
                buf_x.append(float(data[0]))
                buf_y.append(float(data[1]))
                buf_z.append(float(data[2]))
                x_mean = mean(buf_x)
                y_mean = mean(buf_y)
                z_mean = mean(buf_z)
                if len(buf_x) == 10:
                    # renew x,y,z
                    noise_x = [random.gauss(x_mean,sigma) for _ in range(140)]
                    noise_y = [random.gauss(y_mean,sigma) for _ in range(140)]
                    noise_z = [random.gauss(z_mean,sigma) for _ in range(140)]
                    x = deque(noise_x+buf_x, maxlen=150)
                    y = deque(noise_y+buf_y, maxlen=150)
                    z = deque(noise_z+buf_z, maxlen=150)
                    buf_x = []
                    buf_y = []
                    buf_z = []
                    SKIP = False

            else:
                x.append(float(data[0]))
                y.append(float(data[1]))
                z.append(float(data[2]))

                start_time = time.time()

                # get result from model
                result = get_result(model,x,y,z)
                
                if result != None:
                    print(result)
                    # do shortcut action
                    shortcut(shortcut_dict, result)
                    print(f'{time.time()-start_time:.4f}')
                    print('\n')
                    SKIP = True
                

    except KeyboardInterrupt:
        ser.close()