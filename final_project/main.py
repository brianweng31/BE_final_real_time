from model import model
from test import get_result
from shortcut import read_json, shortcut
from utils import get_last_i

import serial
from collections import deque
import numpy as np
import torch

from collections import Counter
import time


COM_PORT = '/dev/cu.usbserial-14140'
BAUD_RATES = 38400
FILE = 'Setting.json'
MODEL_PATH = 'best_model_3axis.pth'

#######################

x = deque([0] * 150, maxlen=150)
y = deque([0] * 150, maxlen=150)
z = deque([0] * 150, maxlen=150)
# rx = deque([0] * 150, maxlen=150)
# ry = deque([0] * 150, maxlen=150)
# rz = deque([0] * 150, maxlen=150)

# data = np.load('../test/BELab/data/GestureUp/chen_4.npz')
# x = data['x']
# y = data['y']
# z = data['z']
# rx = data['rx']
# ry = data['ry']
# rz = data['rz']
# x = deque(x, maxlen=150)
# y = deque(y, maxlen=150)
# z = deque(z, maxlen=150)
# rx = deque(rx, maxlen=150)
# ry = deque(ry, maxlen=150)
# rz = deque(rz, maxlen=150)

if __name__ == '__main__':
    # load json
    shortcut_dict = read_json(FILE)
    # load model
    model = model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    #ser = serial.Serial(COM_PORT, BAUD_RATES)
    try:
        while True:
            # ser.flushInput()
            # data = ser.readline().decode().split('/')

            # while len(data) != 6:
            #     data = ser.readline().decode().split('/')

            
            # x.append(float(data[0]))
            # y.append(float(data[1]))
            # z.append(float(data[2]))
            # rx.append(float(data[3]))
            # ry.append(float(data[4]))
            # rz.append(float(data[5]))
            

            # result = 'UP', 'DOWN', 'LEFT', 'RIGHT', 'V', 'O', 'N', 'Z', None
            start_time = time.time()
            length = [150]
            results = []
            for i in length:
                x_cut,y_cut,z_cut = get_last_i(x,y,z,i)
                results.append(get_result(model,x_cut,y_cut,z_cut))
            print(f'model time = {time.time() - start_time}')

            # print(f'results = {results}')
            counter = Counter([element for element in results if element is not None])
            most_common = counter.most_common(1)
            result = most_common[0][0] if most_common else None
            

            # shortcut
            if result != None:
                print(result)
                # shortcut(shortcut_dict, result)
                

    except KeyboardInterrupt:
        ser.close()    # 清除序列通訊物件