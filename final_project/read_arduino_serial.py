import serial
from collections import deque
import numpy as np
import time


COM_PORT = '/dev/cu.usbserial-14140'
BAUD_RATES = 38400



# Initialize the x, y, z, and t data
# x = np.zeros(1000)
# y = np.zeros(1000)
# z = np.zeros(1000)
# rx = np.zeros(1000)
# ry = np.zeros(1000)
# rz = np.zeros(1000)
x = deque([0] * 1000, maxlen=1000)
y = deque([0] * 1000, maxlen=1000)
z = deque([0] * 1000, maxlen=1000)
rx = deque([0] * 1000, maxlen=1000)
ry = deque([0] * 1000, maxlen=1000)
rz = deque([0] * 1000, maxlen=1000)



Gestures = ['Up', 'Down', 'Left', 'Right', 'O', 'Z', 'V', 'N']
filename = './data/GestureN/'
people = 'chen'

ser = serial.Serial(COM_PORT, BAUD_RATES)

try:
    i = 1000
    count = 0
    start = 1500
    end = 1650

    ser.flushInput()
    while True:
        ser.flushInput()
        data = ser.readline().decode().split('/')

        while len(data) != 6:
            data = ser.readline().decode().split('/')

        
        # x = np.append(x, float(data[0]))
        # y = np.append(y, float(data[1]))
        # z = np.append(z, float(data[2]))
        # rx = np.append(rx, float(data[3]))
        # ry = np.append(ry, float(data[4]))
        # rz = np.append(rz, float(data[5]))
        x.append(float(data[0]))
        y.append(float(data[1]))
        z.append(float(data[2]))
        rx.append(float(data[3]))
        ry.append(float(data[4]))
        rz.append(float(data[5]))

        result = model(x,y,z,rx,ry,rz)

            

        # Pause the loop for a short period of time to create a real-time animation
        # print(f'i = {i}: {data[0]}, {data[1]}, {data[2]}, {data[3]}, {data[4]}, {data[5]} | Time: {time.time() - start}')

        # plt.pause(0.001)

  
        i += 1

except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    




















