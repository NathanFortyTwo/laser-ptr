import pyfirmata
import time
board = pyfirmata.Arduino('COM6')

time.sleep(1)
board.digital[8].mode = pyfirmata.SERVO
time.sleep(1)
for k in range(0,50,5):
    time.sleep(0.2)
    board.digital[8].write(k)

