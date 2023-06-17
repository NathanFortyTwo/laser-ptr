import cv2
import time
import numpy as np

import matplotlib.pyplot as plt
import pyfirmata

board = pyfirmata.Arduino('COM6')

TARGET_COLOR = [255, 0, 0][::-1]  # we reverse to make it RGB and not BGR
POINTER_COLOR = [0, 0, 255][::-1]
CAMERA_INDEX = 0
FRAMERATE = 60
KERNEL_SIZE = 5
TARGET_MAT = np.array([[TARGET_COLOR]*KERNEL_SIZE]*KERNEL_SIZE)
POINTER_MAT = np.array([[POINTER_COLOR]*KERNEL_SIZE]*KERNEL_SIZE)

MODE_TEST = 0

SERVO_HORIZONTAL = 8
SERVO_VERTICAL = 13

board.digital[SERVO_HORIZONTAL].mode = pyfirmata.SERVO
board.digital[SERVO_VERTICAL].mode = pyfirmata.SERVO

SERVO_DELTA_X = 15
SERVO_DELTA_Y = 15

X_STEP = KERNEL_SIZE
Y_STEP = KERNEL_SIZE
# this is for optimization purposes, so we don't have to calculate the distance for every pixel in the image


def testing(func):  # any function decorated with this will only be executed if MODE_TEST is True
    def wrapper(*args, **kwargs):
        if MODE_TEST:
            return func(*args, **kwargs)
    return wrapper


def dist(mat1, mat2):
    return np.linalg.norm(mat1-mat2)


# returns a KERNEL_SIZE x KERNEL_SIZE matrix of pixels around the given coordinates
def get_local_pixel(array, x, y):
    res = array[y:y+TARGET_MAT.shape[0], x:x+TARGET_MAT.shape[1]]
    if res.shape != TARGET_MAT.shape:
        raise IndexError
    return res


def find_spot(frame):
    global L
    dico = {}
    L = []
    array = np.array(frame)
    Y, X, _ = array.shape

    for x in range(0, X, X_STEP):
        for y in range(0, Y, Y_STEP):
            try:
                local_mat = get_local_pixel(array, x, y)
            except IndexError:  # if the local matrix is out of bounds, we just skip it
                continue

            distance = dist(local_mat, TARGET_MAT)
            L.append(distance)
            dico[distance] = (x, y)

    return dico[min(L)]


def find_pointer(frame):
    global L
    dico = {}
    L = []
    array = np.array(frame)
    Y, X, _ = array.shape

    for x in range(0, X, X_STEP):
        for y in range(0, Y, Y_STEP):
            try:
                local_mat = get_local_pixel(array, x, y)
            except IndexError:  # if the local matrix is out of bounds, we just skip it
                continue

            distance = dist(local_mat, POINTER_MAT)
            L.append(distance)
            dico[distance] = (x, y)

    return dico[min(L)]


def get_left_right(coords, target):
    x, y = coords
    x_target, y_target = target
    if x < x_target:
        return 1
    else:
        return -1


def get_up_down(coords, target):
    x, y = coords
    x_target, y_target = target
    if y < y_target:
        return 1
    else:
        return -1


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    #arduino = serial.Serial('COM3', 9600)
    old_vertical = 90
    old_horizontal = 90

    while True:
        # time.sleep(1/20)  # no need to sleep, the processing takes time
        ret, frame = cap.read()

        coords_target = find_spot(frame)
        my_coords = find_pointer(frame)

        add_cross_to_coords(coords_target, frame)
        add_cross_to_coords(my_coords, frame, (0, 255, 0))
        cv2.imshow("frame", frame)
        # control servomotors part

        horizontal_coef = get_left_right(my_coords, coords_target)
        vertical_coef = get_up_down(my_coords, coords_target)
        try:

            board.digital[SERVO_HORIZONTAL].write(
                old_horizontal+horizontal_coef*SERVO_DELTA_X)

            old_horizontal += horizontal_coef*SERVO_DELTA_X

        except ValueError:
            print("Out of bounds : horizontal !\n")
            pass

        try:
            board.digital[SERVO_VERTICAL].write(
                old_vertical+vertical_coef*SERVO_DELTA_Y)

            old_vertical += vertical_coef*SERVO_DELTA_Y

        except ValueError:
            print("Out of bounds : vertical !\n")
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def add_cross_to_coords(coords, frame, color=(0, 0, 255)):
    x, y = coords
    cv2.line(frame, (x, y), (x+X_STEP, y), color, 2)
    cv2.line(frame, (x, y), (x-X_STEP, y), color, 2)
    cv2.line(frame, (x, y), (x, y+Y_STEP), color, 2)
    cv2.line(frame, (x, y), (x, y-Y_STEP), color, 2)
    return frame


@testing
def histogram(L):
    plt.hist(L, bins=200)
    plt.show()


if __name__ == "__main__":
    main()
