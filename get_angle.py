import cv2
import time
import numpy as np
TARGET_COLOR_RGB = [255, 0, 0][::-1]  # we reverse to make it RGB and not BGR
CAMERA_INDEX = 0
FRAMERATE = 60
KERNEL_SIZE = 5
TARGET_MAT = np.array([[TARGET_COLOR_RGB]*KERNEL_SIZE]*KERNEL_SIZE)
MODE_TEST = 0

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
    print(min(L))
    return dico[min(L)]


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        time.sleep(1/FRAMERATE)
        ret, frame = cap.read()
        coords = find_spot(frame)
        # to be removed in the final version
        add_cross_to_coords(coords, frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def add_cross_to_coords(coords, frame):
    x, y = coords
    cv2.line(frame, (x, y), (x+X_STEP, y), (0, 0, 255), 2)
    cv2.line(frame, (x, y), (x-X_STEP, y), (0, 0, 255), 2)
    cv2.line(frame, (x, y), (x, y+Y_STEP), (0, 0, 255), 2)
    cv2.line(frame, (x, y), (x, y-Y_STEP), (0, 0, 255), 2)
    return frame


@testing
def histogram(L):
    import matplotlib.pyplot as plt
    plt.hist(L, bins=200)
    plt.show()


if __name__ == "__main__":
    main()
