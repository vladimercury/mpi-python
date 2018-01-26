import numpy as np


def read_data(file_name):
    with open(file_name, "r") as file:
        size = int(file.readline())
        data = np.zeros(size)
        for i in range(size):
            data[i] = float(file.readline())
        return data