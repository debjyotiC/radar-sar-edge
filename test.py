import numpy as np
from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt

dataset_path = 'data/csv_files/umbc'

all_targets = [target for target in listdir(dataset_path) if isdir(join(dataset_path, target))]

data = np.load("data/npz_files/umbc_outdoor.npz")

range_profile = data['out_x'].reshape(13, 9, 256)
range_profile_label = data['out_y'].reshape(13, 9)


def cell_averaging_peak_detector(matrix, threshold=0.5):
    row_means = np.mean(matrix, axis=1)
    max_values = np.max(matrix, axis=1)
    peak_values = (row_means + max_values) / 2
    peak_detected_matrix = np.zeros_like(matrix, dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold and matrix[i, j] >= peak_values[i]:
                peak_detected_matrix[i, j] = 1
    return peak_detected_matrix


for count, frame in enumerate(range_profile):
    plt.clf()
    frame = cell_averaging_peak_detector(frame)
    y = range_profile_label[count][0]-1
    plt.title(all_targets[y])
    plt.imshow(frame)
    plt.xlabel("Range bin")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.pause(4)
