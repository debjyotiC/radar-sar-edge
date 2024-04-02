import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/npz_files/home_indoor.npz")

range_profile = data['out_x'].reshape(5, 9, 64)
range_profile_label = data['out_y']


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
    plt.imshow(frame)
    plt.pause(0.5)
