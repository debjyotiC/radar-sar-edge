import numpy as np
from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt

dataset_path = 'data/csv_files/umbc'

all_targets = [target for target in listdir(dataset_path) if isdir(join(dataset_path, target))]

data = np.load("data/npz_files/umbc_outdoor.npz")

range_profile = data['out_x'].reshape(49, 9, 256)
range_profile_label = data['out_y'].reshape(49, 9)

configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.04212121212121212,
                    'rangeIdxToMeters': 0.023693181818181818, 'dopplerResolutionMps': 0.12507267556268029,
                    'maxRange': 5.458909090909091, 'maxVelocity': 1.0005814045014423}

rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]


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
    frame = cell_averaging_peak_detector(frame, threshold=70.0)
    y = range_profile_label[count][0] - 1
    plt.title(all_targets[y])
    plt.imshow(frame, extent=[rangeArray[0], rangeArray[-1], 0, 10])
    plt.xlabel("Range (m)")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.pause(1)
