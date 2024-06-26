import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft
from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt

data = np.load("data/npz_files/umbc_outdoor.npz")

range_profile = data['out_x'].reshape(118, 9, 256)
range_profile_label = data['out_y'].reshape(118, 9)

configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.04212121212121212,
                    'rangeIdxToMeters': 0.023693181818181818, 'dopplerResolutionMps': 0.12507267556268029,
                    'maxRange': 5.458909090909091, 'maxVelocity': 1.0005814045014423}

rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]

num_time_steps = 9
num_range_bins = 256
radar_data = np.random.rand(num_time_steps, num_range_bins)


def find_clusters_and_centroids(matrix):
    clusters_indices = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 1:
                clusters_indices.append((i, j))

    centroids = []
    for cluster_indices in clusters_indices:
        centroid = np.array(cluster_indices)
        centroids.append(centroid)

    return centroids


# Apply temporal filtering using a simple moving average filter
def moving_average_filter(radarData, window_size):
    filtered_data = np.zeros_like(radarData)
    for i in range(radarData.shape[0]):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(radarData.shape[0], i + window_size // 2 + 1)
        filtered_data[i] = np.mean(radarData[start_idx:end_idx], axis=0)
    return filtered_data


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


ground_mask = np.ones((9, 256))
ground_mask[:, :10] = 0

processed_range_profile_data = []
processed_range_profile_label = []

for count, frame in enumerate(range_profile):
    plt.clf()
    frame = cell_averaging_peak_detector(frame, threshold=70.1)
    centroids = find_clusters_and_centroids(frame)
    y = range_profile_label[count][0] - 1
    frame = frame * ground_mask

    overall_sum = np.sum(frame)

    if overall_sum > 9.0:
        occupancy_type = "object detected"
        detected = True
    else:
        occupancy_type = "no object detected"
        detected = False

    obj_dict = {"Obj_Detected": detected, "Obj_Class": None, "Obj_Distance": None}
    print(obj_dict)
    processed_range_profile_data.append(frame)
    processed_range_profile_label.append(range_profile_label[count][0])
    plt.title(f"{occupancy_type}")
    plt.imshow(frame, extent=[rangeArray[0], rangeArray[-1], 0, 10])
    plt.xlabel("Range (m)")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.pause(1)
