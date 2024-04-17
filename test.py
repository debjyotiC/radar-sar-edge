import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

data = np.load("data/npz_files/home_indoor.npz")

range_profile = data['out_x'].reshape(5, 9, 64)
# range_profile_label = data['out_y'].reshape(49, 9)

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


for frame in range_profile:
    plt.clf()
    # frame = normalize(frame, axis=1, norm='l1')

    min_value = np.min(frame)
    max_value = np.max(frame)

    # Normalize the matrix
    frame = (frame - min_value) / (max_value - min_value)

    frame = cell_averaging_peak_detector(frame, threshold=0)
    plt.imshow(frame)
    plt.pause(2)

# Parameters for STFT
# window_size = 8
# overlap = 2  # Overlap size, can be adjusted
# fs = 2  # Sampling frequency, assuming one unit per sample
#
# # Compute STFT
# f, t, Zxx = stft(normalized_arr, fs=fs, window='hann', nperseg=window_size, noverlap=overlap)
#
# # Plot STFT
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
# # plt.colorbar(label='Magnitude')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency')
# plt.xlabel('Time')
# plt.show()
