import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/npz_files/home_indoor.npz")

range_profile = data['out_x'].reshape(2, 9, 64)
range_profile_label = data['out_y']

range_data_empty = range_profile[0]
range_data_human = range_profile[1]


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


peak_detected_range_data_empty = cell_averaging_peak_detector(range_data_empty, threshold=0.1)
peak_detected_range_data_human = cell_averaging_peak_detector(range_data_human, threshold=0.1)

fig, axs = plt.subplots(2, 1)

# plot empty field
axs[0].set_title("Empty field")
axs[0].imshow(peak_detected_range_data_empty)
axs[0].set_xlabel('Range bins')
axs[0].set_ylabel('Time (s)')

# plot human in front of radar
axs[1].set_title("Human in front of radar")
axs[1].imshow(peak_detected_range_data_human)
axs[1].set_xlabel('Range bins')
axs[1].set_ylabel('Time (s)')

plt.tight_layout()
plt.show()
