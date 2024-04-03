import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/npz_files/home_indoor.npz")

range_profile = data['out_x'].reshape(5, 9, 64)
range_profile_label = data['out_y']

range_data_empty = range_profile[0]
range_data_human_1 = range_profile[1]
range_data_human_2 = range_profile[2]
range_data_human_3 = range_profile[3]
range_data_human_4 = range_profile[4]


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


def find_clusters_and_centroids(matrix):
    clusters_indices = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 1:
                clusters_indices.append((i, j))

    centroids = []
    for cluster_indices in clusters_indices:
        centroids.append(np.array(cluster_indices))

    return centroids


peak_detected_range_data_empty = cell_averaging_peak_detector(range_data_empty, threshold=0.1)
peak_detected_range_data_human_1 = cell_averaging_peak_detector(range_data_human_1, threshold=0.1)
peak_detected_range_data_human_2 = cell_averaging_peak_detector(range_data_human_2, threshold=0.1)
peak_detected_range_data_human_3 = cell_averaging_peak_detector(range_data_human_3, threshold=0.1)
peak_detected_range_data_human_4 = cell_averaging_peak_detector(range_data_human_4, threshold=0.1)

empty_centroids = find_clusters_and_centroids(peak_detected_range_data_empty)
human_centroids_1 = find_clusters_and_centroids(peak_detected_range_data_human_1)
human_centroids_2 = find_clusters_and_centroids(peak_detected_range_data_human_2)
human_centroids_3 = find_clusters_and_centroids(peak_detected_range_data_human_3)
human_centroids_4 = find_clusters_and_centroids(peak_detected_range_data_human_4)

fig, axs = plt.subplots(5, 1)

# plot empty field
axs[0].set_title("Empty field")
axs[0].imshow(peak_detected_range_data_empty)
axs[0].set_xlabel('Range bins')
axs[0].set_ylabel('Time (s)')

for centroid in empty_centroids:
    axs[0].scatter(centroid[1], centroid[0], color='red', marker='x')

# plot human in front of radar
axs[1].set_title("Human in front of radar")
axs[1].imshow(peak_detected_range_data_human_1)
axs[1].set_xlabel('Range bins')
axs[1].set_ylabel('Time (s)')
for centroid in human_centroids_1:
    axs[1].scatter(centroid[1], centroid[0], color='red', marker='x')


axs[2].set_title("Human in front of radar")
axs[2].imshow(peak_detected_range_data_human_2)
axs[2].set_xlabel('Range bins')
axs[2].set_ylabel('Time (s)')
for centroid in human_centroids_2:
    axs[2].scatter(centroid[1], centroid[0], color='red', marker='x')


axs[3].set_title("Human in front of radar")
axs[3].imshow(peak_detected_range_data_human_3)
axs[3].set_xlabel('Range bins')
axs[3].set_ylabel('Time (s)')
for centroid in human_centroids_3:
    axs[3].scatter(centroid[1], centroid[0], color='red', marker='x')


axs[4].set_title("Human in front of radar")
axs[4].imshow(peak_detected_range_data_human_4)
axs[4].set_xlabel('Range bins')
axs[4].set_ylabel('Time (s)')

for centroid in human_centroids_4:
    axs[4].scatter(centroid[1], centroid[0], color='red', marker='x')

# plt.tight_layout()
plt.show()
