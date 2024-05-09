import pandas as pd
import numpy as np
from os import listdir
from os.path import isdir, join

dataset_path = 'data/csv_files/umbc'

all_targets = sorted([name for name in listdir(dataset_path) if isdir(join(dataset_path, name))], reverse=True)

print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


def calc_range_profile(file_name):
    stacked_range_array = []

    df = pd.read_csv(file_name)

    for col in df.columns:
        stacked_range_array.append(df[col])

    return stacked_range_array


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


out_x_range_profile = []
out_y_range_profile = []

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])

        stacked_range_profile = np.array(calc_range_profile(full_path))

        out_x_range_profile.append(cell_averaging_peak_detector(stacked_range_profile, threshold=0.1))
        out_y_range_profile.append(folder + 1)

data_range_x = np.array(out_x_range_profile)
data_range_y = np.array(out_y_range_profile)

np.savez('data/npz_files/umbc_outdoor.npz', out_x=data_range_x, out_y=data_range_y)
