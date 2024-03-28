import pandas as pd
import numpy as np
from os import listdir
from os.path import isdir, join

dataset_path = 'data/csv_files'

all_targets = [target for target in listdir(dataset_path) if isdir(join(dataset_path, target))]

print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


def calc_range_profile(data_frame, packet_id):
    payload = data_frame[packet_id].to_numpy()
    # Convert levels to dBm
    # payload = 20 * np.log10(payload)
    return payload


out_x_range_profile = []
out_y_range_profile = []

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])

        print(full_path, folder)

        df_data = pd.read_csv(full_path)

        for col in df_data.columns:
            data = calc_range_profile(df_data, col)

            out_x_range_profile.append(data)
            out_y_range_profile.append(folder + 1)

data_range_x = np.array(out_x_range_profile)
data_range_y = np.array(out_y_range_profile)

np.savez('data/npz_files/home_indoor.npz', out_x=data_range_x, out_y=data_range_y)
