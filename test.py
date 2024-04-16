import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/npz_files/umbc_outdoor.npz")

range_profile = data['out_x'].reshape(49, 9, 256)
range_profile_label = data['out_y'].reshape(49, 9)

frame = range_profile[0]

print(frame.shape)

