import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/npz_files/home_indoor.npz")

range_profile = data['out_x'].reshape(2, 9, 64)
range_profile_label = data['out_y']

range_data_empty = range_profile[0]
range_data_human = range_profile[1]

fig, axs = plt.subplots(2, 1)

# plot loss
axs[0].imshow(range_data_empty)
axs[0].set_xlabel('Range (m)')
axs[0].set_ylabel('Time (s)')
# plot accuracy
axs[1].imshow(range_data_human)
axs[1].set_xlabel('Range (m)')
axs[1].set_ylabel('Time (s)')
plt.show()
