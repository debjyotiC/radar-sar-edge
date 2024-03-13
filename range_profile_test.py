import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/range-profile.npz", allow_pickle=True)

range_profiles = data['out_x']

range_compressed_data = range_profiles[:64, :].T

sar_image = np.fft.fft2(range_compressed_data)

plt.imshow(range_compressed_data)
plt.show()
