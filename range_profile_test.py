import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/range-profile-2.npz", allow_pickle=True)

radar_data = data['out_x'].T

# Display the SAR image
plt.imshow(radar_data, extent=[-1, 1, -1, 1])  # Adjust extent based on scene dimensions
plt.show()
