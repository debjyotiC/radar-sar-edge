import numpy as np
import matplotlib.pyplot as plt

file_path = "../data/npz_files/umbc_outdoor.npz"

data = np.load(file_path, allow_pickle=True)

images = data['out_x'].reshape(118, 9, 256)
labels = data['out_y']


def apply_2d_cfar(signal, guard_band_width, kernel_size, threshold_factor):
    num_rows, num_cols = signal.shape
    thresholded_signal = np.zeros((num_rows, num_cols))
    for i in range(guard_band_width, num_rows - guard_band_width):
        for j in range(guard_band_width, num_cols - guard_band_width):
            # Estimate the noise level
            noise_level = np.mean(np.concatenate((
                signal[i - guard_band_width:i + guard_band_width, j - guard_band_width:j + guard_band_width].ravel(),
                signal[i - kernel_size:i + kernel_size, j - kernel_size:j + kernel_size].ravel())))
            # Calculate the threshold for detection
            threshold = threshold_factor * noise_level
            # Check if the signal exceeds the threshold
            if signal[i, j] > threshold:
                thresholded_signal[i, j] = 1
    return thresholded_signal


guard_band_width = 3
kernel_size = 3
threshold_factor = 1.0


for i, frame in enumerate(images):
    plt.clf()
    print(labels[1])
    frame = apply_2d_cfar(frame, guard_band_width, kernel_size, threshold_factor)
    plt.contourf(frame)
    plt.pause(0.1)
