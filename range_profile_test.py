import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/range-profile.npz", allow_pickle=True)

radar_data = data['out_x']

# sar_image = np.fft.fftshift(np.fft.fft2(range_profiles))
#
# # Display the SAR image
# plt.imshow(np.abs(sar_image), extent=[-1, 1, -1, 1])  # Adjust extent based on scene dimensions
# plt.xlabel('Azimuth')
# plt.ylabel('Range')
# plt.title('SAR Image')
# plt.colorbar(label='Intensity')
# plt.show()

c = 3e8  # Speed of light
fc = 5e9  # Center frequency of the radar system

# Define scene parameters
azimuth_resolution = 0.1  # Azimuth resolution in meters
range_resolution = 0.1  # Range resolution in meters

# Calculate parameters
num_ants, num_range_bins = radar_data.shape
max_range = num_range_bins * range_resolution
max_azimuth = num_ants * azimuth_resolution

# Calculate wave number
lambda_ = c / fc
k0 = 2 * np.pi / lambda_

# Initialize focused SAR image
sar_image = np.zeros((num_ants, num_range_bins), dtype=complex)

# Perform range migration
for i in range(num_ants):
    for j in range(num_range_bins):
        r = j * range_resolution
        phi = i * azimuth_resolution

        # Calculate phase correction
        phase_corr = np.exp(1j * k0 * r * np.sin(phi))

        # Apply phase correction
        sar_image[i, j] = radar_data[i, j] * phase_corr

# Inverse Fourier Transform along the range dimension
sar_image = np.fft.ifftshift(sar_image, axes=1)
sar_image = np.fft.ifft(sar_image, axis=1)

# Display the SAR image
plt.imshow(np.abs(sar_image), cmap='gray', extent=[0, max_range, 0, max_azimuth])
plt.xlabel('Range (m)')
plt.ylabel('Azimuth (m)')
plt.title('Focused SAR Image (Range Migration Algorithm)')
plt.colorbar(label='Intensity')
plt.show()
