import numpy as np
import matplotlib.pyplot as plt

# Radar parameters
num_doppler_bins = 16
num_range_bins = 128
range_resolution_meters = 0.04360212053571429
range_idx_to_meters = 0.04360212053571429
doppler_resolution_mps = 0.12518841691334906
max_range = 10.045928571428572
max_velocity = 2.003014670613585


# Generate synthetic data (range profiles)
def generate_range_profile(num_range_bins, target_distance, target_rcs):
    range_axis = np.linspace(0, num_range_bins - 1, num_range_bins) * range_resolution_meters
    target_echo = np.zeros(num_range_bins)
    target_index = int(target_distance / range_idx_to_meters)
    if target_index < num_range_bins:
        target_echo[target_index] = target_rcs
    return target_echo


target_distance = 5  # Ensure the target distance falls within the range bins
target_rcs = 10  # Radar cross-section of target (arbitrary units)

range_profiles = np.zeros((num_doppler_bins, num_range_bins))
for i in range(num_doppler_bins):
    range_profiles[i, :] = generate_range_profile(num_range_bins, target_distance, target_rcs)


# Range compression (matched filtering)
def matched_filter(signal):
    pulse = np.hamming(len(signal))
    return np.convolve(signal, pulse, mode='same')


range_profiles_compressed = np.apply_along_axis(matched_filter, axis=1, arr=range_profiles)


# Doppler processing
def doppler_processing(signal):
    doppler_axis = np.linspace(-max_velocity, max_velocity, num_doppler_bins)
    return np.fft.fftshift(np.fft.fft(signal, axis=0), axes=0), doppler_axis


range_profiles_doppler, doppler_axis = doppler_processing(range_profiles_compressed)

# Plot the resulting SAR image
plt.imshow(np.abs(range_profiles_doppler), cmap='gray', extent=[0, max_range, -max_velocity, max_velocity],
           aspect='auto')
plt.xlabel('Range (m)')
plt.ylabel('Doppler Velocity (m/s)')
plt.title('SAR Image')
plt.colorbar(label='Amplitude')
plt.show()
