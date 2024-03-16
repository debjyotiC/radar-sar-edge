import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/range-doppler.npz", allow_pickle=True)

rd_data = data['out_x']

configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}  # AWR2944X_Deb
# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])


def calculate_doppler_centroid(config_parameters):
    # Extract relevant parameters
    num_doppler_bins = config_parameters['numDopplerBins']
    doppler_resolution = config_parameters['dopplerResolutionMps']

    # Calculate Doppler centroid (assuming uniform distribution)
    centroid_doppler = (num_doppler_bins // 2) * doppler_resolution

    return centroid_doppler


# Step 1: Preprocessing (Assuming no preprocessing for simplicity)
def doppler_centroid_correction(datacube):
    # Assuming Doppler centroid information is available (replace with your method)
    centroid_doppler = calculate_doppler_centroid(configParameters)  # Replace with actual Doppler centroid value
    rows, cols, _ = datacube.shape
    n = np.arange(cols)[:, np.newaxis]  # Option 1: Expand n with new axis
    # n = n.reshape(-1, 1)  # Option 2: Reshape n directly
    return datacube * np.exp(-2j * np.pi * centroid_doppler * n / cols)


data_cube = doppler_centroid_correction(rd_data)

# Step 2: Range Compression (Matched Filtering)
range_compressed_cube = np.fft.fft(data_cube, axis=1)

# Step 3: Doppler Processing (FFT along Doppler dimension)
doppler_processed_cube = np.fft.fftshift(np.fft.fft(range_compressed_cube, axis=2), axes=2)

# Step 4: Motion Compensation (Assuming no motion compensation for simplicity)

# Step 5: Aperture Synthesis (Beam forming)
# In this simplified example, we'll sum up all the slices coherently
aperture_synthesis_image = np.sum(doppler_processed_cube, axis=0)

# Step 6: Image Formation
# Take absolute value (magnitude) to get the intensity of the image
intensity_image = np.abs(aperture_synthesis_image)

# Step 7: Post-Processing (None in this simplified example)

# Display the final image
plt.contourf(rangeArray, dopplerArray, intensity_image)
plt.title('SAR Image')
plt.xlabel('Range')
plt.ylabel('Doppler')
plt.colorbar(label='Intensity')
plt.show()
