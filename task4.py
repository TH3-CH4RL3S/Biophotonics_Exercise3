from spectral import envi
import matplotlib.pyplot as plt
import numpy as np

# Load the ENVI hyperspectral data
hdr_file = r'HyperBlood\data\B_1.hdr'  
data_file = r'HyperBlood\data\B_1.float'
hsi_cube = envi.open(hdr_file, data_file)
hsi_data = hsi_cube.load()

# Dataset wavelength information (in nanometers)
wavelengths = np.array(hsi_cube.metadata.get('wavelength', []), dtype=float)

# Define target RGB wavelength ranges

# These ranges are approximate and can be refined for specific datasets
red_range = (620, 750)    # nm
green_range = (495, 570)  # nm
blue_range = (450, 495)   # nm

# FUNCTION TO GET BAND INDICES
def get_band_indices(wavelengths, target_range):
    return [i for i, w in enumerate(wavelengths) if target_range[0] <= w <= target_range[1]]

# Get band indices for RGB
red_bands = get_band_indices(wavelengths, red_range)
green_bands = get_band_indices(wavelengths, green_range)
blue_bands = get_band_indices(wavelengths, blue_range)

# Average the spectral bands in each range to create R, G, B channels
def average_bands(hsi_data, band_indices):
    if not band_indices:
        raise ValueError("No bands found for the given wavelength range.")
    return np.mean(hsi_data[:, :, band_indices], axis=-1)

red_channel = average_bands(hsi_data, red_bands)
green_channel = average_bands(hsi_data, green_bands)
blue_channel = average_bands(hsi_data, blue_bands)

# Stack the RGB channels
rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

# Display the RGB image
plt.figure(figsize=(10, 10))
plt.title("Reconstructed RGB Image")
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

# Save the RGB image
output_path = "task4output_B.png"
plt.imsave(output_path, rgb_image)
print(f"RGB image saved to {output_path}")