import os
import numpy as np
from PIL import Image
import spectral
import matplotlib.pyplot as plt

def hyperspectral_to_rgb(hsi_data, wavelengths, save_path=None):
    """
    Converts hyperspectral data to an RGB image using a simplified algorithm.

    Parameters:
        hsi_data (numpy.ndarray): Hyperspectral data (H, W, B) where B is the number of bands.
        wavelengths (list): List of wavelengths corresponding to the bands.
        save_path (str, optional): Path to save the resulting RGB image. Default is None.

    Returns:
        numpy.ndarray: RGB image as a NumPy array.
    """
    # Define visible spectrum ranges for Red, Green, and Blue
    red_range = (620, 750)
    green_range = (495, 570)
    blue_range = (450, 495)

    # Initialize RGB channels
    red_channel = np.zeros(hsi_data.shape[:2])
    green_channel = np.zeros(hsi_data.shape[:2])
    blue_channel = np.zeros(hsi_data.shape[:2])

    # Assign intensity to each channel based on spectral bands
    for i, wavelength in enumerate(wavelengths):
        if red_range[0] <= wavelength <= red_range[1]:
            red_channel += hsi_data[:, :, i]
        elif green_range[0] <= wavelength <= green_range[1]:
            green_channel += hsi_data[:, :, i]
        elif blue_range[0] <= wavelength <= blue_range[1]:
            blue_channel += hsi_data[:, :, i]

    # Normalize RGB channels
    rgb = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    rgb = rgb / np.max(rgb)  # Normalize to range [0, 1]

    # Optionally save the image
    if save_path:
        Image.fromarray((rgb * 255).astype(np.uint8)).save(save_path)

    return rgb

output_dir = 'output_images2'

# Load the uploaded band images
band_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]

# Read the images and stack them into a 3D array
bands = [np.array(Image.open(band)) for band in band_files]
hsi_data = np.stack(bands, axis=-1)

# Wavelengths for the bands (replace these with actual wavelengths if known)
example_wavelengths = [450, 500, 550, 600, 650]

# Convert hyperspectral data to RGB
rgb_image = hyperspectral_to_rgb(hsi_data, example_wavelengths)

# Save and display the RGB image
output_path = "task9_newscene.png"
Image.fromarray((rgb_image * 255).astype(np.uint8)).save(output_path)