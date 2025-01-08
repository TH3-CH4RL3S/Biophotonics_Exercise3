import numpy as np
from PIL import Image
import os

def save_spectral_bands(image_path, save_dir):
    """
    Breaks an RGB image into its spectral bands and saves each band as a PNG file.

    Parameters:
        image_path (str): Path to the RGB image file.
        save_dir (str): Directory to save the spectral band PNG files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the RGB image
    rgb_image = np.array(Image.open(image_path)) / 255.0  # Normalize to [0, 1]

    # Split into R, G, and B channels
    bands = [rgb_image[:, :, i] for i in range(3)]

    # Save each band as a separate PNG file
    for i, band in enumerate(bands):
        band_image = (band * 255).astype(np.uint8)
        save_path = os.path.join(save_dir, f"band_{i}.png")
        Image.fromarray(band_image).save(save_path)

def restore_rgb_from_bands(bands_dir, save_path):
    """
    Restores an RGB image from spectral band PNG files.

    Parameters:
        bands_dir (str): Directory containing the spectral band PNG files.
        save_path (str): Path to save the restored RGB image.
    """
    # Load the spectral bands
    band_files = sorted([os.path.join(bands_dir, f) for f in os.listdir(bands_dir) if f.endswith(".png")])
    bands = [np.array(Image.open(band)) / 255.0 for band in band_files]  # Normalize to [0, 1]

    # Stack the bands into an RGB image
    rgb_image = np.stack(bands, axis=-1)

    # Save the RGB image
    Image.fromarray((rgb_image * 255).astype(np.uint8)).save(save_path)

# Break the RGB image into spectral bands
rgb_image_path = "HyperBlood/images/mock_up_scene.png"
bands_save_dir = "task5_output_bands"
save_spectral_bands(rgb_image_path, bands_save_dir)

# Restore the RGB image from the spectral bands
restored_rgb_path = "task5_restored_images/third.png"
restore_rgb_from_bands(bands_save_dir, restored_rgb_path)