import spectral
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def load_hsi_cube(header_path, data_path=None):
    if data_path:
        return spectral.envi.open(header_path, data_path)
    return spectral.envi.open(header_path)

def save_spectral_bands_as_images(hsi_cube, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(hsi_cube.shape[2]):
        band = hsi_cube[:,:,i]
        # Ensures the band data is in a format that PIL can handle
        band = band.astype(np.float32)
        band_normalized = (band - band.min()) / (band.max() - band.min()) * 255
        band_normalized = band_normalized.squeeze()  # Ensure the array is 2D
        img = Image.fromarray(band_normalized.astype(np.uint8), mode='L')
        img.save(os.path.join(output_dir, f'band_{i}.png'))

def show_spectral_bands(hsi_cube, bands):
    plt.figure(figsize=(10, 10))
    for i, band_index in enumerate(bands):
        plt.subplot(2, 2, i + 1)
        plt.imshow(hsi_cube[:,:,band_index], cmap='gray')
        plt.title(f'Spectral Band {band_index}')
    plt.show()

header_path = 'HyperBlood/data/B_1.hdr'
data_path = 'HyperBlood/data/B_1.float'  # Specify the data file path if needed
output_dir = 'output_images'

hsi_cube = load_hsi_cube(header_path, data_path)
save_spectral_bands_as_images(hsi_cube, output_dir)
show_spectral_bands(hsi_cube, [0, 10, 50, 100])  # Displaying 4 spectral bands