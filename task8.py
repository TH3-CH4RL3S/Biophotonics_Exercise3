import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to show monochrome images of chosen spectral bands
def show_monochrome_images(image, band1, band2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title(f"Spectral Band {band1}")
    axes[0].imshow(image[:, :, band1], cmap='gray')

    axes[1].set_title(f"Spectral Band {band2}")
    axes[1].imshow(image[:, :, band2], cmap='gray')

    plt.show()

# Load the image
image_path = "task4output_B.png"  # Path to the PNG file
image = np.array(Image.open(image_path))

# Verify the number of bands
print(f"Image shape: {image.shape}")

# Step 3: Show monochrome images of chosen spectral bands
# Choose bands based on analysis of spectra (e.g., bands 0 and 2 for demonstration)
band1, band2 = 0, min(2, image.shape[-1] - 1)  # Adjust based on the number of available bands
show_monochrome_images(image, band1, band2)
