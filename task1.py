from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Read the RGB image using PIL
image = Image.open('HyperBlood/images/frame_image.png')
image = np.array(image)

# Split the image into its red, green, and blue channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Visualize the images
plt.figure(figsize=(10, 8))

# Original color image
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Color Image')
plt.axis('off')

# Red channel in grayscale
plt.subplot(2, 2, 2)
plt.imshow(red_channel, cmap='gray')
plt.title('Red Channel in Grayscale')
plt.axis('off')

# Green channel in grayscale
plt.subplot(2, 2, 3)
plt.imshow(green_channel, cmap='gray')
plt.title('Green Channel in Grayscale')
plt.axis('off')

# Blue channel in grayscale
plt.subplot(2, 2, 4)
plt.imshow(blue_channel, cmap='gray')
plt.title('Blue Channel in Grayscale')
plt.axis('off')

plt.tight_layout()
plt.show()