import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image

# Function to load the ground truth annotations from a .npz file
def load_annotations(annotation_file):
    """
    Load the ground truth annotations from a .npz file.
    """
    data = np.load(annotation_file)
    gt = data['gt']
    return gt

# Function to allow the user to select two pixels
def pixel_selection(image, title, gt_array):
    """
    Displays an image with annotations and allows the user to select two pixels.
    Identifies whether the selected pixels are real or fake blood.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.imshow(image)

    # Create masks for real and fake blood
    real_blood_mask = (gt_array == 1)
    fake_blood_mask = (gt_array == 2)

    # Plot real blood regions in green
    if real_blood_mask.any():
        real_contour = ax.contour(real_blood_mask, levels=[0.5], colors='green', linewidths=2)

    # Plot fake blood regions in blue
    if fake_blood_mask.any():
        fake_contour = ax.contour(fake_blood_mask, levels=[0.5], colors='blue', linewidths=2)
    selected_points = []

    # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            label = "Unknown"
            if real_blood_mask[y, x]:
                label = "Real Blood"
            elif fake_blood_mask[y, x]:
                label = "Fake Blood"
            selected_points.append((x, y, label))
            ax.plot(x, y, 'ro', label=label)  # Mark the selected point
            fig.canvas.draw()

            if len(selected_points) == 2:  # Stop after 2 points
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)

    # Add a legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Real Blood'),
        Line2D([0], [0], color='blue', lw=2, label='Fake Blood'),
        Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label='Selected Point')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return selected_points

# Function to visualize the reflectance spectra of selected pixels
def visualize_reflectance(image, selected_points):
    """
    Visualizes the reflectance spectra of the selected pixels.
    Labels the spectra based on whether the pixel is real or fake blood.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Reflectance Spectra of Selected Pixels")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Reflectance")

    max_intensity = np.max(image, axis=(0, 1))  # Compute max intensity for normalization across all pixels

    for i, (x, y, label) in enumerate(selected_points):
        spectra = image[y, x, :]  # Extract all bands for the pixel
        reflectance = spectra / max_intensity  # Normalize to obtain reflectance
        ax.plot(range(len(reflectance)), reflectance, marker='o', label=f'Pixel {i+1} ({label})')

    ax.legend()
    plt.show()

# Image and annotations files
image_path = "task4output_B.png" 
annotation_file = r"HyperBlood\anno\B_1.npz"

# Load the image and annotations
image = np.array(Image.open(image_path))
gt_array = load_annotations(annotation_file)

# Verify the number of bands
print(f"Image shape: {image.shape}")

# Step 1: Select two pixels with annotations visualized
selected_pixels = pixel_selection(image, "Select Two Pixels", gt_array)

# Step 2: Visualize the reflectance spectra of the selected pixels
visualize_reflectance(image, selected_pixels)