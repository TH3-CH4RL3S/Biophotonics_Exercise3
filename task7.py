import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image

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
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.imshow(image)

    # Create masks for real and fake blood
    real_blood_mask = (gt_array == 1)
    fake_blood_mask = (gt_array == 2)

    # Plot real blood regions in green
    if real_blood_mask.any():
        ax.contour(real_blood_mask, levels=[0.5], colors='green', linewidths=2, label='Real Blood')

    # Plot fake blood regions in blue
    if fake_blood_mask.any():
        ax.contour(fake_blood_mask, levels=[0.5], colors='blue', linewidths=2, label='Fake Blood')

    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    selected_points = []

    # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            selected_points.append((x, y))
            ax.plot(x, y, 'ro')  # Mark the selected point
            fig.canvas.draw()

            if len(selected_points) == 2:  # Stop after 2 points
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)

    # Add a legend
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
def visualize_spectra(image, selected_points):
    """
    Visualizes the reflectance spectra of the selected pixels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Intensity Spectra of Selected Pixels")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Intensity")

    for i, (x, y) in enumerate(selected_points):
        spectra = image[y, x, :]  # Extract all bands for the pixel
        ax.plot(range(len(spectra)), spectra, marker='o', label=f'Pixel {i+1} ({x}, {y})')

    ax.legend()
    plt.show()

# Define paths to the image and annotations
image_path = "task4output_C.png"  # Path to the PNG file
annotation_file = r"HyperBlood\anno\C_1.npz"  # Path to annotations file

# Load the image and annotations
image = np.array(Image.open(image_path))
gt_array = load_annotations(annotation_file)

# Verify the number of bands
print(f"Image shape: {image.shape}")

# Step 1: Select two pixels with annotations visualized
selected_pixels = pixel_selection(image, "Select Two Pixels with Annotations", gt_array)

# Step 2: Visualize the reflectance spectra of the selected pixels
visualize_spectra(image, selected_pixels)
