import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# Function to load the ground truth annotations from the .npz file
def load_annotations(annotation_file):
    data = np.load(annotation_file)
    gt = data['gt']
    return gt

# Function to extract pixel coordinates for a given label
def extract_coordinates(gt_array, label):
    """Extract pixel coordinates for a given label."""
    return np.argwhere(gt_array == label)

# Function to allow the user to select two pixels
def pixel_selection(image, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.imshow(image)
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
    plt.show()

    return selected_points

# Function to visualize the annotations and selected points
def visualize_annotations(image, gt_array, selected_points):
    # Extract real and fake blood coordinates
    real_blood_coords = extract_coordinates(gt_array, label=1)
    fake_blood_coords = extract_coordinates(gt_array, label=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Real and Fake Blood Visualization")
    ax.imshow(image)

    # Plot real blood in green
    if real_blood_coords.size > 0:
        ax.scatter(real_blood_coords[:, 1], real_blood_coords[:, 0], c='green', s=10, label='Real Blood')

    # Plot fake blood in blue
    if fake_blood_coords.size > 0:
        ax.scatter(fake_blood_coords[:, 1], fake_blood_coords[:, 0], c='blue', s=10, label='Fake Blood')

    # Plot selected points in red
    for x, y in selected_points:
        ax.plot(x, y, 'ro', label='Selected Point' if 'Selected Point' not in ax.get_legend_handles_labels()[1] else "")

    ax.legend()
    plt.show()

# Example usage
rgb_image_path = "task4_output_img.png"  # Replace with your reconstructed RGB image
annotation_file = "HyperBlood/anno/B_1.npz"  # Replace with your annotation file

# Load the RGB image and ground truth annotations
rgb_image = plt.imread(rgb_image_path)
gt_array = load_annotations(annotation_file)

# Step 1: Select two pixels
selected_pixels = pixel_selection(rgb_image, "Select Two Pixels")

# Step 2: Visualize annotations and selected pixels
visualize_annotations(rgb_image, gt_array, selected_pixels)