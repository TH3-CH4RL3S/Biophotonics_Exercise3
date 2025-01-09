
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from spectral import envi

# Function to load the ground truth annotations from the .npz file
def load_annotations(annotation_file):
    data = np.load(annotation_file)
    gt = data['gt']
    return gt

# Function to simulate reflectance data (for demo purposes)
def simulate_reflectance(gt_array):
    height, width = gt_array.shape
    wavelengths = 100  # Simulating 100 wavelengths
    reflectance_array = np.zeros((height, width, wavelengths))

    # Assign random spectra for "real blood" and "fake blood" regions
    for y in range(height):
        for x in range(width):
            if gt_array[y, x] == 1:  # Real blood
                reflectance_array[y, x, :] = np.linspace(0.4, 0.8, wavelengths) + np.random.normal(0, 0.02, wavelengths)
            elif gt_array[y, x] == 2:  # Fake blood
                reflectance_array[y, x, :] = np.linspace(0.6, 0.4, wavelengths) + np.random.normal(0, 0.02, wavelengths)
            else:  # Background or other regions
                reflectance_array[y, x, :] = np.random.normal(0.5, 0.1, wavelengths)

    return reflectance_array

# Unified visualization and interaction function
def interactive_visualization(image, gt_array, reflectance_array):
    fig, (ax_image, ax_spectra) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Interactive Visualization with Reflectance Spectra", fontsize=16)

    # Image and contours
    ax_image.set_title("Image with Contours")
    ax_image.imshow(image)

    # Create masks for real and fake blood
    real_blood_mask = (gt_array == 1)
    fake_blood_mask = (gt_array == 2)

    # Plot real blood regions in green using contours
    if real_blood_mask.any():
        ax_image.contour(real_blood_mask, levels=[0.5], colors='green', linewidths=2, label='Real Blood')

    # Plot fake blood regions in blue using contours
    if fake_blood_mask.any():
        ax_image.contour(fake_blood_mask, levels=[0.5], colors='blue', linewidths=2, label='Fake Blood')

    # Enable cursor and mouse click event handling
    cursor = Cursor(ax_image, useblit=True, color='red', linewidth=2)
    selected_points = []

    # Reflectance spectra plot
    ax_spectra.set_title("Reflectance Spectra")
    ax_spectra.set_xlabel("Wavelength Index")
    ax_spectra.set_ylabel("Reflectance")
    ax_spectra.grid(True)

    # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax_image:
            x, y = int(event.xdata), int(event.ydata)
            selected_points.append((x, y))

            # Plot selected point on the image
            ax_image.plot(x, y, 'ro')
            fig.canvas.draw()

            # Update reflectance spectra
            reflectance = reflectance_array[y, x, :]
            ax_spectra.plot(reflectance, label=f"Point ({x}, {y})")
            ax_spectra.legend()
            fig.canvas.draw()

            # Print selected point for debugging or further use
            print(f"Selected point: ({x}, {y})")

    fig.canvas.mpl_connect('button_press_event', onclick)

    # Add custom legend for image
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Real Blood'),
        Line2D([0], [0], color='blue', lw=2, label='Fake Blood'),
        Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label='Selected Point')
    ]
    ax_image.legend(handles=legend_elements, loc='upper right')

    plt.show()

    return selected_points

# Define the paths to the RGB image and ground truth annotations
rgb_image_path = "task4_output_img.png"  
annotation_file = "HyperBlood/anno/B_1.npz"  

# Load the RGB image and ground truth annotations
rgb_image = plt.imread(rgb_image_path)
gt_array = load_annotations(annotation_file)

# Simulate reflectance data
reflectance_array = simulate_reflectance(gt_array)

# Show the unified interactive UI
selected_pixels = interactive_visualization(rgb_image, gt_array, reflectance_array)

# Output selected points after UI interaction
print(f"Final selected points: {selected_pixels}")