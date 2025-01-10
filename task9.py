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

def pixel_labeling(image, gt_array, title):
    """
    Interactive UI to label pixels as 'real' or 'fake' blood with annotations overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{title}\nLeft click = Real Blood, Right click = Fake Blood\nPress 'Enter' to proceed")
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

    real_pixels = []
    fake_pixels = []

    # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)

            # Left-click for 'real blood', Right-click for 'fake blood'
            if event.button == 1:  # Left-click
                real_pixels.append((x, y))
                ax.plot(x, y, 'go')  # Mark with green for 'real'
            elif event.button == 3:  # Right-click
                fake_pixels.append((x, y))
                ax.plot(x, y, 'bo')  # Mark with blue for 'fake'

            fig.canvas.draw()

    def onkey(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    # Legend for labeling and annotations
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Real Blood (Annotation)'),
        Line2D([0], [0], color='blue', lw=2, label='Fake Blood (Annotation)'),
        Line2D([0], [0], marker='o', color='green', lw=0, markersize=10, label='Real Blood (User)'),
        Line2D([0], [0], marker='o', color='blue', lw=0, markersize=10, label='Fake Blood (User)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return real_pixels, fake_pixels

def plot_collections(image, real_pixels, fake_pixels):
    """
    Plot collections of reflectance spectra for real and fake blood pixels.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Reflectance Spectra for Real and Fake Blood")

    # Plot real blood spectra
    ax[0].set_title("Real Blood Spectra")
    ax[0].set_xlabel("Spectral Band")
    ax[0].set_ylabel("Reflectance")
    for x, y in real_pixels:
        spectra = image[y, x, :]
        ax[0].plot(range(spectra.shape[0]), spectra, label=f"({x}, {y})")
    ax[0].legend(loc='upper right', fontsize=8)

    # Plot fake blood spectra
    ax[1].set_title("Fake Blood Spectra")
    ax[1].set_xlabel("Spectral Band")
    for x, y in fake_pixels:
        spectra = image[y, x, :]
        ax[1].plot(range(spectra.shape[0]), spectra, label=f"({x}, {y})")
    ax[1].legend(loc='upper right', fontsize=8)

    def onkey(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

# Define paths to the new scene image and annotations
image_path = "task4_output_img.png"  # Replace with the path to the new scene image
image_path2 = "task9_newscene.png"  # Replace with the path to the new scene image
annotation_file = "HyperBlood/anno/B_1.npz"  # Replace with the new scene's annotation file
annotation_file2 = r"HyperBlood\anno\D_1.npz"  # Replace with the new scene's annotation file

# Load the image and annotations
image = np.array(Image.open(image_path))
gt_array = load_annotations(annotation_file)

# Interactive pixel labeling
real_pixels, fake_pixels = pixel_labeling(image, gt_array, "Label Real and Fake Blood Pixels")

# Ensure sufficient labeled pixels
if len(real_pixels) < 30 or len(fake_pixels) < 30:
    print("Please label at least 30 real blood pixels and 30 fake blood pixels.")
else:
    print(f"Labeled {len(real_pixels)} real blood pixels and {len(fake_pixels)} fake blood pixels.")

# Plot collections of spectra
plot_collections(image, real_pixels, fake_pixels)

# Load the image and annotations
image2 = np.array(Image.open(image_path2))
gt_array2 = load_annotations(annotation_file2)

# Interactive pixel labeling
real_pixels2, fake_pixels2 = pixel_labeling(image2, gt_array2, "Label Real and Fake Blood Pixels")

# Ensure sufficient labeled pixels
if len(real_pixels2) < 30 or len(fake_pixels2) < 30:
    print("Please label at least 30 real blood pixels and 30 fake blood pixels.")
else:
    print(f"Labeled {len(real_pixels2)} real blood pixels and {len(fake_pixels2)} fake blood pixels.")

# Plot collections of spectra
plot_collections(image2, real_pixels2, fake_pixels2)