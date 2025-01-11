import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image
import json

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
    ax.set_title(f"{title}\nLeft click = Real Blood, Right click = Fake Blood, Middle click = Background\nPress 'Enter' to proceed")
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
    background_pixels = []

    # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)

            # Left-click for 'real blood', Right-click for 'fake blood', Middle-click for 'background'
            if event.button == 1:  # Left-click
                real_pixels.append((x, y))
                ax.plot(x, y, 'go')  # Mark with green for 'real'
            elif event.button == 3:  # Right-click
                fake_pixels.append((x, y))
                ax.plot(x, y, 'bo')  # Mark with blue for 'fake'
            elif event.button == 2:  # Middle-click
                background_pixels.append((x, y))
                ax.plot(x, y, 'yo')  # Mark with yellow for 'background'

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
        Line2D([0], [0], marker='o', color='blue', lw=0, markersize=10, label='Fake Blood (User)'),
        Line2D([0], [0], marker='o', color='yellow', lw=0, markersize=10, label='Background (User)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return real_pixels, fake_pixels, background_pixels

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

def save_labeled_pixels(anno_file1, real_pixels1, fake_pixels1, background_pixels1, anno_file2, real_pixels2, fake_pixels2, background_pixels2, output_json):
    labeled_pixels = {
        anno_file1: {
            'real_pixels': real_pixels1,
            'fake_pixels': fake_pixels1,
            'background_pixels': background_pixels1
        },
        anno_file2: {
            'real_pixels': real_pixels2,
            'fake_pixels': fake_pixels2,
            'background_pixels': background_pixels2
        }
    }

    with open(output_json, 'w') as f:
        json.dump(labeled_pixels, f, indent=4)

    print(f"Labeled pixels saved to {output_json} with separate entries for {anno_file1} and {anno_file2}.")

def main(image, gt_array, image2, gt_array2, annofile1, annofile2):

    # Interactive pixel labeling
    real_pixels, fake_pixels, background_pixels = pixel_labeling(image, gt_array, "Label Real and Fake Blood Pixels")

    # Ensure sufficient labeled pixels
    if len(real_pixels) < 30 or len(fake_pixels) < 30:
        print("Please label at least 30 real blood pixels and 30 fake blood pixels.")
    else:
        print(f"Labeled {len(real_pixels)} real blood pixels and {len(fake_pixels)} fake blood pixels.")

    # Plot collections of spectra
    plot_collections(image, real_pixels, fake_pixels)

    # Interactive pixel labeling
    real_pixels2, fake_pixels2, background_pixels2 = pixel_labeling(image2, gt_array2, "Label Real and Fake Blood Pixels")

    # Ensure sufficient labeled pixels
    if len(real_pixels2) < 30 or len(fake_pixels2) < 30:
        print("Please label at least 30 real blood pixels and 30 fake blood pixels.")
    else:
        print(f"Labeled {len(real_pixels2)} real blood pixels and {len(fake_pixels2)} fake blood pixels.")

    # Plot collections of spectra
    plot_collections(image2, real_pixels2, fake_pixels2)

    # Save labeled pixels to a JSON file
    save_labeled_pixels(annofile1, real_pixels, fake_pixels, background_pixels, annofile2, real_pixels2, fake_pixels2, background_pixels2, 'labeled_pixels.json')

    return real_pixels, fake_pixels, background_pixels, real_pixels2, fake_pixels2, background_pixels2

if __name__ == "__main__":
    # Define paths to the new scene image and annotations
    image_path = "task11_training.png" 
    image_path2 = "task4_output_img.png"  
    annotation_file = "HyperBlood/anno/E_1.npz" 
    annotation_file2 = r"HyperBlood\anno\B_1.npz"  

    # Load the image and annotations
    image = np.array(Image.open(image_path))
    gt_array = load_annotations(annotation_file)

    # Load the image and annotations
    image2 = np.array(Image.open(image_path2))
    gt_array2 = load_annotations(annotation_file2)
    main(image, gt_array, image2, gt_array2, annotation_file, annotation_file2)