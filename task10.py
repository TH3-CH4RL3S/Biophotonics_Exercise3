import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from task9 import main as task9

def load_annotations(annotation_file):
    """
    Load the ground truth annotations from a .npz file.
    """
    data = np.load(annotation_file)
    gt = data['gt']
    return gt

def calculate_average_and_std(image, pixel_list):
    """
    Calculate the average spectral line and standard deviation for a collection of pixels.
    """
    spectra = np.array([image[y, x, :] for x, y in pixel_list])
    average_spectrum = np.mean(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)
    return average_spectrum, std_spectrum

def plot_average_spectra(image1, real_pixels1, fake_pixels1, image2, real_pixels2, fake_pixels2):
    """
    Plot average spectral lines with standard deviation for real and fake blood.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Average Spectra with Standard Deviation")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Reflectance")

    # Calculate for first image
    real_avg1, real_std1 = calculate_average_and_std(image1, real_pixels1)
    fake_avg1, fake_std1 = calculate_average_and_std(image1, fake_pixels1)

    # Calculate for second image
    real_avg2, real_std2 = calculate_average_and_std(image2, real_pixels2)
    fake_avg2, fake_std2 = calculate_average_and_std(image2, fake_pixels2)

    # Combine real and fake averages for both images
    real_avg_combined = (real_avg1 + real_avg2) / 2
    real_std_combined = (real_std1 + real_std2) / 2
    fake_avg_combined = (fake_avg1 + fake_avg2) / 2
    fake_std_combined = (fake_std1 + fake_std2) / 2

    # Plot real blood spectra with standard deviation
    ax.plot(real_avg_combined, label="Real Blood (Average)", color='green')
    ax.fill_between(range(len(real_avg_combined)),
                    real_avg_combined - real_std_combined,
                    real_avg_combined + real_std_combined,
                    color='green', alpha=0.2, label="Real Blood (Std Dev)")

    # Plot fake blood spectra with standard deviation
    ax.plot(fake_avg_combined, label="Fake Blood (Average)", color='blue')
    ax.fill_between(range(len(fake_avg_combined)),
                    fake_avg_combined - fake_std_combined,
                    fake_avg_combined + fake_std_combined,
                    color='blue', alpha=0.2, label="Fake Blood (Std Dev)")

    ax.legend()
    plt.show()

# Define paths to the images and annotations
image_path1 = "task4output_B.png"  # Replace with the first image path
annotation_file1 = r"HyperBlood\anno\B_1.npz"  # Replace with the first annotation file path

image_path2 = "task4output_D"  # Replace with the second image path
annotation_file2 = r"HyperBlood\anno\D_1.npz"  # Replace with the second annotation file path

# Load the images and annotations
image1 = np.array(Image.open(image_path1))
gt_array1 = load_annotations(annotation_file1)

image2 = np.array(Image.open(image_path2))
gt_array2 = load_annotations(annotation_file2)

# Example pixel selections (from task9)

real_pixels1, fake_pixels1, real_pixels2, fake_pixels2 = task9(image1, gt_array1, image2, gt_array2)

# Plot combined average spectra
plot_average_spectra(image1, real_pixels1, fake_pixels1, image2, real_pixels2, fake_pixels2)