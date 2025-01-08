import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def load_scene(scene_label, anno_folder_path="HyperBlood/anno"):
    """
    Load the data for the given scene (A, B, C, or D) from an .npz file.
    Assumes the file is named A_1.npz, B_1.npz, etc., stored in anno_folder_path.
    
    Parameters:
        scene_label (str): One of 'A', 'B', 'C', or 'D'.
        anno_folder_path (str): Path to the folder containing .npz annotation files.

    Returns:
        np.ndarray: The loaded RGB image data under the 'gt' key.
    """
    # Construct full path for the desired scene
    filename = f"{scene_label}_1.npz"  # e.g., 'A_1.npz'
    scene_path = os.path.join(anno_folder_path, filename)

    # Load the .npz file
    data = np.load(scene_path, allow_pickle=True)

    # For debugging: check what keys exist
    print("Available keys in the file:", data.keys())

    # Extract the ground truth image data
    gt_data = data['gt']
    print(f"Loaded scene {scene_label}:")
    print("  - Type of 'gt':", type(gt_data))
    print("  - Shape of 'gt':", gt_data.shape)  # Expect something like (H, W, 3) if it's RGB

    return gt_data

def select_pixels_and_visualize(gt_data):
    """
    Display the loaded scene (assumed to be RGB) and allow the user to select two pixels:
    one for real blood, one for fake blood.

    Parameters:
        gt_data (np.ndarray): The RGB image data to visualize.

    Returns:
        list: Selected pixel coordinates [(x1, y1), (x2, y2)].
    """
    # Create a figure
    fig, ax = plt.subplots()
    ax.set_title("Click: 1) Real blood, 2) Fake blood")

    # Show the scene as RGB
    # If 'gt_data' is already (H, W, 3) and in [0,255], this will show a color image
    ax.imshow(gt_data)

    selected_points = []

    def onclick(event):
        # Guard against clicks outside axes or if user has already selected 2 points
        if event.inaxes != ax or len(selected_points) >= 2:
            return

        # Convert float coords to integer pixel indices
        x, y = int(event.xdata), int(event.ydata)

        # Append point to the list
        selected_points.append((x, y))

        # Mark the pixel
        color = 'r' if len(selected_points) == 1 else 'b'
        label = 'Real' if len(selected_points) == 1 else 'Fake'
        
        # You can mark with a circle instead of a dot for better visualization
        circle = Circle((x, y), radius=5, color=color, fill=False, linewidth=2)
        ax.add_patch(circle)
        # Optionally add text
        ax.text(x, y, label, color=color, fontsize=12, ha='left', va='bottom',
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.8))

        fig.canvas.draw()

        # Once two points are selected, print them out
        if len(selected_points) == 2:
            print("Selected points:", selected_points)

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the figure (blocks until closed)
    plt.show()

    return selected_points


# 1. Load scene A (or B, C, D - just change the label)
scene_label = 'A'  # 'A', 'B', 'C', or 'D'
gt_data = load_scene(scene_label)

# 2. Let the user select two pixels: real blood (first) and fake blood (second)
selected_points = select_pixels_and_visualize(gt_data)

# 3. Print the final result
if len(selected_points) == 2:
    print("Real blood (red) pixel at:", selected_points[0])
    print("Fake blood (blue) pixel at:", selected_points[1])
else:
    print("Two valid points were not selected.")