import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import spectral
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_hsi_cube(header_path, data_path=None):
    if data_path:
        return spectral.envi.open(header_path, data_path).load()
    return spectral.envi.open(header_path).load()

def extract_spectra(hsi_cube, real_pixels, fake_pixels):
    real_spectra = [hsi_cube[y, x, :] for x, y in real_pixels]
    fake_spectra = [hsi_cube[y, x, :] for x, y in fake_pixels]
    real_labels = [0] * len(real_spectra)  # Label for real blood
    fake_labels = [1] * len(fake_spectra)  # Label for fake blood
    return np.array(real_spectra), np.array(fake_spectra), np.array(real_labels), np.array(fake_labels)

def prepare_data(hsi_cube1, real_pixels_1, fake_pixels_1, background_pixels_1, hsi_cube2, real_pixels_2, fake_pixels_2, background_pixels_2):
    # Extract and concatenate data from multiple cubes
    real_spectra_1, fake_spectra_1, real_labels_1, fake_labels_1 = extract_spectra(hsi_cube1, real_pixels_1, fake_pixels_1)
    background_spectra_1 = [hsi_cube1[y, x, :] for x, y in background_pixels_1]
    background_labels_1 = [2] * len(background_spectra_1)  # Label for background

    real_spectra_2, fake_spectra_2, real_labels_2, fake_labels_2 = extract_spectra(hsi_cube2, real_pixels_2, fake_pixels_2)
    background_spectra_2 = [hsi_cube2[y, x, :] for x, y in background_pixels_2]
    background_labels_2 = [2] * len(background_spectra_2)  # Label for background

    # Convert lists to numpy arrays
    background_spectra_1 = np.array(background_spectra_1)
    background_spectra_2 = np.array(background_spectra_2)

    # Debug prints to check shapes
    # print(f"real_spectra_1 shape: {real_spectra_1.shape}")
    # print(f"fake_spectra_1 shape: {fake_spectra_1.shape}")
    # print(f"background_spectra_1 shape: {background_spectra_1.shape}")
    # print(f"real_spectra_2 shape: {real_spectra_2.shape}")
    # print(f"fake_spectra_2 shape: {fake_spectra_2.shape}")
    # print(f"background_spectra_2 shape: {background_spectra_2.shape}")

    # Concatenate only non-empty arrays
    spectra_list = [real_spectra_1, fake_spectra_1, background_spectra_1, real_spectra_2, fake_spectra_2, background_spectra_2]
    spectra_list = [s for s in spectra_list if s.size > 0]
    X = np.concatenate(spectra_list, axis=0)

    labels_list = [real_labels_1, fake_labels_1, background_labels_1, real_labels_2, fake_labels_2, background_labels_2]
    labels_list = [l for l in labels_list if len(l) > 0]
    y = np.concatenate(labels_list, axis=0)

    # Split and train as usual
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load the HSI cubes
header_paths = ['HyperBlood/data/E_1.hdr', 'HyperBlood/data/F_1.hdr']
data_paths = ['HyperBlood/data/E_1.float', 'HyperBlood/data/F_1.float']
hsi_cubes = [load_hsi_cube(header, data) for header, data in zip(header_paths, data_paths)]

# Load the labeled pixels from the JSON file
with open('labeled_pixels.json', 'r') as f:
    labeled_pixels = json.load(f)

# Map each key's values to the respective real, fake, and background pixel index
real_pixels_list = []
fake_pixels_list = []
background_pixels_list = []

for key in labeled_pixels:
    real_pixels_list.append(labeled_pixels[key]["real_pixels"])
    fake_pixels_list.append(labeled_pixels[key]["fake_pixels"])
    background_pixels_list.append(labeled_pixels[key].get("background_pixels", []))  # Add background pixels if available

# Example: Extract data for the first two entries
real_pixels1 = real_pixels_list[0]
fake_pixels1 = fake_pixels_list[0]
background_pixels1 = background_pixels_list[0]
real_pixels2 = real_pixels_list[1]
fake_pixels2 = fake_pixels_list[1]
background_pixels2 = background_pixels_list[1]

X_train, X_test, y_train, y_test = prepare_data(hsi_cubes[0], real_pixels1, fake_pixels1, background_pixels1, hsi_cubes[1], real_pixels2, fake_pixels2, background_pixels2)

class HSIClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HSIClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = hsi_cubes[0].shape[2]  # Assuming all cubes have the same number of spectral bands
hidden_size = 128
output_size = 3  # Three classes: real blood, fake blood, background
learning_rate = 0.001

# Instantiate the model, define the loss function and the optimizer
model = HSIClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1)  # Ensure y_train_tensor is 1D
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).view(-1)  # Ensure y_test_tensor is 1D

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    outputs = outputs.view(-1, output_size)  # Flatten the outputs to match the target shape
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
print("Evaluation started...")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_outputs = test_outputs.view(-1, output_size)  # Flatten the outputs to match the target shape
    _, predicted = torch.max(test_outputs, 1)  # Get the predicted labels
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Ensure y_test_tensor and predicted are 1D tensors
y_test_tensor = y_test_tensor.view(-1)
predicted = predicted.view(-1)

# Calculate accuracy
accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f'Accuracy: {accuracy:.4f}')

print("Evaluation complete!")

# Load a new HSI cube
new_header_path = 'HyperBlood/data/A_1.hdr'
new_data_path = 'HyperBlood/data/A_1.float'  # Specify the data file path if needed
new_hsi_cube = load_hsi_cube(new_header_path, new_data_path)

# Flatten the new HSI cube for prediction
height, width, num_bands = new_hsi_cube.shape
new_hsi_cube_flat = new_hsi_cube.reshape(-1, num_bands)
new_hsi_cube_tensor = torch.tensor(new_hsi_cube_flat, dtype=torch.float32)

print("Making predictions on a new HSI cube...")
# Make predictions
model.eval()
with torch.no_grad():
    new_predictions = model(new_hsi_cube_tensor)
    _, new_predicted_labels = torch.max(new_predictions, 1)

# Reshape the predicted labels to the original image dimensions
new_predicted_labels = new_predicted_labels.numpy().reshape(height, width)


# Flatten predictions and true labels
preds = predicted.flatten().numpy()
labels = y_test_tensor.flatten().numpy()

# Print a classification report
print(classification_report(labels, preds, target_names=["Fake Blood", "Real Blood", "Background"]))

# Print a confusion matrix
cm = confusion_matrix(labels, preds)
print("Confusion Matrix:\n", cm)
print(
    "Explanation of the confusion matrix:\n"
    "For three-class classification (Fake Blood, Real Blood, Background):\n"
    "  - cm[0,0]: True Negatives (Fake Blood correctly classified)\n"
    "  - cm[0,1]: False Positives (Fake Blood misclassified as Real Blood)\n"
    "  - cm[0,2]: False Positives (Fake Blood misclassified as Background)\n"
    "  - cm[1,0]: False Negatives (Real Blood misclassified as Fake Blood)\n"
    "  - cm[1,1]: True Positives (Real Blood correctly classified)\n"
    "  - cm[1,2]: False Positives (Real Blood misclassified as Background)\n"
    "  - cm[2,0]: False Negatives (Background misclassified as Fake Blood)\n"
    "  - cm[2,1]: False Negatives (Background misclassified as Real Blood)\n"
    "  - cm[2,2]: True Positives (Background correctly classified)"
)

# Plot the confusion matrix

# Define the color map and labels
colors = ['red', 'blue', 'gray']  # Corrected: Ensure 'red' is for Real and 'blue' is for Fake
class_labels = ['Real Blood', 'Fake Blood', 'Background']
cmap = mcolors.ListedColormap(colors)

# Plot the classified image
plt.figure(figsize=(10, 6))
plt.imshow(new_predicted_labels, cmap=cmap, interpolation='nearest')
cbar = plt.colorbar()
cbar.set_ticks([0.33, 1, 1.66])  # Place ticks at the center of each color
cbar.set_ticklabels(class_labels)  # Set labels for the classes
cbar.set_label('Class')

# Add titles and axes
plt.title("Predicted Labels for New HSI Cube")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.xticks([])
plt.yticks([])
plt.show()

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake Blood", "Real Blood", "Background"],
            yticklabels=["Fake Blood", "Real Blood", "Background"])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Optional: Representative Spectra
fake_spectra_example = X_train[y_train == 0][:1].squeeze()  # Take one example from Fake Blood
real_spectra_example = X_train[y_train == 1][:1].squeeze()  # Take one example from Real Blood
background_spectra_example = X_train[y_train == 2][:1].squeeze()  # Take one example from Background

plt.figure(figsize=(10, 6))
plt.plot(fake_spectra_example, label="Fake Blood", color="blue")
plt.plot(real_spectra_example, label="Real Blood", color="red")
plt.plot(background_spectra_example, label="Background", color="gray")
plt.title("Representative Spectra for Each Class")
plt.xlabel("Spectral Band")
plt.ylabel("Reflectance/Intensity")
plt.legend()
plt.show()
