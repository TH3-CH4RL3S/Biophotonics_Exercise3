import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import json
import spectral

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

def prepare_data(hsi_cube, real_pixels, fake_pixels):
    real_spectra, fake_spectra, real_labels, fake_labels = extract_spectra(hsi_cube, real_pixels, fake_pixels)
    X = np.concatenate((real_spectra, fake_spectra), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load the HSI cube
header_path = 'HyperBlood/data/B_1.hdr'
data_path = 'HyperBlood/data/B_1.float'
hsi_cube = load_hsi_cube(header_path, data_path)

# Load the labeled pixels from the JSON file
with open('labeled_pixels.json', 'r') as f:
    labeled_pixels = json.load(f)
real_pixels = labeled_pixels['real_pixels']
fake_pixels = labeled_pixels['fake_pixels']

X_train, X_test, y_train, y_test = prepare_data(hsi_cube, real_pixels, fake_pixels)

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
input_size = hsi_cube.shape[2]
hidden_size = 128
output_size = 2  # Three classes: blood, fake blood
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
num_epochs = 100
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
new_header_path = 'HyperBlood/data/C_1.hdr'
new_data_path = 'HyperBlood/data/C_1.float'  # Specify the data file path if needed
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

# import matplotlib.colors as mcolors

# # Define the class labels
# class_labels = ['Fake Blood', 'Real Blood']

# # Create a custom color map
# cmap = mcolors.ListedColormap(['blue', 'red'])
# bounds = [0, 1, 2]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# # Visualize the classification result
# plt.figure(figsize=(10, 10))
# plt.imshow(new_predicted_labels, cmap=cmap, norm=norm)
# plt.title('Classification Result')
# cbar = plt.colorbar(ticks=[0.5, 1.5])
# cbar.ax.set_yticklabels(class_labels[::-1])  # Reverse the class labels to match the color order
# cbar.set_label('Class Labels')
# plt.show()

# Flatten predictions and true labels
preds = predicted.flatten().numpy()
labels = y_test_tensor.flatten().numpy()

# Print a classification report
print(classification_report(labels, preds, target_names=["Fake Blood", "Real Blood"]))

# Print a confusion matrix
cm = confusion_matrix(labels, preds)
print("Confusion Matrix:\n", cm)
print(
    "Explanation of the confusion matrix:\n"
    "For binary classification (Fake Blood vs. Real Blood):\n"
    "  - cm[0,0]: True Negatives (Fake Blood correctly classified)\n"
    "  - cm[0,1]: False Positives (Fake Blood misclassified as Real Blood)\n"
    "  - cm[1,0]: False Negatives (Real Blood misclassified as Fake Blood)\n"
    "  - cm[1,1]: True Positives (Real Blood correctly classified)"
)

# Visualize the classification result
plt.figure(figsize=(10, 10))
plt.imshow(new_predicted_labels, cmap='viridis')
plt.title('Classification Result')
plt.colorbar()
plt.show()
