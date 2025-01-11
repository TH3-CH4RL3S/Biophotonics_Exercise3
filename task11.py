import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load HSI data and ground truth labels
hsi_data = np.load('path/to/hsi_data.npy')  # Shape: (height, width, num_bands)
gt_labels = np.load('path/to/gt_labels.npy')  # Shape: (height, width)

# Flatten the HSI data and labels for train-test split
height, width, num_bands = hsi_data.shape
hsi_data_flat = hsi_data.reshape(-1, num_bands)
gt_labels_flat = gt_labels.flatten()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(hsi_data_flat, gt_labels_flat, test_size=0.2, random_state=42)

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
input_size = num_bands
hidden_size = 128
output_size = 3  # Three classes: blood, fake blood, background
learning_rate = 0.001

# Instantiate the model, define the loss function and the optimizer
model = HSIClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    print(f'Test Accuracy: {accuracy:.4f}')

# Load a new HSI cube
new_hsi_cube = np.load('path/to/new_hsi_cube.npy')  # Shape: (height, width, num_bands)
new_hsi_cube_flat = new_hsi_cube.reshape(-1, num_bands)
new_hsi_cube_tensor = torch.tensor(new_hsi_cube_flat, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
    new_predictions = model(new_hsi_cube_tensor)
    _, new_predicted_labels = torch.max(new_predictions, 1)

# Reshape the predicted labels to the original image dimensions
new_predicted_labels = new_predicted_labels.numpy().reshape(height, width)

# Visualize the classification result
plt.figure(figsize=(10, 10))
plt.imshow(new_predicted_labels, cmap='viridis')
plt.title('Classification Result')
plt.colorbar()
plt.show()