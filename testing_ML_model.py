import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = 10  # Example input size
hidden_size = 5  # Example hidden layer size
output_size = 2  # Example output size
learning_rate = 0.001

# Instantiate the model, define the loss function and the optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Generate some example data
    inputs = torch.randn(10, input_size)  # Batch size of 10
    labels = torch.randint(0, output_size, (10,))  # Random labels

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# **************************************************************************

import matplotlib.pyplot as plt

# Generate some new example data
new_inputs = torch.randn(5, input_size)  # Batch size of 5

# Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    predictions = model(new_inputs)
    predicted_labels = torch.argmax(predictions, dim=1)

# Visualize the predictions
plt.figure(figsize=(10, 5))
plt.plot(predicted_labels.numpy(), 'o', label='Predicted Labels')
plt.title('Model Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Label')
plt.legend()
plt.show()
