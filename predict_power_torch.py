import torch
import torch.nn as nn
import torch.optim as optim
import casadi as ca
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Your data loading and preprocessing steps
directory_path = "C:\\Users\\augus\\OneDrive\\Documents\\Master2024code\\MPC_2D\\results_2802"
file_prefix = "chunk_results_"
all_files = [os.path.join(directory_path, f"{file_prefix}{i}.csv") for i in range(16)]

df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)
data.loc[~data['success'], ['average_energy', 'average_velocity']] = 0

X = data[['alpha_h', 'average_velocity', 'average_energy']].values
y = data[['omega_h', 'delta_h']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X
y_train = y

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)

# Define the PyTorch model
class PyTorchMLP(nn.Module):
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # Assuming input features are of size 2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Assuming a single output
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = PyTorchMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):  # Number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Optionally print the loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Inference mode, gradient calculation is unnecessary
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')


# Saving weights and biases to a file
torch.save({
    'weights_fc1': model.fc1.weight.data.numpy(),
    'bias_fc1': model.fc1.bias.data.numpy(),
    'weights_fc2': model.fc2.weight.data.numpy(),
    'bias_fc2': model.fc2.bias.data.numpy(),
}, 'model_params.pth')




# Extract the trained weights and biases
weights_fc1 = model.fc1.weight.data.numpy()
bias_fc1 = model.fc1.bias.data.numpy()
weights_fc2 = model.fc2.weight.data.numpy()
bias_fc2 = model.fc2.bias.data.numpy()

# CasADi model construction
x = ca.MX.sym('x', 2)
W1 = ca.MX(weights_fc1)
b1 = ca.MX(bias_fc1)
W2 = ca.MX(weights_fc2)
b2 = ca.MX(bias_fc2)

# Neural network in CasADi
layer1 = ca.mtimes(W1, x) + b1
layer1_act = ca.fmax(layer1, 0)  # ReLU activation
output = ca.mtimes(W2, layer1_act) + b2

# Create a CasADi function for the neural network
nn_casadi = ca.Function('nn', [x], [output])

# Example usage of the CasADi neural network model
input_example = np.array([[0.5, -1.5]])  # Note the double brackets for a 2D array
nn_output = nn_casadi(input_example.T)  # Transpose to match CasADi's column vector expectation
print("Neural network output with CasADi:", nn_output)

