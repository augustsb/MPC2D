from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the PyTorch model
class PyTorchMLP(nn.Module):
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(1, 128)  # Assuming input features are of size 2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Assuming a single output
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


def train_neural_network(X_train, y_train, X_test, y_test):

        # Convert DataFrame to numpy array if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.values

    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64)

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

    # Saving weights and biases to a file
    torch.save({
        'weights_fc1': model.fc1.weight.data.numpy(),
        'bias_fc1': model.fc1.bias.data.numpy(),
        'weights_fc2': model.fc2.weight.data.numpy(),
        'bias_fc2': model.fc2.bias.data.numpy(),
    }, 'model_params.pth')

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Inference mode, gradient calculation is unnecessary
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item()}')

    return model



def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression - Mean Squared Error: {mse}")
    print(f"Linear Regression - R-squared: {r2}")

    return model


def train_random_forest(X_train, y_train, X_test, y_test, model_path='model_rf.joblib'):
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    dump(model_rf, model_path)
    print(f"Random Forest - Mean Squared Error: {mse}")
    print(f"Random Forest - R-squared: {r2}")

    return model_rf


def train_polynomial_regression(X, y, X_test, y_test, degree=2):

    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    #poly_model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X_test)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    print(f"Polynomial Regression - Mean Squared Error: {mse_poly}")
    print(f"Polynomial Regression - R-squared: {r2_poly}")
    # Extract components if needed for further analysis
    return poly_model