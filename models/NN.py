'''
Neural Network model to perform a RTM inversion

@author Selene Ledain
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


class SimpleNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNeuralNetwork, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x

class NeuralNetworkRegressor:
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100, batch_size=32, random_state=42):
      '''
      Neural Network regressor model

      :param input_size: number of features in the input data
      :param hidden_size: number of neurons in the hidden layer
      :param output_size: number of neurons in the output layer
      :param learning_rate: learning rate for optimization
      :param epochs: number of training epochs
      :param batch_size: batch size for training
      :param random_state: 
      '''
      self.model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.batch_size = batch_size
      self.random_state = random_state

      # Define loss function and optimizer
      self.criterion = nn.MSELoss()
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

  def fit(self, X: pd.DataFrame, y: pd.Series):
      '''
      Fit the model on the training set

      :param X: training features
      :param y: training labels
      '''
      torch.manual_seed(self.random_state)

      # Convert NumPy arrays to PyTorch tensors
      X_train_tensor = torch.FloatTensor(X.values)
      y_train_tensor = torch.FloatTensor(y.values).view(-1, 1)

      # Create a DataLoader for batch training
      train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

      # Training loop
      for epoch in range(self.epochs):
          self.model.train()

          for batch_X, batch_y in train_loader:
              # Forward pass
              outputs = self.model(batch_X)
              loss = self.criterion(outputs, batch_y)

              # Backward pass and optimization
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()

          if (epoch + 1) % 10 == 0:
              print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

  def predict(self, X_test: pd.DataFrame):
      '''
      Make predictions on test set

      :param X_test: test features
      :return: predictions
      '''
      self.model.eval()

      # Convert NumPy array to PyTorch tensor
      X_test_tensor = torch.FloatTensor(X_test.values)

      # Make predictions on the testing set
      predictions = self.model(X_test_tensor).detach().numpy()

      return predictions

  def test_scores(self, y_test: pd.Series, y_pred: np.array):
      '''
      Compute scores on the test set

      :param y_test: test labels
      :param y_pred: test predictions
      '''
      # Compute RMSE on the test set
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      print(f'Test RMSE: {test_rmse}')
