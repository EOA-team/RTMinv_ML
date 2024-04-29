'''
Neural Network model to perform a RTM inversion

@author Selene Ledain
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pickle



class SimpleNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_layers, output_size):
    super(SimpleNeuralNetwork, self).__init__()
    self.hidden_layers = nn.ModuleList()

    # Add hidden layers
    for _ in range(hidden_layers):
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.hidden_layers.append(nn.ReLU())
        input_size = hidden_size  # Update input size for the next layer
    
    # Add the output layer
    self.output_layer = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    for layer in self.hidden_layers:
        x = layer(x)
    return self.output_layer(x)
"""

class SimpleNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNeuralNetwork, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)


  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
"""

class NeuralNetworkRegressor(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_layer, output_size, epochs=100, batch_size=32, 
  optim={'name': 'Adam', 'learning_rate': 0.01}, criterion='MSE', random_state=42):
      '''
      Neural Network regressor model

      :param input_size: number of features in the input data
      :param hidden_size: number of neurons in the hidden layer
      :param hidden_layer: number of hidden layers
      :param output_size: number of neurons in the output layer
      :param learning_rate: learning rate for optimization
      :param epochs: number of training epochs
      :param batch_size: batch size for training
      :param optim_kwargs: optimizer name (see torch.optim) and parameters to set it up
      :param random_state: 
      '''
      super(NeuralNetworkRegressor, self).__init__()

      self.model = SimpleNeuralNetwork(input_size, hidden_size, hidden_layer, output_size)
      self.epochs = epochs
      self.batch_size = batch_size
      self.random_state = random_state

      # Define loss function and optimizer
      self.optim_kwargs = optim
      self.optimizer = self.get_optim(self.optim_kwargs)
      self.criterion = self.get_criterion(criterion) # nn.L1Loss() , nn.MSELoss()  


  def get_criterion(self, criterion: str) -> optim:
      ''' 
      Initialise optimizer criterion for the model training

      :param criterion: str with crtierion name
      See torch.nn documentation for options and details
      :returns: Instatiated loss function for the optimizer
      '''

      if criterion == 'MSE':
        return nn.MSELoss()
      if criterion == 'L1':
        return nn.L1Loss()
      else:
        raise Exception(f'Criterion {criterion} not implemented, please select another')


  def get_optim(self, optim_kwargs: dict) -> optim:
      ''' 
      Initialise optimizer for the model training

      :param optim_kwargs: dictionary with optimizer name and parameters to instantiate it
      See torch.nn.optim documentation for options and details
      :returns: Instatiated optimizer
      '''

      if optim_kwargs['name'] == 'Adam':
        return optim.Adam(self.model.parameters(), **{k: v for k, v in optim_kwargs.items() if k != 'name'})
      if optim_kwargs['name'] == 'SGD':
        return optim.SGD(self.model.parameters(), **{k: v for k, v in optim_kwargs.items() if k != 'name'})
      else:
        raise Exception(f'Optimizer {optim_kwargs["name"]} not implemented, please select another')


  def forward(self, x):
        return self.model(x)
        

  def fit(self, X: np.array, y: np.array):
      '''
      Fit the model on the training set
      :param X: training features. Is tensor if GPU used
      :param y: training labels. Is tensor if GPU used
      '''
      torch.manual_seed(self.random_state)
      # Convert NumPy arrays to PyTorch tensors
      if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).view(-1, 1)

      # Create a DataLoader for batch training
      train_dataset = TensorDataset(X, y)
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
              print(f'Epoch [{epoch+1}/{self.epochs}], Loss (MSE): {loss.item():.4f}')

  def predict(self, X_test: np.array):
      '''
      Make predictions on test set

      :param X_test: test features
      :return: predictions
      '''
      self.model.eval()

      # Convert NumPy array to PyTorch tensor
      X_test_tensor = torch.FloatTensor(X_test) if not torch.is_tensor(X_test) else X_test

      # Make predictions on the testing set
      predictions = self.model(X_test_tensor).detach().cpu().numpy()

      return predictions

  def test_scores(self, y_test: pd.Series, y_pred: np.array, dataset: str):
      '''
      Compute scores on the test set

      :param y_test: test labels
      :param y_pred: test predictions
      :param dataset: dataset name
      '''
      # Move y_pred to CPU if it's on CUDA device
      if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
          y_pred = y_pred.cpu().detach().numpy()
      if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
          y_test = y_test.cpu().detach().numpy()

      # Compute different scores on the test set
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      test_mae = mean_absolute_error(y_test, y_pred)
      test_r2 = r2_score(y_test, y_pred)
      slope, intercept = np.polyfit(y_test.flatten(), y_pred.flatten(), 1)
      rmselow = mean_squared_error(y_test[y_test<3], y_pred[y_test<3], squared=False)
      print(f'{dataset} RMSE: {test_rmse}')
      print(f'{dataset} MAE: {test_mae}')
      print(f'{dataset} R2: {test_r2}')
      print('Regression slope:', slope)
      print('Regression intercept:', intercept)
      print(f'{dataset} rmselow: {rmselow}')

  def save(self, model, model_filename: str) -> None:
      ''' 
      Save trained model

      :param model: trained model
      :param model_filename: path to save model as .pkl file
      '''
      # Save the trained model to a file using pickle
      with open(model_filename, 'wb') as file:
          pickle.dump(model, file)
      print(f'Trained model saved to {model_filename}')
