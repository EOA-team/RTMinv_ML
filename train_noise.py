'''
Pass multiple config files and call the training script for each model

@author Selene Ledain
'''

import os
import yaml
from train import load_config, train_model
from test import prepare_data_test
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import r2_score

def test_model(config: dict) -> None:
    ''' 
    Test model on a dataset and get scores

    :param config: dictionary of configuration parameters
    '''

    #############################################
    # DATA
    X_test, y_test = prepare_data_test(config=config)

    # Move data to CUDA if GPUs requested and available
    device = torch.device('cuda' if config['Model'].get('gpu') and torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
      X_test, y_test = (
        torch.FloatTensor(X_test).to(device),
        torch.FloatTensor(y_test).view(-1, 1).to(device),
      ) 

    #############################################
    # MODEL
    save_model = config['Model'].pop('save')
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path') 
    with open(model_filename, 'rb') as f:
      model = pickle.load(f)

    #############################################
    # TEST

    if model_name == 'GPR': # Active learning
      y_pred, y_std = model.predict(X_test, return_std=True)
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      return model_name, test_rmse, y_std.mean()
    else: 
      y_pred = model.predict(X_test=X_test)
      # Move y_pred to CPU if it's on CUDA device
      if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
          y_pred = y_pred.cpu().detach().numpy()
      if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
          y_test = y_test.cpu().detach().numpy()
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      test_mae = mean_absolute_error(y_test, y_pred)
      slope, intercept = np.polyfit(y_test, y_pred, 1)
      rmse_low = mean_squared_error(y_test[y_test<3], y_pred[y_test<3], squared=False)

      # Fabio score
      fabio = abs(np.mean(y_test-y_pred)) + np.std(y_test-y_pred) - np.sqrt(np.cov(y_test, y_pred.flatten())[0,1])
      fabio2 = abs(np.mean(y_test-y_pred)) + np.std(y_test-y_pred)

      return model_name, test_rmse, test_mae, r2_score(y_test, y_pred), slope[0], intercept[0], rmse_low, fabio, fabio2


def test_model_lowLAI(config: dict) -> None:
    ''' 
    Test model on a dataset for LAI<=3 and get scores

    :param config: dictionary of configuration parameters
    '''

    #############################################
    # DATA
    X_test, y_test = prepare_data(config=config)

    X_test = X_test[y_test > 3] #X_test[y_test <= 3]
    y_test = y_test[y_test > 3] #y_test[y_test <= 3]

    # Move data to CUDA if GPUs requested and available
    device = torch.device('cuda' if config['Model'].get('gpu') and torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
      X_test, y_test = (
        torch.FloatTensor(X_test).to(device),
        torch.FloatTensor(y_test).view(-1, 1).to(device),
      ) 

    #############################################
    # MODEL
    save_model = config['Model'].pop('save')
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path') 
    with open(model_filename, 'rb') as f:
      model = pickle.load(f)

    #############################################
    # TEST

    if model_name == 'GPR': # Active learning
      y_pred, y_std = model.predict(X_test, return_std=True)
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      return model_name, test_rmse, y_std.mean()
    else: 
      y_pred = model.predict(X_test=X_test)
      # Move y_pred to CPU if it's on CUDA device
      if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
          y_pred = y_pred.cpu().detach().numpy()
      if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
          y_test = y_test.cpu().detach().numpy()
      test_rmse = mean_squared_error(y_test, y_pred, squared=False)
      return model_name, test_rmse, r2_score(y_test, y_pred)


if __name__ == "__main__":

    config_path = 'configs/config_NN.yaml'

    noise_levels = [1, 3, 5, 10, 15, 20]
    noise_types = ['additive', 'multiplicative', 'combined', 'inverse', 'inverse_combined'] 

    results = {noise_type: {'rmse': [], 'std': [], 'slope': [], 'intercept': [], 'rmselow': [], 'fabio': [], 'fabio2': []} for noise_type in noise_types}

    for noise_type in noise_types:
      for noise_level in noise_levels:
        print(f'Training with noise {noise_level}% {noise_type}')

        # Modify config: pass data with noise, change model name
        config = load_config(config_path)
        config['Model']['save_path'] = config['Model']['save_path'].split('.pkl')[0] + f'_{noise_type}{noise_level}_noisev2.pkl'
        config['Data']['data_path'] = [f.split('.pkl')[0] + f'_{noise_type}{noise_level}_v2.pkl' for f in config['Data']['data_path']]
        config_test = copy.deepcopy(config)

        #train_model(config)
        model_name, test_rmse, test_mae, y_std, slope, intercept, rmse_low, fabio, fabio2 = test_model(config_test)
        #model_name, test_rmse, y_std = test_model_lowLAI(config_test)
        results[noise_type]['rmse'].append(test_rmse)
        results[noise_type]['std'].append(y_std)  
        results[noise_type]['slope'].append(slope)  
        results[noise_type]['intercept'].append(intercept)  
        results[noise_type]['rmselow'].append(rmse_low)
        results[noise_type]['fabio'].append(fabio) 
        results[noise_type]['fabio2'].append(fabio2)  
 
    # Save results to Excel
    data = {'Noise Level': noise_levels}
    for noise_type, values in results.items():
        data[f'{noise_type}_rmse'] = values['rmse']
        data[f'{noise_type}_std/r2'] = values['std']
        data[f'{noise_type}_std/r2'] = values['std']
        data[f'{noise_type}_slope'] = values['slope']
        data[f'{noise_type}_intercept'] = values['intercept']
        data[f'{noise_type}_rmselow'] = values['rmselow']
        data[f'{noise_type}_fabio'] = values['fabio']
        data[f'{noise_type}_fabio2'] = values['fabio2']

    df_results = pd.DataFrame(data)
    df_results.to_excel('../results/noise_results_NN_noisev2.xlsx', index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    for noise_type in noise_types:
        plt.plot(df_results['Noise Level'], df_results[f'{noise_type}_rmse'], label=noise_type)

    plt.xlabel('Noise Level')
    plt.ylabel('Test RMSE')
    plt.title('NN with noise RMSE')
    plt.legend()
    plt.savefig('../results/noise_results_NN_noisev2.png')  # Save plot as image
