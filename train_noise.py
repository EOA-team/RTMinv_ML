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
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path')
    score_path = config['Model'].pop('score_path') if 'score_path' in config['Model'].keys() else None 
    with open(model_filename, 'rb') as f:
      model = pickle.load(f)
    if device == torch.device('cuda'):
      model.to(device)

    #############################################
    # TEST
     
    y_pred = model.predict(X_test=X_test)
    # Move y_pred to CPU if it's on CUDA device
    if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
        y_pred = y_pred.cpu().detach().numpy().flatten()
    if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
        y_test = y_test.cpu().detach().numpy().flatten()
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred)
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    rmse_low = mean_squared_error(y_test[y_test<3], y_pred[y_test<3], squared=False)

    return model_name, test_rmse, test_mae, r2_score(y_test, y_pred), slope[0], intercept[0], rmse_low


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
    results_path = '../results/noise_results_NN_nosoil.xlsx'

    noise_levels = [1, 3, 5, 10, 15, 20]
    noise_types = ['additive', 'multiplicative', 'combined', 'inverse', 'inverse_combined'] 

    
    for noise_type in noise_types:
      for noise_level in noise_levels:

        ########### 
        # TRAIN

        print(f'Training with noise {noise_level}% {noise_type}')

        # Modify config: pass data with noise, change model name
        config = load_config(config_path)
        config['Model']['save_path'] = config['Model']['save_path'].split('.pkl')[0] + f'_{noise_type}{noise_level}_seed.pkl'
        config['Data']['data_path'] = [f.split('.pkl')[0] + f'_{noise_type}{noise_level}.pkl' for f in config['Data']['data_path']]
        config['Data']['test_data_path'] = [f.split('.pkl')[0] + f'_{noise_type}{noise_level}.pkl' for f in config['Data']['test_data_path']]
        config_test = copy.deepcopy(config)

        train_model(config)

        ########### 
        # TEST

        if not isinstance(config_test['Seed'], list):
          config['Seed'] = [config_test['Seed']]
        
        model_basename = config_test['Model']['save_path']
        print(model_basename)

        for seed in config_test['Seed']:
          print('Validating with seed', seed)
          config_test['Model']['save_path'] = model_basename.split('.pkl')[0] + f'{seed}.pkl'
          model_name, test_rmse, test_mae, r2, slope, intercept, rmse_low = test_model(config_test)

          score_data = {
              'Dataset': ['Val'],
              'Type': [noise_type],
              'Level': [noise_level],
              'Seed': [seed],
              'RMSE': [test_rmse],
              'MAE': [test_mae],
              'R2': [r2],
              'Slope': [slope],
              'Intercept': [intercept],
              'RMSelow': [rmse_low]
          }
          score_df = pd.DataFrame(score_data)

          if results_path is not None:
            if os.path.exists(results_path):
                existing_df = pd.read_excel(results_path)
                updated_df = pd.concat([existing_df, score_df], ignore_index=True)
            else:
                updated_df = score_df
            updated_df.to_excel(results_path, index=False)
 
    
    # Plot
    mean_scores = updated_df.drop(['Dataset', 'Seed'], axis=1).groupby(['Type', 'Level']).mean().reset_index()
    std_scores = updated_df.drop(['Dataset', 'Seed'], axis=1).groupby(['Type', 'Level']).std().reset_index()

    # RMSE
    plt.figure(figsize=(10, 6))
    for noise_type in noise_types:
        mean_rmse = mean_scores[mean_scores.Type == noise_type]['RMSE']
        std_rmse = std_scores[std_scores.Type == noise_type]['RMSE']
        plt.errorbar(noise_levels, mean_rmse, yerr=std_rmse, label=noise_type, capsize=5)
    plt.xticks(noise_levels)
    plt.xlabel('Noise Level [%]')
    plt.ylabel('RMSE')
    plt.ylim(0,2)
    plt.title('Val set RMSE of NN with soil and noise ')
    plt.legend()
    plt.savefig('../results/NN_noise_nosoil_RMSE.png')  # Save plot as image

    # R2
    plt.figure(figsize=(10, 6))
    for noise_type in noise_types:
        mean_r2 = mean_scores[mean_scores.Type == noise_type]['R2']
        std_r2 = std_scores[std_scores.Type == noise_type]['R2']
        plt.errorbar(noise_levels, mean_r2, yerr=std_r2, label=noise_type, capsize=5)
    plt.xticks(noise_levels)
    plt.xlabel('Noise Level [%]')
    plt.ylabel('R2')
    plt.ylim(-1,1)
    plt.title('Val set RMSE of NN with soil and noise ')
    plt.legend()
    plt.savefig('../results/NN_noise_nosoil_R2.png') 