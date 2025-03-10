'''
Test a trained model

@author Selene Ledain
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from argparse import ArgumentParser
import yaml
from typing import Dict, Tuple, Union, Any
import pickle
import torch
import time
from sklearn.metrics import mean_squared_error
from scipy import stats
import os

from models import MODELS

def load_config(config_path: str) -> Dict:
  ''' 
  Load configuration file

  :param config_path: path to yaml file
  :returns: dictionary of parameters
  '''
  with open(config_path, "r") as config_file:
      config = yaml.safe_load(config_file)
  return config


def prepare_data_train(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare training sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  data_path = config['Data']['data_path']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
    X = df[config['Data']['train_cols']]
    y = df[config['Data']['target_col']]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])

    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_pickle(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))

    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([y_train , y_soil], ignore_index=True)

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X_train.diff(axis=1)
      for col in X_train.columns[1:]:
          X_train[col + '_derivative'] = derivatives[col]
      derivatives = X_test.diff(axis=1)
      for col in X_test.columns[1:]:
          X_test[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X_train['ndvi'] = (X_train['B08'] - X_train['B04'])/(X_train['B08'] + X_train['B04'])
      X_test['ndvi'] = (X_test['B08'] - X_test['B04'])/(X_test['B08'] + X_test['B04'])

    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)
      return X_train, X_test, y_train, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

  elif isinstance(data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    # Sample 50000 data pairs
    #sampled_df = concatenated_df.sample(50000, random_state=config['Seed']) if len(concatenated_df) > 50000 else concatenated_df
    X = concatenated_df[config['Data']['train_cols']] #sampled_df[config['Data']['train_cols']] #
    y = concatenated_df[config['Data']['target_col']] #sampled_df[config['Data']['target_col']] #  
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])

    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_pickle(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))
    
    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([pd.Series(y_train), y_soil], ignore_index=True)

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X_train.diff(axis=1)
      for col in X_train.columns[1:]:
          X_train[col + '_derivative'] = derivatives[col]
      derivatives = X_test.diff(axis=1)
      for col in X_test.columns[1:]:
          X_test[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X_train['ndvi'] = (X_train['B08'] - X_train['B04'])/(X_train['B08'] + X_train['B04'])
      X_test['ndvi'] = (X_test['B08'] - X_test['B04'])/(X_test['B08'] + X_test['B04'])

    #print(len(X_train), len(X_test))
    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)
      return X_train, X_test, y_train, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

  else:
      return None


def prepare_data_test(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare test sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  data_path = config['Data']['val_data_path']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
    df = df[~df[config['Data']['target_col']].isna()]
    X = df[config['Data']['train_cols']]
    y = df[config['Data']['target_col']]

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X.diff(axis=1)
      for col in X.columns[1:]:
          X[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X['ndvi'] = (X['B08'] - X['B04'])/(X['B08'] + X['B04'])
      

    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X = scaler.transform(X)
      return X, y.values
      #X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])
      #print('here')
      #return X_test, y_test
    else:
      return X, y.values

  elif isinstance(data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    concatenated_df = concatenated_df[~concatenated_df[config['Data']['target_col']].isna()]
    X = concatenated_df[config['Data']['train_cols']] #  concatenated_df[config['Data']['train_cols']]
    y = concatenated_df[config['Data']['target_col']] #  concatenated_df[config['Data']['target_col']]

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X.diff(axis=1)
      for col in X.columns[1:]:
          X[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X['ndvi'] = (X['B08'] - X['B04'])/(X['B08'] + X['B04'])
      
          
    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X = scaler.transform(X)
      return X, y.values
    else:
      return X, y.values

  else:
      return None


def build_model(config: dict) -> Any:
  ''' 
  Instantiated model

  :param config: dictionary of configuration parameters
  :returns: model
  '''
  model_name = config['Model']['name']
  if model_name not in MODELS:
    raise ValueError(f"Invalid model type: {model_name}")
  else:
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model = MODELS[model_name](**model_params)
  
  return model



def test_model(config: dict) -> None:
  ''' 
  Test model on a dataset and get scores

  :param config: dictionary of configuration parameters
  '''

  if not isinstance(config['Seed'], list):
    config['Seed'] = [config['Seed']]

  model_basename = config['Model']['save_path']  
  save_model = config['Model'].pop('save')
  score_path = config['Model'].pop('score_path') if 'score_path' in config['Model'].keys() else None
    

  for seed in config['Seed']:
    print('Running with seed', seed)

    config['Model']['save_path'] = model_basename.split('.')[0] + f'{seed}.pkl'

    #############################################
    # DATA
    X_test, y_test = prepare_data_test(config=config) # unseen validation data (in situ)
    #_, X_train, _, y_train = prepare_data_train(config=config) # performance on training data

    # Move data to CUDA if GPUs requested and available
    device = torch.device('cuda' if config['Model'].get('gpu') and torch.cuda.is_available() else 'cpu')
    print(device)
    if device == torch.device('cuda'):
        X_test, y_test = (
            torch.FloatTensor(X_test).to(device),
            torch.FloatTensor(y_test).view(-1, 1).to(device)
        )
        """ 
        X_train, y_train = (
            torch.FloatTensor(X_train).to(device),
            torch.FloatTensor(y_train).view(-1, 1).to(device)
        )
        """

    #############################################
    # MODEL
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path') 
    with open(model_filename, 'rb') as f:
      model = pickle.load(f)

    # Move model to CUDA if GPUs are available
    if device == torch.device('cuda'):
        model.to(device)

    #############################################
    # TEST
    if model_name == 'GPR': # Active learning
      y_pred, y_std = model.predict(X_test, return_std=True)
      print('Mean test std', y_std.mean()) 
      model.test_scores(y_test=y_test, y_pred=y_pred)
    else: 
      y_pred = model.predict(X_test=X_test)
      if not np.isnan(y_pred.flatten()).any():
        model.test_scores(y_test=y_test.flatten(), y_pred=y_pred.flatten(), dataset=f'Val {seed}', score_path=score_path)

        # Compute r2 and nRMSE
        compute_other_scores(y_test=y_test.flatten(), y_pred=y_pred.flatten(), dataset=f'Val {seed}', score_path=score_path)        

  return


def compute_other_scores(y_test, y_pred, dataset, score_path):
  """ 
  Compute nromalised RMSE and pearson's r squared, and add to score_path
  """

  # Move y_pred to CPU if it's on CUDA device
  if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
      y_pred = y_pred.cpu().detach().numpy()
  if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
      y_test = y_test.cpu().detach().numpy()

  nrmse = mean_squared_error(y_test, y_pred, squared=False)/(np.max(y_test) - np.min(y_test))
  pearson = stats.pearsonr(y_test, y_pred).statistic

  print(f'nRMSE: {nrmse}')
  print(f'Pearson r2: {pearson**2}')
  """
  # Open excel file at score_path and append results
  score_data = {
      'Dataset': [dataset],
      'nRMSE': [nrmse],
      'r2': [pearson**2],
  }
  score_df = pd.DataFrame(score_data)
  """
  if score_path is not None:
    if os.path.exists(score_path):
        existing_df = pd.read_excel(score_path)
        if 'nRMSE' not in existing_df.columns:
            existing_df['nRMSE'] = [None]*len(existing_df)
        existing_df.loc[existing_df['Dataset'] == dataset, 'nRMSE']= nrmse
        if 'r2' not in existing_df.columns:
            existing_df['r2'] = [None]*len(existing_df)
        existing_df.loc[existing_df['Dataset'] == dataset, 'r2'] = pearson**2
      
        existing_df.to_excel(score_path, index=False)

  return

    
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
  args = parser.parse_args()

  config = load_config(args.setting)
  test_model(config)
