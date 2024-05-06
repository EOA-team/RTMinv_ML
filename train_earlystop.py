'''
Train a model to perform a RTM inversion

@author Selene Ledain
'''

import os
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
import test

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


def prepare_data(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare training and testing sets

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
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
      os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
      
      with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
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
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
      os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

      with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
      return X_train, X_test, y_train.values, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

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
    #if model_name == 'NN':
      #torch.manual_seed(config['Seed'])
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model = MODELS[model_name](**model_params)
  
  return model



def train_model(config: dict) -> None:
  ''' 
  Train model on training set, get scores on test set, save model

  :param config: dictionary of configuration parameters
  '''

  torch.manual_seed(config['Seed'])
  np.random.seed(config['Seed'])

  #############################################
  # DATA
  X_train, X_test, y_train, y_test = prepare_data(config=config)
  X_val, y_val = test.prepare_data_test(config=config)

  # Move data to CUDA if GPUs requested and available
  device = torch.device('cuda' if config['Model'].get('gpu') and torch.cuda.is_available() else 'cpu')
  #print(torch.cuda.is_available())
  if device == torch.device('cuda'):
    X_train, X_test, y_train, y_test = (
      torch.FloatTensor(X_train).to(device),
      torch.FloatTensor(X_test).to(device),
      torch.FloatTensor(y_train).view(-1, 1).to(device),
      torch.FloatTensor(y_test).view(-1, 1).to(device),
    ) 

  #############################################
  # MODEL
  if config['Model'].get('gpu') and torch.cuda.is_available():
    print('Using GPUs')
  if 'gpu' in config['Model'].keys():
    config['Model'].pop('gpu')

  save_model = config['Model'].pop('save')
  model_name = config['Model']['name']
  model_filename = config['Model'].pop('save_path') if 'save_path' in config['Model'].keys() else model_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl' # path to save trained model 
  model = build_model(config=config)
  if device == torch.device('cuda'):
    model.to(device)
  #print(f"Model on: {next(model.parameters()).device}")
  #print(f"Data on: {X_test.device}")

   
  #############################################
  # TRAIN
  if model_name == 'GPR': # Active learning
    model.fit(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test)
  elif model_name == 'NN2': # NN with early stopping
    model.fit(X=X_train,y=y_train, X_val=X_val, y_val=y_val)
  else:
    model.fit(X=X_train, y=y_train)
    #############################################
    # TEST 
    y_pred = model.predict(X_test=X_test)
    model.test_scores(y_test=y_test, y_pred=y_pred, dataset='Test')

  #############################################
  # SAVE 
  if save_model:
    model.save(model=model, model_filename=model_filename)

  return


    
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
  args = parser.parse_args()

  config = load_config(args.setting)
  train_model(config)
