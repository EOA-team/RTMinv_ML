'''
Train a model to perform a RTM inversion

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
    if config['Data']['normalize']:
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
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
    sampled_df = concatenated_df.sample(50000, random_state=config['Seed'])
    X = sampled_df[config['Data']['train_cols']] #  concatenated_df[config['Data']['train_cols']]
    y = sampled_df[config['Data']['target_col']] #  concatenated_df[config['Data']['target_col']]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])
    if config['Data']['normalize']:
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
      with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
      return X_train, X_test, y_train, y_test
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
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model = MODELS[model_name](**model_params)
  
  return model



def train_model(config: dict) -> None:
  ''' 
  Train model on training set, get scores on test set, save model

  :param config: dictionary of configuration parameters
  '''
  #############################################
  # DATA
  X_train, X_test, y_train, y_test = prepare_data(config=config)

  #############################################
  # MODEL
  save_model = config['Model'].pop('save')
  model_name = config['Model']['name']
  model_filename = config['Model'].pop('save_path') if 'save_path' in config['Model'].keys() else model_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl' # path to save trained model 
  model = build_model(config=config)
   
  #############################################
  # TRAIN
  if model_name == 'GPR': # Active learning
    model.fit(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test)
  else:
    model.fit(X=X_train, y=y_train)
    #############################################
    # TEST 
    y_pred = model.predict(X_test=X_test)
    model.test_scores(y_test=y_test, y_pred=y_pred)

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
