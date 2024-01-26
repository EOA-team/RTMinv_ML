'''
Train a model to perform a RTM inversion

@author Selene Ledain
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from argparse import ArgumentParser
import yaml
from typing import Dict, Tuple, Union, Any

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


def prepare_data(config: dict) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None]:
  ''' 
  Load data and prepare training and testing sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  df = pd.read_pickle(config['Data']['data_path'])
  X = df[config['Data']['train_cols']]
  y = df[config['Data']['target_col']]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['Data']['test_size'], random_state=config['Seed'])
  # Could normalise/scale data

  return X_train, X_test, y_train, y_test


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


def train_model_al(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
  ''' 
  Train model with active learning
  Repeat training multiple times on smaller subsets of training data using active learning
  Test RMSE is the avg RMSE of all the model runs

  :param X_train: training data
  :param y_train: training labels
  :param X_test: testing data
  :param y_test: testing labels
  '''
  # Call GPR fit passing test data too
  return



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
  model = build_model(config=config)
  model_name = config['Model']['name']
  model_filename = model_name +  datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl' # path to save trained model 
  
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
