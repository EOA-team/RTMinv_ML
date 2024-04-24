''' 
Perform hyperparameter tuning for a RTM-inversion model

@Selene Ledain
'''
# Read model architecture to tune
# Read parameter grid
# What default values and data path to use? Should put in the yaml anyways?
# Send parameters to train.py
# Save results somehow (.csv?)

from argparse import ArgumentParser
import yaml
from typing import Dict, Tuple, Union, Any
from datetime import datetime
import os
import csv
from subprocess import run, PIPE
import itertools



def load_config(config_path: str) -> Dict:
  ''' 
  Load configuration file

  :param config_path: path to yaml file
  :returns: dictionary of parameters
  '''
  with open(config_path, "r") as config_file:
      config = yaml.safe_load(config_file)
  return config


def update_config(names: tuple, values: list) -> Dict:
  '''
  Update the config with a specific hyperparameter setup

  :params names: hypereparameter names
  :params values: values to set
  '''
  for i, name in enumerate(names):
    if name=='lr':
      config['Model']['optim']['lr'] = values[i]
    elif name=='optim':
      config['Model']['optim']['name'] = values[i]
    else:
      config['Model'][name] = values[i]
  
  return config


def save_updated_config(updated_config, temp_config_path):
  ''' 
  Temporarily save config file fo hyperparam config

  :params updated_config: new config dict
  :param temp_config_path: path
  '''
  with open(temp_config_path, 'w') as temp_config_file:
      yaml.dump(updated_config, temp_config_file)
  return temp_config_path


def tune_model(config: dict) -> None:
  ''' 
  Tune model based on grid of hyperparameter values

  :param config: dictionary of configuration parameters
  '''
  # Get hyperparam grid
  hyperparam_grid = config['Grid']
  hyperparam_combinations = list(itertools.product(*hyperparam_grid.values()))

  # Temporarily save the model and scaler to use during testing
  config['Model']['save_path'] = config['Model']['save_path'].split('.pkl')[0] + '_tmp.pkl'

  # Create a directory to store results
  results_dir = 'tuning_results/'
  os.makedirs(results_dir, exist_ok=True)

  # Open a CSV file to store the results
  results_file = os.path.join(results_dir, config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_tuning.csv')
  with open(results_file, 'w', newline='') as csvfile:
    fieldnames = list(hyperparam_grid.keys()) +['Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_slope', 'Test_int']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over hyperparameter combinations
    for hyperparam_values in hyperparam_combinations:
      print('\n'.join([f'Training with hyperparameters ' + ', '.join([f'{key}: {value}' for key, value in zip(hyperparam_grid.keys(), hyperparam_values)])]))

      # Update the hyperparameters in the config
      config = update_config(hyperparam_grid.keys(), hyperparam_values)
    
      # Save the updated config to a temporary file
      temp_config_path = os.path.join(results_dir, 'temp_config.yaml')
      temp_config_path = save_updated_config(config, temp_config_path)

      # Call train.py with the updated config
      train_cmd = ['python', 'train.py', temp_config_path]
      process = run(train_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

      # Call test.py (to test on validation set) with the updated config
      test_cmd = ['python', 'test.py', temp_config_path]
      process = run(test_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

      # Extract the test score from the output
      error_lines = process.stderr.split('\n')
      output_lines = process.stdout.split('\n')
      test_rmse_line = [line for line in output_lines if 'Test RMSE' in line]
      test_rmse = float(test_rmse_line[0].split(': ')[1]) if test_rmse_line else None
      test_mae_line = [line for line in output_lines if 'Test MAE' in line]
      test_mae = float(test_mae_line[0].split(': ')[1]) if test_mae_line else None
      test_r2_line = [line for line in output_lines if 'Test R2' in line]
      test_r2 = float(test_r2_line[0].split(': ')[1]) if test_r2_line else None
      test_slope_line = [line for line in output_lines if 'Test slope' in line]
      test_slope = float(test_slope_line[0].split(': ')[1]) if test_slope_line else None
      test_int_line = [line for line in output_lines if 'Test intercept' in line]
      test_int = float(test_int_line[0].split(': ')[1]) if test_int_line else None

      # Write the results to the CSV file
      row_dict = {h: hyperparam_values[i] for i, h in enumerate(hyperparam_grid.keys())}
      row_dict['Test_RMSE'] = test_rmse
      row_dict['Test_MAE'] = test_mae
      row_dict['Test_R2'] = test_r2
      row_dict['Test_slope'] = test_slope
      row_dict['Test_int'] = test_int
      writer.writerow(row_dict)

      # Delete the temporary saved model and scaler
      print(config['Model']['save_path'])
      os.remove(config['Model']['save_path'])
      os.remove(config['Model']['save_path'].split('.')[0] + '_scaler.pkl')

  
  # Delete the temporary config file
  os.remove(temp_config_path)

  print(f'Tuning results saved in {results_file}')
  return


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('tune_settings', type = str, metavar='path/to/setting.yaml', help='yaml with settings for hyperparam tuning')
  args = parser.parse_args()

  config = load_config(args.tune_settings)
  tune_model(config)
