'''
Add noise to training data.
Implement multiple noise models that can be called together with a noise level.

@Selene Ledain
'''

import pandas as pd
import numpy as np
import pickle


def additive(df: pd.DataFrame, noise_level: int):
  '''
  Add gaussian noise to the data
  '''
  # Calculate the standard deviation for each column (band) based on the noise_level
  std_devs = (noise_level / 100.0) * df.apply(np.max)

  # Add Gaussian noise to each element in the DataFrame: reflectance + noise(sigma)
  noisy_df = df + np.random.normal(0, std_devs, df.shape)

  return noisy_df


def multiplicative(df: pd.DataFrame, noise_level: int):
  '''
  Add multiplicate gaussian noise to the data
  '''
  # Calculate the standard deviation for the Gaussian noise based on the noise_level
  std_devs = noise_level / 100.0 * df.apply(np.max)

  # Add noise to each element in the DataFrame: reflectance*(1 + noise(sigma))
  noise = np.random.normal(loc=0, scale=std_devs, size=df.shape)
  noisy_df = df * (1 + noise)

  return noisy_df


def combined(df: pd.DataFrame, noise_level: int):
  '''
  Additive and mutliplicative noise
  '''
  # Calculate the standard deviation for the Gaussian noise based on the noise_level
  std_devs = noise_level / 100.0 *df.apply(np.max)

  # Add noise to each element in the DataFrame: reflectance*(1 + noise(2*sigma)) + noise(sigma)
  
  additive_noise = np.random.normal(loc=0, scale=std_devs, size=df.shape)
  multiplicative_noise = np.random.normal(loc=0, scale=2 * std_devs, size=df.shape)
  noisy_df = df * (1 + multiplicative_noise) + additive_noise

  return noisy_df
  

def inverse(df: pd.DataFrame, noise_level: int):
  '''
  Inverse mutliplicative noise
  '''
  # Calculate the standard deviation for the Gaussian noise based on the noise_level
  std_devs = noise_level / 100.0 * df.apply(np.max)

  # Add noise to each element in the DataFrame: 1 - [(1-reflectance)*(1 + noise(sigma))]
  multiplicative_noise = np.random.normal(loc=0, scale=std_devs, size=df.shape)
  noisy_df = 1 - (1 - df) * (1 + multiplicative_noise)

  return noisy_df


def inverse_combined(df: pd.DataFrame, noise_level: int):
  '''
  Inverse combined mutliplicative noise
  '''
  # Calculate the standard deviation for the Gaussian noise based on the noise_level
  std_devs = noise_level / 100.0 * df.apply(np.max)

  # Add noise to each element in the DataFrame: 1 - [(1-reflectance)*(1 + noise(2*sigma))] + noise(sigma)
  additive_noise = np.random.normal(loc=0, scale=std_devs, size=df.shape)
  multiplicative_noise = np.random.normal(loc=0, scale=2 * std_devs, size=df.shape)
  noisy_df = (1 - (1 - df) * (1 + multiplicative_noise)) + additive_noise

  return noisy_df


def add_noise(df: pd.DataFrame, noise_level: int, noise_type: str) -> pd.DataFrame:
  """ 
  Add noise to the data

  :param df: original data
  :param noise_level: percentege of noise to add
  :param noise_type: type of noise amoong 'additive', 'multiplicative', 'combined', 'inverse', 'inverse_combined'
  """

  if noise_type == 'additive':
    return additive(df, noise_level)
  elif noise_type == 'multiplicative':
    return multiplicative(df, noise_level)
  elif noise_type == 'combined':
    return combined(df, noise_level)
  elif noise_type == 'inverse':
    return inverse(df, noise_level)
  elif noise_type == 'inverse_combined':
    return inverse_combined(df, noise_level)
  else:
    raise Exception('Noise function not yet implemented!')
  return


if __name__ == '__main__':

  """ 
  noise_level = 10
  noise_type = 'inverse_combined'
  cols = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

  # Load data
  data_path = ['../results/lut_based_inversion/prosail_danner-etal_switzerland_S2A_lai-cab-ccc-car_lut_no-constraints.pkl', \
    '../results/lut_based_inversion/prosail_danner-etal_switzerland_S2B_lai-cab-ccc-car_lut_no-constraints.pkl']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
  elif isinstance(data_path, list):
    dfs = [pd.read_pickle(path) for path in data_path]
    df = pd.concat(dfs, axis=0, ignore_index=True)
  
  df_noisy = add_noise(df[cols], noise_level, noise_type)
  # Re add columns that where removed
  df_noisy = pd.concat([df[df.columns.difference(cols)], df_noisy], axis=1)


  # Save noisy data
  with open(f'../results/lut_based_inversion/prosail_danner-etal_switzerland_lai-cab-ccc-car_lut_no-constraints_{noise_type}.pkl', 'wb') as f:
    pickle.dump(df_noisy, f)
  """

  noise_levels = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
  noise_types = ['additive', 'multiplicative', 'combined', 'inverse', 'inverse_combined']
  cols = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B10', 'B11', 'B12']

  # Load data
  data_path = ['../results/lut_based_inversion/soil_scaled/prosail_danner-etal_switzerland_S2A_lai-cab-ccc-car_lut_no-constraints.pkl', \
    '../results/lut_based_inversion/soil_scaled/prosail_danner-etal_switzerland_S2B_lai-cab-ccc-car_lut_no-constraints.pkl']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
  elif isinstance(data_path, list):
    dfs = [pd.read_pickle(path) for path in data_path]
    df = pd.concat(dfs, axis=0, ignore_index=True)

  for noise_type in noise_types:
    for noise_level in noise_levels:
      print(f'Adding noise {noise_level}% {noise_type}')

      df_noisy = add_noise(df[cols], noise_level, noise_type)
      # Re add columns that where removed
      df_noisy = pd.concat([df[df.columns.difference(cols)], df_noisy], axis=1)

      # Save noisy data
      with open(f'../results/lut_based_inversion/soil_scaled/prosail_danner-etal_switzerland_lai-cab-ccc-car_lut_no-constraints_{noise_type}{noise_level}.pkl', 'wb') as f:
        pickle.dump(df_noisy, f)

