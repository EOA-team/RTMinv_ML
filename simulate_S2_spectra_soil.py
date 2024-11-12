"""
Generate PROSAIL RTM simulations with soil spectra and a corresponding look up table.

@author Selene Ledain
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from rtm_inv.core.lookup_table import LookupTable, generate_lut, simulate_from_lut
import pickle
import numpy as np
import pandas as pd


def get_logger():
  """
  Returns a logger object with stream and file handler
  """
  
  CURRENT_TIME: str = datetime.now().strftime("%Y%m%d-%H%M%S")
  LOGGER_NAME: str = "CropCovEO"
  LOG_FORMAT: str = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
  LOG_DIR: str = str(Path.home())  # ..versionadd:: 0.2.1
  LOG_FILE: str = os.path.join(LOG_DIR, f"{CURRENT_TIME}_{LOGGER_NAME}.log")
  LOGGING_LEVEL: int = logging.INFO

  # create file handler which logs even debug messages
  logger = logging.getLogger(LOGGER_NAME)
  logger.setLevel(LOGGING_LEVEL)
  
  fh: logging.FileHandler = logging.FileHandler(LOG_FILE)
  fh.setLevel(LOGGING_LEVEL)
  # create console handler with a higher log level
  ch: logging.StreamHandler = logging.StreamHandler()
  ch.setLevel(LOGGING_LEVEL)
  # create formatter and add it to the handlers
  formatter: logging.Formatter = logging.Formatter(LOG_FORMAT)
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  # add the handlers to the logger
  logger.addHandler(fh)
  logger.addHandler(ch)

  return logger


def generate_spectra(
    output_dir: Path,
    lut_params_dir: Path,
    lut_config: Dict[str, Any],
    rtm_config: Dict[str, Any],
    traits: List[str]
  ) -> None:

  logger = get_logger()
  
  # run PROSAIL forward runs for the different parametrizations available
  logger.info('Starting PROSAIL runs')
  lut_params_pheno = lut_params_dir.joinpath('prosail_danner-etal_switzerland_nosoil.csv') # Only use general params for LUT

  pheno_phases = \
      lut_params_pheno.name.split('.csv')[0] +'_S2A'
      #lut_params_pheno.name.split('etal')[-1].split('.')[0][1::]

  # generate lookup-table
  trait_str = '-'.join(traits)
  fpath_lut = output_dir.joinpath(
    f'{pheno_phases}_{trait_str}_lut_no-constraints.pkl') 

  # if LUT exists, continue, else generate it
  if not fpath_lut.exists():
    lut_inp = lut_config.copy()
    lut_inp['lut_params'] = lut_params_pheno
    lut = generate_lut(**lut_inp)
    lut = simulate_from_lut(lut, **rtm_config)

    # special case CCC (Canopy Chlorophyll Content) ->
    # this is not a direct RTM output
    if 'ccc' in traits:
      lut['ccc'] = lut['lai'] * lut['cab']
      # convert to g m-2 as this is the more common unit
      # ug -> g: factor 1e-6; cm2 -> m2: factor 1e-4
      lut['ccc'] *= 1e-2


    lut.dropna(inplace=True)

    # save LUT to file
    with open(fpath_lut, 'wb') as f:
      pickle.dump(lut, f, protocol=3)

  else:
    pass

  logger.info('Finished PROSAIL runs')


def generate_spectra_soil(
  output_dir: Path,
  lut_params_dir: Path,
  lut_config: Dict[str, Any],
  rtm_config: Dict[str, Any],
  traits: List[str],
  soil_df: pd.DataFrame
  ) -> None:

  logger = get_logger()
  
  # run PROSAIL forward runs for the different parametrizations available
  logger.info('Starting PROSAIL runs')
  lut_params_pheno = lut_params_dir.joinpath('prosail_danner-etal_switzerland_soil.csv') # Only use general params for LUT

  pheno_phases = \
      lut_params_pheno.name.split('.csv')[0] +'_S2B'
      #lut_params_pheno.name.split('etal')[-1].split('.')[0][1::]

  # generate lookup-table
  trait_str = '-'.join(traits)
  fpath_lut = output_dir.joinpath(
    f'{pheno_phases}_{trait_str}_lut_constratints.pkl') #

  # if LUT exists, continue, else generate it
  if not fpath_lut.exists():
    # Generate LUT
    lut_inp = lut_config.copy()
    lut_inp['lut_params'] = lut_params_pheno
    lut = generate_lut(**lut_inp)
    print(lut._samples)
    return
    # Simulate with RTM
    rtm_inp = rtm_config.copy()
    # Create LUT subgroups of size lut_size/len(soil_df). 
    # Pass each subgroup with a soil spectra and simulate.
    # Concatenate all simulations
    sub_luts = get_random_subgroups(lut._samples, len(soil_df))
    lut_allsoils = []
    for i, (original_idx, sub_lut) in enumerate(sub_luts):
      logger.info(f'Simulating with soil spectra {i+1} of {len(soil_df)}')
      rtm_inp['rsoil0'] = None
      rtm_inp['soil_spectrum1'] = soil_df.iloc[i].values 
      rtm_inp['soil_spectrum2'] = np.zeros_like(soil_df.iloc[i].values) 
      sub_lut = dataframe_to_lookup_table(sub_lut, lut)
      lut_soilspectra = simulate_from_lut(sub_lut, **rtm_config)
      lut_soilspectra.index = original_idx # keep original order
      lut_allsoils.append(lut_soilspectra)
    
    lut = pd.concat(lut_allsoils).sort_index()


    # special case CCC (Canopy Chlorophyll Content) ->
    # this is not a direct RTM output
    if 'ccc' in traits:
      lut['ccc'] = lut['lai'] * lut['cab']
      # convert to g m-2 as this is the more common unit
      # ug -> g: factor 1e-6; cm2 -> m2: factor 1e-4
      lut['ccc'] *= 1e-2

    # prepare LUT for model training
    # lut = lut[band_selection + traits].copy()
    lut.dropna(inplace=True)

 
    # save LUT to file
    with open(fpath_lut, 'wb+') as f:
      pickle.dump(lut, f)

  else:
    pass

  logger.info('Finished PROSAIL runs')



def get_random_subgroups(df, n):
    # Step 1: Shuffle the indices of the DataFrame
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    # Step 2: Split the shuffled indices into `n` subgroups of roughly equal size
    subgroups = np.array_split(indices, n)
    
    # Step 3: Create subgroups of the DataFrame using these indices
    subgroups_dfs = [(indices, df.iloc[indices]) for indices in subgroups]
    
    return subgroups_dfs

def dataframe_to_lookup_table(df: pd.DataFrame, original_lut: LookupTable) -> LookupTable:
    """
    Converts a DataFrame to a LookupTable object.

    :param df: DataFrame to convert
    :param original_lut: Original LookupTable object to copy parameters from
    :return: LookupTable object
    """
    lut = LookupTable(original_lut._params_df)
    lut.samples = df.reset_index(drop=True).copy()
    return lut



if __name__ == '__main__':

  #logger = get_logger()
  #logger.info('Testing logger')

  cwd = Path(__file__).parent.absolute()
  import os
  os.chdir(cwd)

  # global setup
  out_dir = Path('results').joinpath('lut_based_inversion/test/')
  out_dir.mkdir(exist_ok=True)

  # spectral response function of Sentinel-2 for resampling PROSAIL output
  fpath_srf = Path(
      'data/auxiliary/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx')
  # Configurations for lookup-table generation and RTM simulations
  lut_config = {
      'lut_size': 10000,
      'sampling_method': 'FRS',
      'apply_glai_ccc_constraint': False,
      'apply_chlorophyll_carotiniod_constraint': False
  }
  rtm_config = {
    'sensor': 'Sentinel2B',
    'fpath_srf': fpath_srf,
    'remove_invalid_green_peaks': True,
    'linearize_lai': False,
  }

  # directory with LUT parameters for different phenological macro-stages
  lut_params_dir = Path('lut_params')
  # Path to soil spectra to use
  soil_path = Path('results/GEE_baresoil_v2/sampled_spectra_all_CH.pkl')

  # target trait(s)
  traits = ['lai', 'cab', 'ccc', 'car']

  #######################################################################

  if soil_path is None:
    # Call RTM and generate LUT
    try:
        generate_spectra(
            output_dir=out_dir,
            lut_params_dir=lut_params_dir,
            lut_config=lut_config,
            rtm_config=rtm_config,
            traits=traits
        )

    except Exception as e:
        print(f'Error: {e}')
        pass


  if soil_path is not None:
    # Loop over soil spectra
    soil_df = pd.read_pickle(soil_path)
    #n_samples_per_spectra = rtm_lut_config['lut_size']//len(soil_df) # number of samples to generate with one soil spectra

    try:
        generate_spectra_soil(
            output_dir=out_dir,
            lut_params_dir=lut_params_dir,
            lut_config=lut_config,
            rtm_config=rtm_config,
            traits=traits,
            soil_df=soil_df
        )
    except Exception as e:
        print(f'Error: {e}')
        pass

