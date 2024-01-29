"""
Generate PROSAIL RTM simulations and a corresponding look up table.

@author Selene Ledain
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from rtm_inv.core.lookup_table import generate_lut
import pickle


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
  rtm_lut_config: Dict[str, Any],
  traits: List[str]
  ) -> None:

  logger = get_logger()
  
  # run PROSAIL forward runs for the different parametrizations available
  logger.info('Starting PROSAIL runs')
  lut_params_pheno = lut_params_dir.joinpath('prosail_danner-etal_switzerland.csv') # Only use general params for LUT

  pheno_phases = \
      lut_params_pheno.name.split('.csv')[0] +'_S2B'
      #lut_params_pheno.name.split('etal')[-1].split('.')[0][1::]

  # generate lookup-table
  trait_str = '-'.join(traits)
  fpath_lut = output_dir.joinpath(
    f'{pheno_phases}_{trait_str}_lut_no-constraints.pkl')

  # if LUT exists, continue, else generate it
  if not fpath_lut.exists():
    lut_inp = rtm_lut_config.copy()
    lut_inp['lut_params'] = lut_params_pheno
    lut = generate_lut(**lut_inp) # TO DO: need to modify function so that it simulates for a range of angles

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
    with open(fpath_lut, 'wb') as f:
      pickle.dump(lut, f, protocol=3)

  else:
    pass

  logger.info('Finished PROSAIL runs')



if __name__ == '__main__':

  #logger = get_logger()
  #logger.info('Testing logger')

  cwd = Path(__file__).parent.absolute()
  import os
  os.chdir(cwd)

  # global setup
  out_dir = Path('../results').joinpath('lut_based_inversion')
  out_dir.mkdir(exist_ok=True)

  # spectral response function of Sentinel-2 for resampling PROSAIL output
  fpath_srf = Path(
      '../data/auxiliary/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx')
  # RTM configurations for lookup-table generation
  rtm_lut_config = {
      'sensor': 'Sentinel2B',
      'lut_size': 50000,
      'fpath_srf': fpath_srf,
      'remove_invalid_green_peaks': True,
      'sampling_method': 'FRS',
      'linearize_lai': False,
      'apply_glai_ccc_constraint': False,
      'apply_chlorophyll_carotiniod_constraint': False
  }
  # directory with LUT parameters for different phenological macro-stages
  lut_params_dir = Path('lut_params')

  # target trait(s)
  traits = ['lai', 'cab', 'ccc', 'car']

  #######################################################################

  # Call RTM and generate LUT
  try:
      generate_spectra(
          output_dir=out_dir,
          lut_params_dir=lut_params_dir,
          rtm_lut_config=rtm_lut_config,
          traits=traits
      )
  except Exception as e:
      print(f'Error: {e}')
      pass