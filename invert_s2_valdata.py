'''
Inversion of Sentinel-2 data for crop trait retrieval using
PROSAIL lookup tables

@author Lukas Valentin Graf, Selene Ledain
'''

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

import sys
import os
from pathlib import Path
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, os.path.join(base_dir, "eodal"))
from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import Dict, List, Optional
from rtm_inv.core.inversion import inv_df, retrieve_traits

logger = get_settings().logger
warnings.filterwarnings('ignore')

band_selection = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


def invert_scenes(
    data_path: str,
    lut_paths: List[str],
    n_solutions: Dict[str, int],
    cost_functions: Dict[str, str],
    aggregation_methods: Dict[str, str],
    traits: Optional[List[str]] = ['lai', 'ccc']
):
    """
    Lookup table based inversion of S2 imagery. The inversion setup can
    be adopted for each phenological macro-stage.

    :param data_path:
        path to where data to invert is plocated. Should be a .pkl file with a gpd 
    :param lut_paths:
        list of LUTs to use for inversion
    :param n_solutions:
        number of solutions of the inversion to use per phenological
        macro-stage
    :param cost_functions:
        cost function per phenological macro-stage
    :param aggregation_methods:
        aggregation methods of the solutions found per phenological macro-stage
    :param traits:
        list of traits to extract (this is used to find the correct LUT file).
        Defaults to 'lai', 'cab', and 'ccc'.
    """

    # Load the LUTs
    lut = pd.concat([pd.read_pickle(path) for path in lut_paths], axis=0, ignore_index=True)
    s2_lut = lut[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]

    # Load the data to invert
    df_data = pd.read_pickle(data_path).reset_index(drop=True)

    # Perform inversion
    lut_idxs, cost_function_values = inv_df(
        lut=s2_lut,
        df=df_data,
        cost_function=cost_functions,
        n_solutions=n_solutions,
    )

    trait_img, q05_img, q95_img = retrieve_traits(
        lut=lut,
        lut_idxs=lut_idxs,
        traits=traits,
        cost_function_values=cost_function_values,
        measure=aggregation_methods
    )

    # Save traits to the dataframe 
    for i, trait in enumerate(traits):
        df_data[f'{trait}_retrieved'] = trait_img[:, i]
        df_data[f'{trait}_q05'] = q05_img[:, i]
        df_data[f'{trait}_q95'] = q95_img[:, i]

    # Save lowest, median and highest cost function value
    highest_cost_function_vals = [np.max(c) for c in cost_function_values]
    lowest_cost_function_vals = [np.min(c) for c in cost_function_values]
    median_cost_function_vals = [np.median(c) for c in cost_function_values]
    df_data['cost_func_max'] = highest_cost_function_vals
    df_data['cost_func_min'] = lowest_cost_function_vals
    df_data['cost_func_median'] = median_cost_function_vals


    # Save to pickle
    fname = data_path.split('.pkl')[0] + '_retrievednoise.pkl'
    with open(fname, 'wb') as f:
        df_data.to_pickle(f)

    return



if __name__ == '__main__':

    data_path = '../results/validation_data_extended_lai.pkl'
    lut_dir = Path('../results/lut_based_inversion/soil_scaled')
    lut_paths = [lut_dir.joinpath('prosail_danner-etal_switzerland_lai-cab-ccc-car_lut_no-constraints_inverse10.pkl')] #[lut_dir.joinpath('prosail_danner-etal_switzerland_S2A_lai-cab-ccc-car_lut_no-constraints.pkl'), lut_dir.joinpath('prosail_danner-etal_switzerland_S2B_lai-cab-ccc-car_lut_no-constraints.pkl'),]
    

    cost_functions = 'mae'
    aggregation_methods ='median'
    n_solutions = 5000
    traits = ['lai']

    invert_scenes(
        data_path,
        lut_paths,
        n_solutions=n_solutions,
        cost_functions=cost_functions,
        aggregation_methods=aggregation_methods,
        traits=traits
    )
