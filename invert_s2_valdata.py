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
from typing import Dict, List, Optional
from rtm_inv.core.inversion import inv_df, retrieve_traits
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
from scipy import stats

#logger = get_settings().logger
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

def compute_scores(data_path):
    fname = data_path.split('.pkl')[0] + '_retrievednoise.pkl'
    df = pd.read_pickle(fname)
    df = df[~df['lai'].isna()]
    rmse = mean_squared_error(df.lai, df.lai_retrieved, squared=False)
    r2 = r2_score(df.lai, df.lai_retrieved)
    nrmse = mean_squared_error(df.lai, df.lai_retrieved, squared=False)/(np.max(df.lai) - np.min(df.lai))
    pearson = stats.pearsonr(df.lai, df.lai_retrieved).statistic
    print(f'LUT retrieval: RMSE {rmse}, R2: {r2}, nRMSE: {nrmse}, r2: {pearson**2}')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    ax = df.plot(kind='scatter', x='lai', y='lai_retrieved', figsize=(8,8), s=50)
    ax.set_xlabel('Validation LAI', fontsize=16)
    ax.set_ylabel('Retrieved LAI', fontsize=16)
    # Set axis limits
    ax.set_xlim((0, 8))
    ax.set_ylim((0, 8))
    # Customize ticks
    #ax.tick_params(axis='both', which='major', labelsize=16)
    # Plot the y=x line
    ax.plot([0, 8], [0, 8], color='gray', linestyle='--', label='1:1 fit')
    # Plot the regression line
    slope, intercept = np.polyfit(df.lai, df.lai_retrieved, 1)
    xseq = np.linspace(0, 8, num=100)
    ax.plot(xseq, intercept + slope * xseq, color="r", linestyle='--', label='Linear fit')
    # Add legend
    ax.legend(fontsize=16)
    # Text for displaying on plot
    textstr = f'RMSE: {rmse:.3f}\nnRMSE: {nrmse:.3f}\n$R^2$: {pearson**2:.3f}'
    # Add text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.03, 0.75, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
    # Save plot
    plt.savefig('notebooks/manuscript_figures/lut_retreival_scatter_soilnoise_r2.png')

    return

if __name__ == '__main__':

    data_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/results/validation_data_extended_angles_shift.pkl')
    lut_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/results/lut_based_inversion/soil/')
    lut_paths = [os.path.join(lut_dir, 'prosail_danner-etal_switzerland_soil_lai-cab-ccc-car_lut_no-constraints_multiplicative1.pkl')] #, os.path.join(lut_dir, 'prosail_danner-etal_switzerland_nosoil_S2B_lai-cab-ccc-car_lut_no-constraints.pkl')]

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

    compute_scores(data_path)
