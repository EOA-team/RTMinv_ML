'''
Extract bare soil spectra from S2 data

@Selene Ledain
'''
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
from typing import List
from shapely.geometry import Polygon, MultiPolygon, Point
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d, pchip_interpolate
import matplotlib.pyplot as plt
import time
import glob
import contextily as ctx
import calendar
import rasterio

import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent


def load_raster_gdf(tif_path: Path):
    """Load raster data into a GeoDataFrame.
    
    :param tif_path: Path to the GeoTIFF raster file
    """

    # Read the raster data using rasterio
    with rasterio.open(tif_path) as src:
        # Read raster data as a NumPy array
        raster_array = src.read()  # Read all bands

        # Get the metadata for coordinate reference system (CRS) and transform
        crs = src.crs.to_string()

    # Create a grid of points representing each pixel
    rows, cols = raster_array.shape[1:]
    x_coords, y_coords = np.meshgrid(np.linspace(src.bounds.left, src.bounds.right, cols),
                                    np.linspace(src.bounds.top, src.bounds.bottom, rows))

    # Flatten the coordinates
    flat_x_coords = x_coords.flatten()
    flat_y_coords = y_coords.flatten()

    # Create a GeoDataFrame with points
    geometry = [Point(x, y) for x, y in zip(flat_x_coords, flat_y_coords)]
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)

    # Add columns for each band value
    col_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'ndvi', 'nbr2']
    for i in range(len(col_names)):  
        band_values = raster_array[i]
        gdf[col_names[i]] = band_values.flatten()
   
    gdf.dropna(inplace=True)

    # Bands are between 0 - 10000, need to be normalised
    for band in col_names[:-2]:
        gdf[band] = gdf[band] / 10000.0

    return gdf


def compute_gdf_percentiles(gdf, bands):
    ''' 
    Find the 10th and 90th percentile for each band in the geodataframe

    :param gdf: Geodataframe with Sentinel-2 reflectance values
    :params bands: list of bands to compute percentiles for
    '''
    lower_threshold = {}
    upper_threshold = {}

    for band in bands:
        if band not in lower_threshold:
            lower_threshold.update({band: float('inf')})  # Initialize to positive infinity
        if band not in upper_threshold:
            upper_threshold.update({band: float('-inf')})  # Initialize to negative infinity

        band_values = gdf[band].dropna().values  # Drop NaN values for the band
        if len(band_values) > 0:
            band_lower = np.percentile(band_values, 10)
            band_upper = np.percentile(band_values, 90)

            # Update lower_threshold and upper_threshold for each band
            lower_threshold[band] = min(lower_threshold[band], band_lower)
            upper_threshold[band] = max(upper_threshold[band], band_upper)

    return lower_threshold, upper_threshold


def filter_gdf(gdf):
    '''
    Remove pixels where a band is below/above the 10th/90th percentile for that raster (in gdf format)

    :param gdf: Geodataframe with Sentinel-2 reflectance values to filter
    '''

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    # Find the 10th and 90th percentiles per band
    lower_threshold, upper_threshold = compute_gdf_percentiles(gdf, bands)

    for band in bands:
        lower_thr = lower_threshold[band]
        upper_thr = upper_threshold[band]
        # Filter values outside the threshold
        gdf = gdf[(gdf[band] >= lower_thr) & (gdf[band] <= upper_thr)]

    return gdf


def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    print('Seaching for optimal clusters')
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(data)
        if i > 2:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Find the optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 due to starting from 2 clusters

    """
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    """

    return optimal_clusters


def sample_bare_pixels(pixs_df, samples_per_cluster):
  '''
  Use the top num_scenes scenes where there were the most number of bare pixels (clearest days with bare pixels.
  Gather all pixels and perform k-means clustering, then sample samples_per_cluster from each cluster.

  :param pixs_df: bare soil composite of tile in GeoDataFrame
  :param samples_per_cluster: number of samples to get from each cluster

  :returns: dataframe with sampled pixels
  '''

  bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

  if len(pixs_df):
    # Apply k-means clustering on the pixel dataset
    #optimal_clusters = find_optimal_clusters(pixs_df[bands], max_clusters=10)
    optimal_clusters = 5
    num_clusters = min(optimal_clusters, len(pixs_df))
    print(f'Sampling from {num_clusters} clusters')
    if len(pixs_df)<=num_clusters:
        print('Warning: all pixels will be sampled as there are not enough samples for k-means clustering')
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    pixs_df['cluster'] = kmeans.fit_predict(pixs_df[bands])

    # Sample one pixel from each cluster
    samples = []
    for cluster_id in range(num_clusters):
        cluster_pixs = pixs_df[pixs_df['cluster'] == cluster_id]
        if not cluster_pixs.empty:
            # Sample one pixel from this cluster
            sampled_rows = cluster_pixs.sample(samples_per_cluster) if samples_per_cluster<=len(cluster_pixs) else cluster_pixs
            samples.extend(sampled_rows.to_dict(orient='records'))

    # Convert GeoDataFrames to DataFrames before creating the final DataFrame
    df_sampled = pd.DataFrame(samples)
    return df_sampled

  else:
    print('No bare pixels left to sample.')
    return pd.DataFrame()


def upsample_spectra(df, wavelengths, new_wavelengths, method):
    '''
    Upsample spectra from a sensor

    :param df: dataframe containing pixel spectra all oroginaitng from one sensor
    :param wavelengths: sensor wavelengths
    :param new_wavelengths: wavelengths to upsample to    
    :param method: inteprolation method among ['spline', 'pchip']

    :returns: same dataframe but upsampled to new_wavelengths
    '''
    if method == 'spline':     
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      df.loc[:, '2500'] = df.apply(lambda row: row.min(), axis=1) # Bound the values for the end of the spectra
      f = interp1d([400] + wavelengths + [2500], df.values, kind='cubic', fill_value="extrapolate")
      interpolated_values = f(new_wavelengths)
      interpolated_df = pd.DataFrame(interpolated_values, columns=new_wavelengths, index=df.index)


    if method == 'pchip':
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      interpolated_values = pchip_interpolate([400] + wavelengths, df.values.T, new_wavelengths).T
      interpolated_df = pd.DataFrame(interpolated_values, index=df.index, columns=new_wavelengths)


    if method == 'combined':
      # First part of spectra with spline, second with pchip
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      spline_cols = ['400', 'B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A']
      f = interp1d([400] + wavelengths[:-2], df[spline_cols].values, kind='cubic', fill_value="extrapolate")
      interpolated_values_spline = f(new_wavelengths[:865-400])
      interpolated_df_spline = pd.DataFrame(interpolated_values_spline, columns=new_wavelengths[:865-400], index=df[spline_cols].index)
      
      pchip_cols = ['B10', 'B11', 'B12']
      interpolated_values_pchip = pchip_interpolate(wavelengths[-2:], df[pchip_cols].values.T, new_wavelengths[865-400:]).T
      interpolated_df_pchip = pd.DataFrame(interpolated_values_pchip, index=df[pchip_cols].index, columns=new_wavelengths[865-400:])

      interpolated_df = pd.concat([interpolated_df_spline, interpolated_df_pchip], axis=1)

    return interpolated_df


def resample_df(df_sampled, s2, new_wavelengths, method):
    '''
    Resample all sampled pixels from Sentinel-2A and Sentinel-2B

    :param df_sampled:
    :param s2: wavelengths for Sentinel-2 (generic for 2A and 2B)
    :param new_wavelengths: wavelegnths to upsample to
    :param method: inteprolation method among ['spline', 'pchip']
    '''

    spectra = pd.DataFrame()

    s2_vals = df_sampled[['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]

    # Resample to 1nm
    df_new = upsample_spectra(s2_vals, s2, new_wavelengths, method)

    return df_new


def drop_snow_invalid(gdf, b_thresh, g_thresh, r_thresh):
    '''
    Remove pixels: drop if B02 > b_thresh, B03 > g_thresh, B04 > r_thresh
    Impose B01 < B02

    :param gdf: Geodataframe with Sentinel-2 reflectance values
    :param b_thresh: threshold for B02
    :param g_thresh: threshold for B03
    :param r_thresh: threshold for B04
    '''
    gdf = gdf[(gdf['B02'] < b_thresh) & (gdf['B03'] < g_thresh) & (gdf['B04'] < r_thresh)]
    gdf = gdf[gdf['B01'] <= gdf['B02']]

    return gdf




if __name__ == '__main__':

    #s2a = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7,  1613.7, 2202.4]
    #s2b = [442.2, 492.1, 559.0, 664.9, 703.8, 739.1, 779.7, 832.9, 864.0, 1610.4, 2185.7]
    s2 = [443, 492, 560, 665, 704, 740, 781, 833, 864, 1612, 2194] # don't use B09 for interpolation

    new_wavelengths = np.arange(400, 2501, 1)

    soil_spectra = pd.DataFrame()

    tiles = ['32TMT', '32TLT', '32TNT', '31TGM', '31TGN'] #'32TNS', '32TMS', '32TLS',  
    year_list = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

    upsample_method = 'pchip'

    for tile_id in tiles:
        print(f'Sampling bare soil from tile {tile_id}...')
        
        # Load all tif files for that tile
        tile_BS_paths = glob.glob(str(base_dir.joinpath(f'data/*{tile_id}*.tif')))

        # Put in a geodataframe
        gdfs_tile = [load_raster_gdf(f) for f in tile_BS_paths]
        gdf_tile = pd.concat(gdfs_tile, ignore_index=True)

        # Drop invalid (snowy) or nonsensical data
        gdf_tile = drop_snow_invalid(gdf_tile, b_thresh=0.1, g_thresh=0.4, r_thresh=0.4)

        # Remove top and bottom 10% of data
        gdf_tile = filter_gdf(gdf_tile)
        

        # Sample pixels: 5 pixs per date, for top 6 dates with most bare pixels
        df_sampled = sample_bare_pixels(gdf_tile, samples_per_cluster=5)
        if len(df_sampled):
            sampled_path = base_dir.joinpath(f'results/GEE_baresoil_v2/sampled_pixels_{tile_id}.pkl')
            with open(sampled_path, 'wb+') as dst:
                pickle.dump(df_sampled, dst)

            # Resample to 1nm 
            spectra = resample_df(df_sampled, s2, new_wavelengths, upsample_method)
            spectra_path = base_dir.joinpath(f'results/GEE_baresoil_v2/sampled_spectra_{tile_id}.pkl')
            with open(spectra_path, 'wb+') as dst:
                pickle.dump(spectra, dst)

            soil_spectra = pd.concat([soil_spectra, spectra])
        
        """ 
        sampled_path = base_dir.joinpath(f'results/GEE_baresoil_v2/sampled_pixels_{tile_id}.pkl')
        with open(sampled_path, 'rb+') as dst:
            df_sampled = pickle.load(dst)
        
        # Resample to 1nm 
        spectra = resample_df(df_sampled, s2, new_wavelengths, upsample_method)
        spectra_path = base_dir.joinpath(f'results/GEE_baresoil_v2/sampled_spectra_{tile_id}.pkl')
        with open(spectra_path, 'wb+') as dst:
            pickle.dump(spectra, dst)

        soil_spectra = pd.concat([soil_spectra, spectra])
        """

    print(f'Saving all samples...')
    spectra_path = base_dir.joinpath(f'results/GEE_baresoil_v2/sampled_spectra_all_CH.pkl')
    with open(spectra_path, 'wb+') as dst:
        pickle.dump(soil_spectra, dst)
    print('Finished.')


    


        