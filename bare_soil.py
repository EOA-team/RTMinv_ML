'''
Extract bare soil spectra from S2 data

@Selene Ledain
'''
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
from typing import List
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, os.path.join(base_dir, "eodal"))
import eodal

from datetime import datetime
from eodal.config import get_settings
from eodal.core.scene import SceneCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

Settings = get_settings()
# set to False to use a local data archive
Settings.USE_STAC = True


def preprocess_sentinel2_scenes(
    ds: Sentinel2,
    target_resolution: int,
    ) -> Sentinel2:
    """
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
    based on the Scene Classification Layer (SCL).

    NOTE:
        Depending on your needs, the pre-processing function can be
        fully customized using the full power of EOdal and its
    interfacing libraries!

    :param target_resolution:
        spatial target resolution to resample all bands to.
    :returns:
        resampled, cloud-masked Sentinel-2 scene.
    """
    # resample scene
    ds.resample(inplace=True, target_resolution=target_resolution)
    # mask clouds, shadows, and snow
    ds.mask_clouds_and_shadows(inplace=True, cloud_classes=[3, 8, 9, 10, 11])
    return ds


def extract_s2_data(
        aoi: gpd.GeoDataFrame,
        time_start: datetime,
        time_end: datetime,
        scene_cloud_cover_threshold: float = 50,
        feature_cloud_cover_threshold: float = 80,
        spatial_resolution: int = 10
    ) -> SceneCollection:
    """
    Extracts Sentinel-2 data from the STAC SAT archive for a given area and time period.
    Scenes that are too cloudy or contain nodata (blackfill), only, are discarded.
    Keep only bare soil pixels

    The processing level of the Sentinel-2 data is L2A (surface reflectance factors).

    :param parcel:
        field parcel geometry (defines the spatial extent to extract)
    :param time_start:
        start of the time period to extract
    :param end_time:
        end of the time period to extract
    :param scene_cloud_cover_threshold:
        scene-wide cloudy pixel percentage (from Sentinel-2 metadata) to filter out scenes
        with too high cloud coverage values [0-100%]
    :param feature_cloud_cover_threshold:
        cloudy pixel percentage [0-100%] on the parcel level. Only if the parcel has a
        lower percentual share of cloudy pixels (based on the scene classification layer) than
        the threshold specified, the Sentinel-2 observation is kept
    :param spatial_resolution:
        spatial resolution of the Sentinel-2 data in meters (Def: 10m)
    :param resampling_method:
        spatial resampling method for those Sentinel-2 bands not available in the target
        resolution. Nearest Neighbor by default
    :returns:
        dictionary with the list of scenes for the field parcel (`feature_scenes`), the
        DataFrame of (un)used scenes and the reason for not using plus some basic scene
        metadata (`scene_properties`)
    """
    # setup the metadata filters (cloud cover and processing level)
    metadata_filters = [
        Filter('cloudy_pixel_percentage','<', scene_cloud_cover_threshold),
        Filter('processing_level', '==', 'Level-2A')
    ]
    # setup the spatial feature for extracting data
    feature = Feature.from_geoseries(aoi.geometry)
    
    # set up mapping configs
    mapper_configs = MapperConfigs(
        collection='sentinel2-msi',
        time_start=time_start,
        time_end=time_end,
        feature=feature,
        metadata_filters=metadata_filters
    )

    # get a new mapper instance. Set sensor to `sentinel2`
    mapper = Mapper(mapper_configs)

    # query the STAC (looks for available scenes in the selected spatio-temporal range)
    mapper.query_scenes()

    # get observations (loads the actual Sentinel2 scenes)
    # the data is extract for the extent of the parcel
    scene_kwargs = {
        'scene_constructor': Sentinel2.from_safe,            # this tells the mapper how to read and load the data (i.e., Sentinel-2 scenes)
        'scene_constructor_kwargs': {},                      # here you could specify which bands to read
        'scene_modifier': preprocess_sentinel2_scenes,       # this tells the mapper about (optional) pre-processing of the loaded scenes (must be a callable)
        'scene_modifier_kwargs': {'target_resolution': 10}   # here, you have to specify the value of the arguments the `scene_modifier` requires
    }
    mapper.load_scenes(scene_kwargs=scene_kwargs)

    # loop over available Sentinel-2 scenes stored in mapper.data as a EOdal SceneCollection and check
    # for no-data. These scenes will be removed from the SceneCollection
    scenes_to_del = []
    mapper.metadata['scene_used'] = 'yes'

    if mapper.data is not None:
        for scene_id, scene in mapper.data:

            # check if scene is blackfilled (nodata); if yes continue
            if scene.is_blackfilled:
                scenes_to_del.append(scene_id)
                mapper.metadata.loc[mapper.metadata.sensing_time.dt.strftime('%Y-%m-%d %H:%M') == scene_id.strftime('%Y-%m-%d %H:%M')[0:16], 'scene_used'] = 'No [blackfill]'
                continue

            # check cloud coverage (including shadows and snow) of the field parcel
            feature_cloud_cover = scene.get_cloudy_pixel_percentage(cloud_classes=[3, 8, 9, 10, 11])

            # if the scene is too cloudy, we skip it
            if feature_cloud_cover > feature_cloud_cover_threshold:
                scenes_to_del.append(scene_id)
                mapper.metadata.loc[mapper.metadata.sensing_time.dt.strftime('%Y-%m-%d %H:%M') == scene_id.strftime('%Y-%m-%d %H:%M')[0:16], 'scene_used'] = 'No [clouds]'
                continue

            # calculate the NDVI and NBR2
            scene.calc_si('NDVI', inplace=True)
            scene.calc_si('NBR2', inplace=True)

            # Check if there are any bare soil pixels
            ndvi = scene.get_band('NDVI').values
            nbr2 =  scene.get_band('NBR2').values
            ndvi.fill_value = np.nan
            nbr2.fill_value = np.nan
            bare_condition = (0 < ndvi) & (ndvi <= 0.25) & (nbr2 <= 0.075)
            bare_soil = np.ma.masked_array(ndvi.data, mask=~bare_condition)
            if not np.sum(~bare_soil.mask): # all pixels are masked -> no bare soil
                scenes_to_del.append(scene_id)
            else:
                #print(scene_id, np.sum(~bare_soil.mask))
                scene.mask(bare_soil.mask, keep_mask_values=True, inplace=True)
                # Save the number of bare soil pixels
                mapper.metadata.loc[mapper.metadata['sensing_date'] == scene_id.date(), 'n_bare'] = np.sum(~bare_soil.mask)
            
                
        # delete scenes too cloudy or containing only no-data or with no bare soil pixels
        for scene_id in scenes_to_del:
            del mapper.data[scene_id]
        # Keep only metadata for corresponding scenes
        dates_to_del = [scene_id.date() for scene_id in scenes_to_del]
        mapper.metadata = mapper.metadata[~mapper.metadata['sensing_date'].isin(dates_to_del)]
    
    return mapper


def compute_scoll_percentiles(scoll):
  ''' 
  Find the 1st and 99th percentile for each band in the scene collection

  :params scolle: EOdal scene collection object
  '''
  lower_threshold = {}
  upper_threshold = {}
  bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12']

  for scene_id, scene in scoll:
    df_scene = scene.to_dataframe()
    # Compute 1st and 99th percentiles for each band across whole scoll
    for band in bands:
        if band not in lower_threshold:
            lower_threshold.update({band: float('inf')})  # Initialize to positive infinity
        if band not in upper_threshold:
            upper_threshold.update({band: float('-inf')})  # Initialize to negative infinity

        band_values = df_scene[band].dropna().values  # Drop NaN values for the band
        if len(band_values) > 0:
            band_lower = np.percentile(band_values, 1)
            band_upper = np.percentile(band_values, 99)

            # Update lower_threshold and upper_threshold for each band
            lower_threshold[band] = min(lower_threshold[band], band_lower)
            upper_threshold[band] = max(upper_threshold[band], band_upper)

  return lower_threshold, upper_threshold


def filter_dataframe(df, lower_threshold, upper_threshold, bands):
    '''
    Remove pixels where a band is below/above the 1st/99th percentile (across the whole collection of scenes)

    :param df: dataframe to filter
    :param lower_threshold: dictionary with lower thresh values for each band
    :param upper_threshold: dictionary with upper thresh values for each band
    :param bands: columns in df that correspond to bands to filter
    '''
    for band in bands:
        lower_thr = lower_threshold[band]
        upper_thr = upper_threshold[band]
        # Set values outside the threshold to np.nan
        df[band] = np.where((df[band] < lower_thr) | (df[band] > upper_thr), np.nan, df[band])
    
    df = df.dropna()
    return df


def sample_bare_pixels(scoll, metadata, lower_threshold, upper_threshold):
  '''
  Sample bare pixels from scenes based on k-means clustering.
  Use the top 3 scenes where there were the most number of bare pixels (clearest days with bare pixels)

  :param scoll: EOdal scene collection
  :param metadta: metadata dataframe
  :param lower_threshold: dictionary with lower thresh values for each band
  :param upper_threshol: dictionary with upper thresh values for each band

  :returns: dataframe with sampled pixels
  '''

  bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12']
  samples = []

  for i, row in metadata.sort_values(by='n_bare', ascending=False).head(3).iterrows():
    date = row['sensing_date']
    sensor = row['spacecraft_name']

    # Get scene for that date
    scoll_idx = [i for i, _ in scoll if i.date()==date][0]
    scene = scoll.__getitem__(scoll_idx) # get scene collection

    # Get pixels in scene
    pixs = scene.to_dataframe()

    # Remove top and botthom 1% for each band 
    pixs = filter_dataframe(pixs, lower_threshold, upper_threshold, bands)

    if len(pixs):
      # Add some useful metadata: sensing date, sensor
      pixs['sensing_date'] = [date]*len(pixs)
      pixs['sensor'] = [sensor]*len(pixs)

      """ 
      # Get the pixels with min, max and median SWI (soil moisture index)
      pixs['SMI'] = (pixs.B08 - pixs.B11)*(pixs.B08 / pixs.B11)
      min_swi_row = pixs.loc[pixs['SMI'].idxmin()]
      max_swi_row = pixs.loc[pixs['SMI'].idxmax()]
      # Calculate median SWI and find the closest row
      median_swi = pixs['SMI'].median()
      closest_to_median_row = pixs.iloc[(pixs['SMI'] - median_swi).abs().argsort()[:1]].iloc[0]

      # Append the sampled rows to the list
      samples.extend([min_swi_row, max_swi_row, closest_to_median_row])
      """

      # Apply k-means clustering on the pixel data
      num_clusters = 5
      kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
      pixs['cluster'] = kmeans.fit_predict(pixs[bands])

      # Sample one pixel from each cluster
      for cluster_id in range(num_clusters):
          cluster_pixs = pixs[pixs['cluster'] == cluster_id]
          if not cluster_pixs.empty:
              # Sample one pixel from this cluster
              sampled_row = cluster_pixs.sample(1).iloc[0]
              samples.append(sampled_row)

  # Convert GeoDataFrames to DataFrames before creating the final DataFrame
  df_sampled = pd.DataFrame([gdf.to_dict() for gdf in samples])

  return df_sampled


def upsample_spectra(df, wavelengths, new_wavelengths):
    '''
    Upsample spectra from a sensor

    :param df: dataframe containing pixel spectra all oroginaitng from one sensor
    :param wavelengths: sensor wavelengths
    :param new_wavelengths: wavelengths to upsample to

    :returns: same dataframe but upsampled to new_wavelengths
    '''
    # Interpolate along the rows (pixels) for each wavelength
    interpolated_data = {}
    for index, row in df.iterrows():
        row['2500'] = row.min() # bound the interpolation so that function doesnt explode
        f = interp1d(wavelengths + [2500], row, kind='cubic', fill_value="extrapolate")
        interpolated_data[index] = f(new_wavelengths)

    # Create a new DataFrame with the interpolated data
    interpolated_df = pd.DataFrame(interpolated_data, index=new_wavelengths)

    # Transpose the DataFrame to have pixels as columns
    interpolated_df = interpolated_df.T

    # Print the result
    return interpolated_df


def resample_df(df_sampled, s2a, s2b):
  '''
  Resample all sampled pixels from Sentinel-2A and Sentinel-2B
  
  :param df_sampled:
  :param s2a: wavelengths for Sentinel-2A
  :param s2b: wavelengths for Sentinel-2B
  :param new_wavelengths: wavelegnths to upsample to
  '''

  s2a_df = df_sampled[df_sampled['sensor'] == 'Sentinel-2A']
  s2b_df = df_sampled[df_sampled['sensor'] == 'Sentinel-2B']
  df_list = [s2a_df, s2b_df]

  spectra = pd.DataFrame()

  for df in df_list:
    if len(df):
      s2_vals = df[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]

      # Resample to 1nm
      if np.unique(df.sensor) == 'Sentinel-2A':
        df_new = upsample_spectra(s2_vals, s2a, new_wavelengths)
      if np.unique(df.sensor) == 'Sentinel-2B':
        df_new = upsample_spectra(s2_vals, s2b, new_wavelengths)
      
      # Append
      spectra = pd.concat([spectra, df_new])
  
  return spectra




if __name__ == '__main__':

    # Set up parameters
    locations = ['Strickhof.shp', 'SwissFutureFarm.shp', 'Witzwil.shp']
    date_start = '2017-03-01'
    date_end = '2023-12-31'

    s2a = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4]
    s2b = [492.1, 559.0, 664.9, 703.8, 739.1, 779.7, 832.9, 864.0, 1610.4, 2185.7]
    new_wavelengths = np.arange(400, 2501, 1)

    soil_spectra = pd.DataFrame()
    
    for i, loc in enumerate(locations):

        print(f'Sampling bare soil from {loc}...')

        # Setup parameters
        shp_path = base_dir.joinpath(f'data/{loc}')
        path_suffix = loc.split('.')[0]
        save_path = base_dir.joinpath(f'results/eodal_baresoil_s2_data_{path_suffix}.pkl')
        metadata_path = save_path.with_name(save_path.name.replace('data', 'metadata'))

        # Add buffer to remove field edges
        aoi = gpd.read_file(shp_path).dissolve()
        aoi = aoi.to_crs('EPSG:2056')
        aoi_with_buffer = aoi.copy()
        buffer_distance = -20 # in meters
        aoi_with_buffer['geometry'] = aoi.buffer(buffer_distance)
        aoi_with_buffer = aoi_with_buffer.to_crs('EPSG:4326')

        # Get data if not done yet
        if not save_path.exists() or not metadata_path.exists():
            res_baresoil = extract_s2_data(
            aoi=aoi_with_buffer,
            time_start=datetime.strptime(date_start, '%Y-%m-%d'),
            time_end=datetime.strptime(date_end, '%Y-%m-%d')
            )

            # Save data for future use
            with open(save_path, 'wb+') as dst:
                dst.write(res_baresoil.data.to_pickle())     
            with open(metadata_path, 'wb+') as dst:
                pickle.dump(res_baresoil.metadata, dst)
        

        # Load data    
        scoll = SceneCollection.from_pickle(stream=save_path)
        metadata = pd.read_pickle(metadata_path)

        # Compute band percentile thresholds in scene collection (for filtering)
        lower_threshold, upper_threshold = compute_scoll_percentiles(scoll)

        # Sample pixels: 5 pixs per date, for top3 dates with most bare pixels
        df_sampled = sample_bare_pixels(scoll, metadata, lower_threshold, upper_threshold)

        # Resample to 1nm 
        spectra = resample_df(df_sampled, s2a, s2b)

        spectra_path = base_dir.joinpath(f'results/sampled_spectra_{path_suffix}.pkl')
        with open(spectra_path, 'wb+') as dst:
            pickle.dump(spectra, dst)

        soil_spectra = pd.concat([soil_spectra, spectra])
        print('Done.')
    
    print('Saving all samples...')
    spectra_path = base_dir.joinpath(f'results/sampled_spectra_all.pkl')
    with open(spectra_path, 'wb+') as dst:
        pickle.dump(soil_spectra, dst)
    print('Finished.')
    


        