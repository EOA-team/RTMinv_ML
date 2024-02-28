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
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d, pchip_interpolate
import matplotlib.pyplot as plt
import time
import glob
import contextily as ctx
import calendar

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
from eodal.utils.reprojection import infer_utm_zone

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
        spatial_resolution: int = 10,
        tile: str = ''
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
    :param tile: Sentinel-2 tile name, to download data only from one tile
    :returns:
        dictionary with the list of scenes for the field parcel (`feature_scenes`), the
        DataFrame of (un)used scenes and the reason for not using plus some basic scene
        metadata (`scene_properties`)
    """
    # setup the metadata filters (cloud cover and processing level)
    metadata_filters = [
        Filter('cloudy_pixel_percentage','<', scene_cloud_cover_threshold),
        Filter('processing_level', '==', 'Level-2A'),
        Filter('tile_id', '==', tile)
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
        'scene_constructor_kwargs': {'band_selection': ['B01','B02','B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B11', 'B12', 'SCL']}, # here you could specify which bands to read
        'scene_modifier': preprocess_sentinel2_scenes,       # this tells the mapper about (optional) pre-processing of the loaded scenes (must be a callable)
        'scene_modifier_kwargs': {'target_resolution': spatial_resolution
        }   # here, you have to specify the value of the arguments the `scene_modifier` requires
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

  :params scoll: EOdal scene collection object
  '''
  lower_threshold = {}
  upper_threshold = {}
  bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

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
        if band == 'B01':
            # don't use B01 for filtering, just for future interpolation
            continue
        else:
            lower_thr = lower_threshold[band]
            upper_thr = upper_threshold[band]
            # Set values outside the threshold to np.nan
            df[band] = np.where((df[band] < lower_thr) | (df[band] > upper_thr), np.nan, df[band])
    
    df = df.dropna()
    return df


def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []

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


def sample_bare_pixels(scoll, metadata, lower_threshold, upper_threshold, num_scenes, samples_per_cluster):
  '''
  Use the top num_scenes scenes where there were the most number of bare pixels (clearest days with bare pixels.
  Gather all pixels and perform k-means clustering, then sample samples_per_cluster from each cluster.

  :param scoll: EOdal scene collection
  :param metadta: metadata dataframe
  :param lower_threshold: dictionary with lower thresh values for each band
  :param upper_threshold: dictionary with upper thresh values for each band
  :param num_scenes: number of scenes to keep in orer of count of bare soil pixels
  :param samples_per_cluster: number of samples to get from each cluster

  :returns: dataframe with sampled pixels
  '''

  bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
  pixs_df = pd.DataFrame()

  for i, row in metadata.sort_values(by='n_bare', ascending=False).head(num_scenes).iterrows():
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
      pixs_df = pd.concat([pixs_df, pixs])


  if len(pixs_df):
    # Apply k-means clustering on the pixel dataset
    optimal_clusters = find_optimal_clusters(pixs_df[bands], max_clusters=10)
    num_clusters = min(optimal_clusters, len(pixs_df))
    print(f'Sampling from {num_clusters} clusters')
    if len(pixs_df)<=num_clusters:
        print('Warning: all pixels will be sampled as ther are not enough samples for k-means clustering')
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


def sample_bare_pixels_perdate(scoll, metadata, lower_threshold, upper_threshold, num_scenees):
  '''
  Sample bare pixels from scenes based on k-means clustering.
  Use the top 3 scenes where there were the most number of bare pixels (clearest days with bare pixels)

  :param scoll: EOdal scene collection
  :param metadta: metadata dataframe
  :param lower_threshold: dictionary with lower thresh values for each band
  :param upper_threshold: dictionary with upper thresh values for each band
  :param num_scenes: number of scenes to keep in orer of count of bare soil pixels

  :returns: dataframe with sampled pixels
  '''

  bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12']
  samples = []

  for i, row in metadata.sort_values(by='n_bare', ascending=False).head(num_scenees).iterrows():
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
      num_clusters = min(5, len(pixs))
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


def resample_df(df_sampled, s2a, s2b, new_wavelengths, method):
  '''
  Resample all sampled pixels from Sentinel-2A and Sentinel-2B
  
  :param df_sampled:
  :param s2a: wavelengths for Sentinel-2A
  :param s2b: wavelengths for Sentinel-2B
  :param new_wavelengths: wavelegnths to upsample to
  :param method: inteprolation method among ['spline', 'pchip']
  '''

  s2a_df = df_sampled[df_sampled['sensor'] == 'Sentinel-2A']
  s2b_df = df_sampled[df_sampled['sensor'] == 'Sentinel-2B']
  df_list = [s2a_df, s2b_df]

  spectra = pd.DataFrame()

  for df in df_list:
    if len(df):
      s2_vals = df[['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]

      # Resample to 1nm
      if np.unique(df.sensor) == 'Sentinel-2A':
        df_new = upsample_spectra(s2_vals, s2a, new_wavelengths, method)
      if np.unique(df.sensor) == 'Sentinel-2B':
        df_new = upsample_spectra(s2_vals, s2b, new_wavelengths, method)
      
      # Append
      spectra = pd.concat([spectra, df_new])
  
  return spectra


def load_scolls(streams: list):
    """
    Load SceneCollection from pickled binary stream.

    :param streams:
        list of pickled binary stream to load into a SceneCollection or
        file-path to pickled binary on disk.
    :returns:
        `SceneCollection` instance.
    """
    # open empty scene collection
    scoll_out = SceneCollection()

    for stream in streams:
        if isinstance(stream, Path):
            with open(stream, "rb") as f:
                reloaded = pickle.load(f)
        elif isinstance(stream, str):
            with open(stream, "rb") as f:
                reloaded = pickle.load(f)
        elif isinstance(stream, bytes):
            reloaded = pickle.loads(stream)
        else:
            raise TypeError(f"{type(stream)} is not a supported data type")
      
        # add scenes one by one
        for _, scene in reloaded["collection"].items():
            scoll_out.add_scene(scene)

    return scoll_out


if __name__ == '__main__':

    s2a = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7,  1613.7, 2202.4]
    s2b = [442.2, 492.1, 559.0, 664.9, 703.8, 739.1, 779.7, 832.9, 864.0, 1610.4, 2185.7]
    new_wavelengths = np.arange(400, 2501, 1)

    soil_spectra = pd.DataFrame()

    tiles = {#'32TMT': [(7.66286529045287, 47.845914826827),(7.68758850048273, 46.8581759574933),(9.12805716115049, 46.8656282472998),(9.1304703619161, 47.8536277114283),(7.66286529045287, 47.845914826827)],
            #'32TLT': [(6.32792675947954,47.8225951267953),(6.37728036218156, 46.8356437136561),(7.81664846692137, 46.8595830833696),(7.79435498889756, 47.8473711518664),(6.32792675947954, 47.8225951267953)],
            #'32TNT': [(8.9997326423426, 47.8537018420053),(8.99973758744939,46.8656998728757),(10.4401498244026, 46.8566398814282),(10.4672773767603, 47.8443250460311),(8.9997326423426, 47.8537018420053)],
            #'32TNS': [(8.99973715756222,46.9537091787344),(8.9997418700468, 45.9655511480415),(10.4166561015996, 45.9567698586004),(10.442508097938, 46.9446214409232),(8.99973715756222, 46.9537091787344)],
            '32TMS': [(7.68543924709582, 46.9461622205863),(7.7089998702566, 45.9582586884214),(9.12596726307512, 45.9654817261501),(9.12826694512377, 46.9536373337673),(7.68543924709582, 46.9461622205863)],
            '32TLS': [(6.37298980762373, 46.9235610080068),(6.42002507864337, 45.9364192287437),(7.83595551698163, 45.9596225320207),(7.81471043979331, 46.94757365544),(6.37298980762373, 46.9235610080068)],
            '31TGM': [(5.62648535526562, 46.9235730577955),(5.57945945554239, 45.9364308725672),(6.99300216652316, 45.8957352511058),(7.06565713267192, 46.8814596607493),(5.62648535526562, 46.9235730577955)],
            '31TGN': [(5.67153942638492, 47.8226075594756),(5.62219565549863, 46.8356557266896),(7.05902986502132, 46.793670687313),(7.13525847525188, 47.7791570713891),(5.67153942638492, 47.8226075594756)]}
    
    year_list = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

    upsample_method = 'pchip'

    for tile_id, tile_coords in tiles.items():
        print(f'Sampling bare soil from tile {tile_id}...')

        # TO DO: could check if tile needs to be processed and skip cropping geom in necessary
        # Prepare geometry
        tile = Polygon(tile_coords)
        tile_gdf = gpd.GeoDataFrame(geometry=[tile], crs='EPSG:4326')

        # Take extra surrounding fields using GeoWP
        geowp_path = '/home/f80873755@agsad.admin.ch/mnt/Data-Work-RE/27_Natural_Resources-RE/99_GIS_User_protected/GeoWP/Landuse/Landw_Kulturflaechen/2021/01_Geodata/SHP/Nutzungsflaechen_BLW_Schweizweit_merge/ln_nutzung_BLW_2021_CH.shp'
        geowp = gpd.read_file(geowp_path).to_crs(4326)
        geowp = geowp.cx[tile_gdf.total_bounds[0]:tile_gdf.total_bounds[2], tile_gdf.total_bounds[1]:tile_gdf.total_bounds[3]]
        geowp_with_buffer = geowp.copy()
        buffer_distance = -60 # in meters, if the crs of the gdf is metric
        geowp_with_buffer['geometry'] = geowp.to_crs(2056).buffer(buffer_distance)
        geom = gpd.overlay(tile_gdf.to_crs(geowp_with_buffer.crs), geowp_with_buffer, how='intersection')

        """
        # Plot the geometry and save image
        fig, ax = plt.subplots(figsize=(10, 8))
        geom.to_crs('EPSG:4326').plot(ax=ax, color='r')
        bounds = tile_gdf.total_bounds
        ax.set_xlim((bounds[0]-0.02, bounds[2]+0.02))
        ax.set_ylim((bounds[1]-0.01, bounds[3]+0.01))
        ctx.add_basemap(ax=ax, crs='EPSG:4326')
        plt.savefig(base_dir.joinpath(f'results/geom_{tile_id}.png'))
        plt.close()          
        """

        for YEAR in year_list:
            for MONTH in range(1, 13):  # Months range from 1 to 12
                # Determine the last day of the month
                last_day = calendar.monthrange(YEAR, MONTH)[1]
                date_start = datetime(YEAR, MONTH, 1)
                date_end = datetime(YEAR, MONTH, last_day) # Includes last date

                # Setup parameters
                save_path = base_dir.joinpath(f'results/eodal_baresoil_s2_data_{tile_id}_{YEAR}_{MONTH}.pkl')
                metadata_path = save_path.with_name(save_path.name.replace('data', 'metadata'))
            
                # Get data if not done yet
                if not save_path.exists() or not metadata_path.exists():
                    print('Extracting data for year ' + str(YEAR) + '-' +str(MONTH))

                    res_baresoil = extract_s2_data(
                        aoi=geom.to_crs(4326).dissolve(), 
                        time_start=date_start,
                        time_end=date_end,
                        tile=tile_id
                    )

                    if len(res_baresoil.metadata):
                        # Save data for future use
                        with open(save_path, 'wb+') as dst:
                            dst.write(res_baresoil.data.to_pickle())     
                        with open(metadata_path, 'wb+') as dst:
                            pickle.dump(res_baresoil.metadata, dst)
                    else:
                        print(f'No data for tile {tile_id} in {YEAR}-{MONTH}. Skipping...')
                        continue
            
            
        # Load data: all scene collections and metadata for a given tile
        scoll_pattern = f'results/eodal_baresoil_s2_data_{tile_id}_*.pkl'
        scoll_files = glob.glob(str(base_dir.joinpath(scoll_pattern)))
        scoll_files = [f for f in scoll_files] # need to order the filenames?
        meta_pattern = f'results/eodal_baresoil_s2_metadata_{tile_id}_*.pkl'
        meta_files = glob.glob(str(base_dir.joinpath(meta_pattern)))
        meta_files = [f for f in meta_files] # need to order the filenames?
        print(meta_files)
    
        scoll = load_scolls(streams=scoll_files)
        dfs = []
        for file_path in meta_files:
            df = pd.read_pickle(file_path)
            dfs.append(df)
        metadata = pd.concat(dfs, ignore_index=True)
        print(len(metadata))


        # Compute band percentile thresholds in scene collection (for filtering)
        lower_threshold, upper_threshold = compute_scoll_percentiles(scoll)

        # Sample pixels: 5 pixs per date, for top 6 dates with most bare pixels
        df_sampled = sample_bare_pixels(scoll, metadata, lower_threshold, upper_threshold, num_scenes=12, samples_per_cluster=5)
        if len(df_sampled):
            sampled_path = base_dir.joinpath(f'results/sampled_pixels_{tile_id}.pkl')
            with open(sampled_path, 'wb+') as dst:
                pickle.dump(df_sampled, dst)

            # Resample to 1nm 
            spectra = resample_df(df_sampled, s2a, s2b, new_wavelengths, upsample_method)
            spectra_path = base_dir.joinpath(f'results/sampled_spectra_{tile_id}.pkl')
            with open(spectra_path, 'wb+') as dst:
                pickle.dump(spectra, dst)

            soil_spectra = pd.concat([soil_spectra, spectra])


    print(f'Saving all samples for tile {tile_id}...')
    spectra_path = base_dir.joinpath(f'results/sampled_spectra_all_CH.pkl')
    with open(spectra_path, 'wb+') as dst:
        pickle.dump(soil_spectra, dst)
    print('Finished.')


    


        