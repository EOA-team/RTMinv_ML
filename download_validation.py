''' 
Download S2 data for validation site and form a validation dataset

@Selene Ledain
'''

import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, os.path.join(base_dir, "eodal"))
import eodal
from eodal.config import get_settings
from eodal.core.scene import SceneCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

Settings = get_settings()
# set to False to use a local data archive
Settings.USE_STAC = True

import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from geopy.distance import geodesic
from pyproj import Proj, transform
import pickle



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
        scene_cloud_cover_threshold: float = 80,
        feature_cloud_cover_threshold: float = 50,
        spatial_resolution: int = 10
    ) -> SceneCollection:
    """
    Extracts Sentinel-2 data from the STAC SAT archive for a given area and time period.
    Scenes that are too cloudy or contain nodata (blackfill), only, are discarded.

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
        'scene_constructor_kwargs': {'band_selection': ['B01','B02','B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B11', 'B12', 'SCL']},                      # here you could specify which bands to read
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

        # delete scenes containing only no-data
        for scene_id in scenes_to_del:
            del mapper.data[scene_id]

    
    return mapper


def find_nearest(point, tree):
    ''' 
    Find the nearest point in pixs for each point in gdf_date
    '''
    _, index = tree.query((point.x, point.y))
    return pixs.iloc[index]

def compute_distance(row):
    return geodesic((row['geometry'].y, row['geometry'].x), (row['geometry_s2'].y, row['geometry_s2'].x)).meters




if __name__ == '__main__':

  # Path to validation in situ measurements
  lai_path = base_dir.joinpath('data/in-situ_glai.gpkg')
  gdf = gpd.read_file(lai_path)
  locations = ['Strickhof', 'SwissFutureFarm', 'Witzwil']

  # Loop over dates of val data
  # Query s2 data for those dates
  # Keep pixel closest to geom
  # Save pixel LAI and spectra

  val_df = pd.DataFrame()

  for loc in locations:
    gdf_loc = gdf[gdf.location==loc]

    for d in gdf_loc['date']:
        print('date', d)
        try:
            s2_data = extract_s2_data(
                aoi=gdf_loc.dissolve(),
                time_start=pd.to_datetime(d) - timedelta(days=1), 
                time_end=pd.to_datetime(d) + timedelta(days=1)
            )
        except:
            pass
        
        if s2_data.data is not None:
            print('Not null')
            for scene_id, scene in s2_data.data:
                """
                pixs = scene.to_dataframe()
                pixs = pixs.to_crs('EPSG:4326')
                
                # Find S2-data closest (within 10m) to validation data for that date
                gdf_date = gdf_loc[gdf_loc['date'] == d]
                gdf_date = gdf_date.to_crs(pixs.crs) # to meters, EPSG:32632

                pixs_tree = cKDTree(pixs['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist())

                # Apply the function to each point in gdf_date
                nearest_points = gdf_date['geometry'].apply(lambda point: find_nearest(point, pixs_tree))
                nearest_points = nearest_points.rename(columns={'geometry': 'geometry_s2'})
                gdf_date = pd.concat([gdf_date[['date', 'lai', 'location', 'geometry']], nearest_points], axis=1)

                # # Keep if less than 10m away to ensure its the right pixel
                gdf_date['distance'] = gdf_date.apply(compute_distance, axis=1)      
                gdf_date = gdf_date[gdf_date['distance'] <10]
                
                # Save lai and spectra 
                if (len(gdf_date)):
                    val_df = pd.concat([val_df, gdf_date[['lai', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12','geometry', 'date', 'location']]])
                """
                gdf_date = gdf_loc[gdf_loc['date'] == d]
                gdf_date = gdf_date.to_crs('EPSG:32632')
                for i, row in gdf_date.iterrows():
                    pt = row['geometry']
                    res = 1 # meter
                    # Create a box with 1m side centered around pt
                    bbox = box(pt.x - res/2, pt.y - res/2, pt.x + res/2, pt.y + res/2)
                    try:
                        # Clip bands to bbox
                        scene.clip_bands(clipping_bounds=bbox, inplace=True)
                        pixs = scene.to_dataframe()
                        if len(pixs==1):
                            df_data = pd.DataFrame([row[['date', 'lai', 'location', 'geometry']]])
                            val_df = pd.concat([val_df, df_data], axis=1)
                        else:
                            print(f'Found multiple pixels {len(pixs)}')
                    except:
                        pass


  # Save in-situ val data
  data_path = base_dir.joinpath(f'results/validation_data2.pkl')
  with open(data_path, 'wb+') as dst:
      pickle.dump(val_df, dst)
