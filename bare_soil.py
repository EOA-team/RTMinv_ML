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


if __name__ == '__main__':

    # Setup parameters
    shp_path = base_dir.joinpath('data/Strickhof.shp')
    res_dir = base_dir.joinpath('results/eodal_s2_baresoil') # where to save rasters
    date_start = '2017-03-01'
    date_end = '2023-12-31'
    save_path = base_dir.joinpath(f'results/eodal_baresoil_s2_data_strickhof.pkl')

    # Add buffer to remove field edges
    aoi = gpd.read_file(shp_path).dissolve()
    aoi = aoi.to_crs('EPSG:2056')
    aoi_with_buffer = aoi.copy()
    buffer_distance = -20 # in meters
    aoi_with_buffer['geometry'] = aoi.buffer(buffer_distance)
    aoi_with_buffer = aoi_with_buffer.to_crs('EPSG:4326')

    # Get data
    res_baresoil = extract_s2_data(
      aoi=aoi_with_buffer,
      time_start=datetime.strptime(date_start, '%Y-%m-%d'),
      time_end=datetime.strptime(date_end, '%Y-%m-%d')
    )

    # Save data for future use
    with open(save_path, 'wb+') as dst:
        dst.write(res_baresoil.data.to_pickle())     
    metadata_path = save_path.with_name(save_path.name.replace('data', 'metadata'))
    with open(metadata_path, 'wb+') as dst:
        pickle.dump(res_baresoil.metadata, dst)