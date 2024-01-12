'''
Extract Sentinel-2 surface reflectance (SRF) data for selected field parcels
over the growing season and run PROSAIL simulations.

@author Selene Ledain
'''

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tempfile
import urllib
import uuid
from datetime import datetime
import planetary_computer

import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, os.path.join(base_dir, "eodal"))
from eodal.config import get_settings
from eodal.core.scene import SceneCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from eodal.metadata.sentinel2.parsing import parse_MTD_TL
from eodal.utils.sentinel2 import get_S2_platform_from_safe, _url_to_safe_name
from pathlib import Path
from rtm_inv.core.lookup_table import generate_lut
from typing import Any, Dict, List, Optional
from shapely.geometry import Point

from utils import get_farms

settings = get_settings()
settings.USE_STAC = True # False # 
logger = settings.logger

# Sentinel-2 bands to extract and use for PROSAIL runs
band_selection = [
    'B02']


def angles_from_mspc(url: str) -> Dict[str, float]:
    """
    Extract viewing and illumination angles from MS Planetary Computer
    metadata XML (this is a work-around until STAC provides the angles
    directly)

    :param url:
        URL to the metadata XML file
    :returns:
        extracted angles as dictionary
    """
    response = urllib.request.urlopen(planetary_computer.sign_url(url)).read()
    temp_file = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}.xml')
    with open(temp_file, 'wb') as dst:
        dst.write(response)

    metadata = parse_MTD_TL(in_file=temp_file)
    # get sensor zenith and azimuth angle
    sensor_angles = ['SENSOR_ZENITH_ANGLE', 'SENSOR_AZIMUTH_ANGLE']
    sensor_angle_dict = {
        k: v for k, v in metadata.items() if k in sensor_angles}
    return sensor_angle_dict


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
    ds.mask_clouds_and_shadows(inplace=True)
    return ds


def get_s2_mapper(
    s2_mapper_config: MapperConfigs,
    output_dir: Path
    ) -> Mapper:
    """
    Setup an EOdal `Mapper` instance, query and load Sentinel-2 data

    :param s2_mapper_config:
        configuration telling EOdal what to do (which geographic region and
        time period should be processed)
    :param output_dir:
        directory where to store the query for documentation
    :returns:
        EOdal `Mapper` instance with populated `metadata` and `data`
        attributes
    """
    # setup Sentinel-2 mapper to get the relevant scenes
    mapper = Mapper(s2_mapper_config)
    # check if the metadata and data has been already saved.
    # In this case we can simply read the data from file and create
    # a new mapper instance
    fpath_metadata = output_dir.joinpath('eodal_mapper_metadata.gpkg')
    fpath_mapper = output_dir.joinpath('eodal_mapper_scenes.pkl')
    if fpath_mapper.exists() and fpath_metadata.exists():
        metadata = gpd.read_file(fpath_metadata)
        scenes = SceneCollection.from_pickle(stream=fpath_mapper)
        mapper.data = scenes
        mapper.metadata = metadata
        return mapper
    # otherwise, it's necessary to query the data again
    # query metadata records
    mapper.query_scenes()
    
    # extract the angular information about the sensor (still not
    # part of the default STAC metadata). Since it's only a work-
    # around it's a bit slow...
    mapper.metadata['href_xml'] = mapper.metadata.assets.apply(
        lambda x: x['granule-metadata']['href']
    )
    mapper.metadata['sensor_angles'] = mapper.metadata['href_xml'].apply(
        lambda x, angles_from_mspc=angles_from_mspc: angles_from_mspc(x)
    )
    mapper.metadata['sensor_zenith_angle'] = \
        mapper.metadata['sensor_angles'].apply(
            lambda x: x['SENSOR_ZENITH_ANGLE'])

    mapper.metadata['sensor_azimuth_angle'] = \
        mapper.metadata['sensor_angles'].apply(
            lambda x: x['SENSOR_AZIMUTH_ANGLE'])

    return mapper



if __name__ == '__main__':

    cwd = Path(__file__).parent.absolute()
    import os
    os.chdir(cwd)

    # metadata filters for retrieving S2 scenes
    metadata_filters = [
        Filter('processing_level', '==', 'Level-2A')
    ]

    # global setup
    out_dir = Path('../results')
    out_dir.mkdir(exist_ok=True)

    #######################################################################
    """
    # extraction for 2019
    data_dir = Path('../data/auxiliary/field_parcels_ww_2019')
    year = 2019
    farms = ['SwissFutureFarm']

    # get field parcel geometries organized by farm
    farm_gdf_dict = get_farms(data_dir, farms, year)
    
    """

    data_dir = Path('../data/coords.csv')
    year = 2019

    # loop over coordinates, get angles and add save in a single dataframe/file
    coords = pd.read_csv(data_dir, delimiter=';')

    angles_df = pd.DataFrame()
    for i, coord in coords.iterrows():
        point = Point(coord['lon'], coord['lat'])
        gds = gpd.GeoSeries([point], crs='EPSG:4326') #'EPSG:32632'
        #gds.set_crs(epsg=32632)
        gds.name = 'geometry'

        feature = Feature.from_geoseries(gds=gds) 
        s2_mapper_config = MapperConfigs(
            collection='sentinel2-msi',
            time_start=datetime(year, 1, 1),
            time_end=datetime(year, 12, 31),
            feature=feature,
            metadata_filters=metadata_filters
        )

        try:
            mapper = get_s2_mapper(
                s2_mapper_config=s2_mapper_config,
                output_dir=out_dir
            )
            angles_df = pd.concat([angles_df, mapper.metadata[['sun_azimuth_angle', 'sun_zenith_angle', 'sensor_azimuth_angle', 'sensor_zenith_angle']]])
            
        except Exception as e:
            logger.error(f'Failed {farm}: {e}')
            pass

    with open(out_dir.joinpath('s2_angles_switzerland.pkl'), 'wb') as f:
        pickle.dump(angles_df, f)