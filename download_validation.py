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
from eodal.metadata.sentinel2.parsing import parse_MTD_TL

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
from pyproj import Proj
import pyproj
from shapely.ops import transform
import pickle
import tempfile
import urllib
import uuid
import planetary_computer
from scipy.interpolate import interp2d




def angles_from_mspc(url: str):
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
    ds.mask_clouds_and_shadows(inplace=True, cloud_classes=[3, 8, 9, 10, 11])
    return ds
    

def extract_s2_data(
        aoi: gpd.GeoDataFrame,
        time_start: datetime,
        time_end: datetime,
        scene_cloud_cover_threshold: float = 80,
        feature_cloud_cover_threshold: float = 50,
        spatial_resolution: int = 20
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
    
    if len(mapper.metadata): 
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
        # Keep only metadata for corresponding scenes
        dates_to_del = [scene_id.date() for scene_id in scenes_to_del]
        mapper.metadata = mapper.metadata[~mapper.metadata['sensing_date'].isin(dates_to_del)]

    
    return mapper


def find_nearest(point, tree, pixs, k=1):
    ''' 
    Find the nearest point in pixs for each point in gdf_date
    '''
    
    distances, indices = tree.query((point.x, point.y), k=k)
    return pixs.iloc[indices]


def compute_distance(row):
    return geodesic((row['geometry'].y, row['geometry'].x), (row['geometry_s2'].y, row['geometry_s2'].x)).meters


def convert_date(date_string):
    # Function to convert date string to desired format
    if len(date_string.split()[0].split('.')[-1]) == 4:
        input_format = "%d.%m.%Y %H:%M"
    else:
        input_format = "%d.%m.%y %H:%M"
    output_format = "%Y-%m-%d %H:%M:%S"
    dt_object = datetime.strptime(date_string, input_format)
    return dt_object.strftime(output_format)


def load_data(data_path: str):
    """ 
    Load in situ data from gpkg or csv file

    :param data_path: str
    :return gdf: gpd.GeoDataFrame containing the data
    :return loc: str containing column name where locations stored
    :return traits: list containing columns of LAI and GCC stored
    """

    if data_path.endswith('.gpkg'):
        gdf = gpd.read_file(data_path).to_crs(crs='EPSG:2056').drop_duplicates()
        cols = gdf.columns
        trait = 'lai' if 'lai' in cols else 'gcc'
        loc = 'location' if 'location' in cols else None
        traits = ['lai']
        # Convert GCC to LAI
        if trait == 'gcc':
            gdf['lai'] = [None]*len(gdf)
            """ 
            gdf['lai'] = gdf['green_canopy_cover'] / 100
            """
            traits += ['green_canopy_cover']
        if not gdf['date'].apply(lambda x: isinstance(x, str)).all():
            # Convert date to string '%Y-%m-%d %H:%M:%S'
            gdf['date'] = gdf['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    if data_path.endswith('csv'):
        gdf = pd.read_csv(data_path, delimiter=';').drop_duplicates()
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf['X'], gdf['Y']), crs='EPSG:2056')
        cols = gdf.columns
        """ 
        # Convert GCC to LAI where LAI is missing
        gdf['GLAI [m2/m2]'] = gdf.apply(lambda row: row['GreenCanopyCover [%]'] / 100 if pd.isnull(row['GLAI [m2/m2]']) else row['GLAI [m2/m2]'], axis=1)
        """ 
        gdf.rename(columns={'SiteName': 'location', 'GLAI [m2/m2]':'lai', 'GreenCanopyCover [%]':'green_canopy_cover'}, inplace=True)
        traits = ['lai', 'green_canopy_cover']     
        loc = 'location'
        gdf['date'] = gdf['Time'].apply(lambda x: convert_date(x))


    return gdf, loc, traits


def transform_to_center(point):
    pixel_size = 10
    x_ul, y_ul = point.x, point.y
    x_center = x_ul + pixel_size / 2
    y_center = y_ul - pixel_size / 2
    return Point(x_center, y_center) 

if __name__ == '__main__':

  # Path to validation in situ measurements
  insitu_paths = ['data/in-situ/in-situ_glai_2022.gpkg', 'data/in-situ/in-situ_glai_2023.gpkg',\
     'data/in-situ/in-situ_gcc_2022.gpkg', 'data/in-situ/WW_traits_PhenomEn_2022.csv','data/in-situ/WW_traits_PhenomEn_2023.csv']
  insitu_paths = [base_dir.joinpath(p) for p in insitu_paths]

  val_df = pd.DataFrame()

  for insitu_path in insitu_paths:
    gdf, loc_col, trait_cols = load_data(str(insitu_path))
    locations = gdf[loc_col].unique()
    
    # Loop over dates of val data
    # Query s2 data for those dates
    # Keep pixel closest to geom
    # Save pixel LAI and spectra

    for loc in locations:
        gdf_loc = gdf[gdf[loc_col]==loc]

        for d in gdf_loc['date']:
            if d == 'April': # For 2019 data
                try:
                    s2_data = extract_s2_data(
                        aoi=gdf_loc.dissolve(),
                        time_start=pd.to_datetime('2019-04-01'), # - timedelta(days=1), 
                        time_end=pd.to_datetime('2019-04-30') + timedelta(days=1)
                    )
                except:
                    pass


            else:
                try:
                    s2_data = extract_s2_data(
                        aoi=gdf_loc.dissolve(),
                        time_start=pd.to_datetime(d), # - timedelta(days=1), 
                        time_end=pd.to_datetime(d) + timedelta(days=1)
                    )
                except:
                    pass
                
                if s2_data.data is not None:
                    for scene_id, scene in s2_data.data:
                        
                        pixs = scene.to_dataframe()
                        # Shift pixels to center
                        pixs = pixs.to_crs('EPSG:2056')
                        pixs['geometry'] = pixs['geometry'].apply(transform_to_center)
                        pixs = pixs.to_crs('EPSG:4326')
                        
                        # Find S2-data closest (within 10m) to validation data for that date
                        gdf_date = gdf_loc[gdf_loc['date'] == d]
                        gdf_date = gdf_date.to_crs(pixs.crs) # to meters, EPSG:32632

                        pixs_tree = cKDTree(pixs['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist())

                        # Apply the function to each point in gdf_date
                        points_coords = gdf_date['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()
                        distances, indices = pixs_tree.query(points_coords, k=4)
                        nearest_points = pixs.iloc[indices[0]]
                        
                        nearest_points = gdf_date['geometry'].apply(lambda point: find_nearest(point, pixs_tree, pixs, k=1))
                        nearest_points = nearest_points.rename(columns={'geometry': 'geometry_s2'})
                        gdf_date = pd.concat([gdf_date[['date', 'geometry'] +trait_cols +[loc_col]], nearest_points], axis=1)
                        """
                        gdf_date = pd.concat([gdf_date.reset_index(drop=True)] * len(nearest_points), ignore_index=True)
                        nearest_points = nearest_points.rename(columns={'geometry': 'geometry_s2'})
                        gdf_date = pd.concat([nearest_points.reset_index(drop=True), gdf_date], axis=1)
                        """

                        # # Keep if less than 10m away to ensure its the right pixel
                        gdf_date['distance'] = gdf_date.apply(compute_distance, axis=1)
                        gdf_date = gdf_date[gdf_date['distance'] < 20] #<10
                        
                        # Add data on sun/sensor angles
                        gdf_date['view_zenith'] = s2_data.metadata['sensor_zenith_angle'].values[0]
                        gdf_date['sun_zenith'] = s2_data.metadata['sun_zenith_angle'].values[0]
                        gdf_date["relative_azimuth"] = (s2_data.metadata["sun_azimuth_angle"].values[0] - s2_data.metadata["sensor_azimuth_angle"].values[0])%360
                        
                        # Save lai and spectra 
                        if len(gdf_date):
                            val_df = pd.concat([val_df, gdf_date[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12','geometry', 'date', 'view_zenith', 'sun_zenith', 'relative_azimuth'] +trait_cols +[loc_col]]])
                        
                        """ 
                        ## OTHER method
                        gdf_date = gdf_loc[gdf_loc['date'] == d]
                        gdf_date = gdf_date.to_crs('EPSG:32632')
                        for i, row in gdf_date.iterrows():
                            pt = row['geometry']
                            res = 1 # meter
                            # Create a box with 1m side centered around pt
                            bbox = box(pt.x - res/2, pt.y - res/2, pt.x + res/2, pt.y + res/2)
                            projector = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:4326', always_xy=True).transform
                            bbox = transform(projector, bbox)  
                            try:
                                # Clip bands to bbox
                                scene = scene.clip_bands(clipping_bounds=bbox)
                                pixs = scene.to_dataframe()
                                if len(pixs==1):
                                    df_data = pd.DataFrame([row[['date', 'lai', 'location', 'geometry']]]).reset_index(drop=True)
                                    df_data = pd.concat([pixs[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']], df_data], axis=1)
                                    val_df = pd.concat([val_df, df_data], axis=0, ignore_index=True)
                                else:
                                    print(f'Found multiple pixels {len(pixs)}')
                            except:
                                pass
                        """

  # Save in-situ val data
  data_path = base_dir.joinpath(f'results/validation_data_extended_angles_shift.pkl')
  with open(data_path, 'wb+') as dst:
      pickle.dump(val_df, dst)


