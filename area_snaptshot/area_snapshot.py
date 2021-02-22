from area_snaptshot.map_config import MAP_CONFIG
from geo_utils import get_bounding_box, isin_box
import os
import fire
import pandas as pd
import datetime
import geopandas as gpd
from shapely.geometry import shape
import Geohash
from keplergl import KeplerGl
import logging

ACTIVITIES_FILES = ['mooring.csv.gz', 'drifting.csv.gz', 'port_calls.csv.gz', 'anchoring.csv.gz']

VESSELS_RELEVANT_COLS = ['_id', 'class_calc', 'subclass_documented', 'built_year', 'name', 'deadweight', 'size', 'age']


def extract_coordinates(df, col='firstBlip'):

    """
    a function that extracts lat and lng from a Series with geometry dict
    :param df: Dataframe with coordinates dict columns
    :param col: name of column for coordinates extraction
    :return:
    """

    if col+'_lng' not in df.columns and col+'_lat' not in df.columns:

        df[[col+'_lng', col+'_lat']] = df[col].apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series)

    return df


def today_str():
    """
    a simple function that returns today's date
    :return:
    """
    today = datetime.date.today().strftime('%Y-%m-%d')
    return today


def load_and_process_polygons_file(polygons_file_path, area_geohash):

    """
    a function that loading the polygons file, converting to GeoPandas, and drops  polygons out of area snapshot
    :param import_path: path to folder in which polygons_file_name
    :param polygons_file_path: i.e 'polygons.json'
    :param area_geohash: geohash of snapshot area
    :return:
    """

    logging.info(f'loading file polygons...')
    polygons_df = pd.read_json(polygons_file_path, orient='index')
    polygons_df = gpd.GeoDataFrame(polygons_df)  # convert to GeoPandas
    polygons_df['geometry'] = polygons_df['geometry'].apply(shape)  # convert to shape object
    polygons_df['centroid'] = polygons_df['geometry'].centroid  # get polygons centroid
    polygons_df['geohash'] = polygons_df['centroid'].apply(lambda x: Geohash.encode(x.y, x.x, 2))  # resolve geohash per polygon centroid
    polygons_df = polygons_df[polygons_df['geohash'] == area_geohash].drop('centroid', axis=1)  # drop polygons out of geohash

    return polygons_df


def main(lat, lng, import_path, export_path, distance=100, debug=True):

    """
    This code will create an snapshot of a area for a given location and distance
    :param lat: latitude of the snapshot location
    :param lng: longitude of the snapshot location
    :param import_path: path to directory with all relevant files
    :param export_path: path in which the output will be exported
    :param distance: bounding box length in Km
    :param debug: if True, only a first 10K rows of each file will be processed
    :return:
    """

    map_file_name = f'area_snapshot_{lat}_{lng}_{today_str()}.html'

    vessels_file_name = f'area_snapshot_{lat}_{lng}_{today_str()}.csv'

    kepler_map = KeplerGl()

    #  define the map center by the given coordinates
    MAP_CONFIG['config']['mapState']['latitude'] = lat
    MAP_CONFIG['config']['mapState']['longitude'] = lng

    results_list = []

    bounding_box = get_bounding_box(lat, lng, distance)

    area_geohash = Geohash.encode(lat, lng, 2)

    files_list = os.listdir(import_path)

    nrows = 10000 if debug else None  # 10K rows per file if debug mode == True

    vessels_df = pd.read_csv(os.path.join(import_path, 'vessels.csv.gz'), compression='gzip',
                             usecols=VESSELS_RELEVANT_COLS)

    vessels_df = vessels_df.add_prefix('vessel_')

    polygons_file_path = os.path.join(import_path, 'polygons.json')
    polygons_df = load_and_process_polygons_file(polygons_file_path, area_geohash)

    for file_name in files_list:

        file_path = os.path.join(import_path, file_name)

        if file_name in ACTIVITIES_FILES:
            logging.info(f'loading file {file_name}...')
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            df = extract_coordinates(df, 'firstBlip')
            df = df[['_id', 'vesselId', 'firstBlip_lat', 'firstBlip_lng']]

            df = df[df.apply(lambda x: isin_box(x['firstBlip_lat'], x['firstBlip_lng'], bounding_box), axis=1)]
            df['action'] = file_name.split('.')[0]

            results_list.append(df)

    results_df = pd.concat(results_list)

    results_df = results_df.merge(vessels_df, left_on='vesselId', right_on='vessel__id').drop('vessel__id', axis=1)

    kepler_map.add_data(results_df, 'vessels_behavior')
    kepler_map.add_data(polygons_df, 'polygons')

    results_df.to_csv(os.path.join(export_path, vessels_file_name), index=False)
    kepler_map.save_to_html(file_name=os.path.join(export_path, map_file_name), config=MAP_CONFIG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)

