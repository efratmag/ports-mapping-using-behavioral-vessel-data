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

# CONSTANTS_FILES = ['vessels.csv.gz', 'polygons.json', 'ports.json']

VESSELS_RELEVANT_COLS = ['_id', 'class_calc', 'subclass_documented', 'built_year', 'name', 'deadweight', 'size', 'age']


def extract_coordinates(df, col='firstBlip'):

    df[[col+'_lng', col+'_lat']] = df[col].apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series)

    return df


def today_str():
    today = datetime.date.today().strftime('%Y-%m-%d')
    return today


def main(lat, lng, import_path, export_path, distance=100, debug=True):

    """

    :param lat: latitude of the snapshot location
    :param lng: longitude of the snapshot location
    :param import_path: path to directory with all relevant files
    :param export_path: path in which the output will be exported
    :param distance: bounding box Length in Km
    :param debug:
    :return:
    """

    map_file_name = f'area_snapshot_{lat}_{lng}_{today_str()}.html'

    vessels_file_name = f'area_snapshot_{lat}_{lng}_{today_str()}.csv'

    kepler_map = KeplerGl()

    results_list = []

    bounding_box = get_bounding_box(lat, lng, distance)

    area_geohash = Geohash.encode(lat, lng, 3)

    files_list = os.listdir(import_path)

    nrows = 10000 if debug else None  # 10K rows per file if debug mode == True

    vessels_df = pd.read_csv(os.path.join(import_path, 'vessels.csv.gz'), compression='gzip',
                             usecols=VESSELS_RELEVANT_COLS)

    logging.info(f'loading file polygons...')
    polygons_df = pd.read_json(os.path.join(import_path, 'polygons.json'), orient='index')
    polygons_df = gpd.GeoDataFrame(polygons_df)
    polygons_df['geometry'] = polygons_df['geometry'].apply(shape)
    polygons_df['centroid'] = polygons_df['geometry'].centroid
    polygons_df['geohash'] = polygons_df['centroid'].apply(lambda x: Geohash.encode(x.y, x.x, 3))  # resolve geohash per polygon centroid
    polygons_df = polygons_df[polygons_df['geohash'] == area_geohash].drop('centroid', axis=1)  # drop polygons out of geohash

    vessels_df = vessels_df.add_prefix('vessel_')

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

    results_df.to_csv(os.path.join(export_path, vessels_file_name))
    kepler_map.save_to_html(file_name=os.path.join(export_path, map_file_name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

