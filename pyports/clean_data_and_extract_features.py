from shapely.geometry import shape, Point

from shapely import ops
from pyports.geo_utils import haversine, extract_coordinates
import Geohash
import os
import fire
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree

import logging

# TODO: add (1)new class name (2)next port name

ACTIVITIES_FILES = ['mooring.csv.gz', 'drifting.csv.gz', 'port_calls.csv.gz', 'anchoring.csv.gz']

VESSELS_RELEVANT_COLS = ['_id', 'class_calc', 'subclass_documented', 'built_year',
                         'name', 'deadweight', 'draught', 'size', 'age']


def load_and_process_polygons_file(polygons_file_path, area_geohash=None):

    """
    a function that loads the polygons file, converting to GeoPandas and drops polygons out of area snapshot
    :param polygons_file_path: i.e 'polygons.json'
    :param area_geohash: geohash of snapshot area
    :return:
    """

    logging.info(f'loading file polygons...')
    polygons_df = pd.read_json(polygons_file_path, orient='index')
    polygons_df = gpd.GeoDataFrame(polygons_df)  # convert to GeoPandas
    polygons_df['polygon_id'] = polygons_df.index
    polygons_df = polygons_df.drop('_id', axis=1).reset_index(drop=True)
    polygons_df['geometry'] = polygons_df['geometry'].apply(shape)  # convert to shape object
    polygons_df['centroid'] = polygons_df['geometry'].centroid  # get polygons centroid
    polygons_df['geohash'] = polygons_df['centroid'].apply(lambda x: Geohash.encode(x.y, x.x, 2))  # resolve geohash per polygon centroid
    polygons_df['polygon_area_type'] = [d.get('areaType') for d in polygons_df.properties]  # add areaType column from nested dict
    polygons_df['name'] = [d.get('title') for d in polygons_df.properties]  # add name column from nested dict

    if area_geohash:
        polygons_df = polygons_df[polygons_df['geohash'] == area_geohash].drop('centroid', axis=1)  # drop polygons out of geohash

    return polygons_df


def find_intersection_with_polygons(df, polygons_df, col_prefix, force_enrichment=True):

    logging.info(f'converting to Point - {col_prefix}...')

    if col_prefix+'_polygon_id' not in df.columns or force_enrichment:

        df['geometry'] = df.apply(lambda x: Point(x[col_prefix + '_lng'], x[col_prefix + '_lat']) if not pd.isna(
            x[col_prefix + '_lat']) else None, axis=1)

        df = gpd.GeoDataFrame(df)

        logging.info(f'performing spatial join - {col_prefix}...')
        df = gpd.sjoin(df, polygons_df[['polygon_id', 'polygon_area_type', 'geometry'
                                        ]], how='left').drop('index_right', axis=1)

        df['polygon_id'], df['polygon_area_type'] = df['polygon_id'].astype(str), df['polygon_area_type'].astype(str)

        logging.info(f'aggregating polygons - {col_prefix}...')
        merged_polygons = df.groupby('_id', as_index=False).agg({'polygon_id': ', '.join,
                                                                 'polygon_area_type': ', '.join}).replace({'nan': None})

        df = df.drop_duplicates('_id').drop(['polygon_id', 'polygon_area_type'], axis=1)
        df = df.merge(merged_polygons, on='_id')
        df = df.rename(columns={'polygon_id': col_prefix + '_polygon_id',
                                'polygon_area_type': col_prefix + '_polygon_area_type'})

    return df


def calc_distance_from_shore(df, shore_lines, col="firstBlip"):

    logging.info(f'Calc Distance From Shore - {col}...')

    if col+'_closer_shore_lat' not in df.columns:

        df['geometry'] = df.apply(lambda x: Point(x[col + '_lng'], x[col + '_lat']) if not pd.isna(
            x[col + '_lat']) else None, axis=1)

    df = gpd.GeoDataFrame(df)

    df['nearest_shore_point'] = df.apply(lambda x: ops.nearest_points(shore_lines, x['geometry'])[0], axis=1)
    df[col+'_nearest_shore_lng'] = df['nearest_shore_point'].apply(lambda x: x.x)
    df[col+'_nearest_shore_lat'] = df['nearest_shore_point'].apply(lambda x: x.y)

    df = df.drop('nearest_shore_point', axis=1)

    df[col+'_distance_from_shore'] = df.apply(lambda x: haversine((x[col + '_lat'],
                                                                   x[col + '_lng']),
                                                                  (x[col+'_nearest_shore_lat'],
                                                                   x[col+'_nearest_shore_lng'])), axis=1)

    return df


def load_and_process_shorelines_df(shorelines_file_path):

    logging.info(f'loading file shorelines...')
    shore_lines_df = gpd.read_file(shorelines_file_path)
    shore_lines = ops.linemerge(shore_lines_df['geometry'].values)

    return shore_lines


def is_in_polygon_features(df):

    df["firstBlip_in_polygon"] = df["firstBlip_polygon_id"].notna()

    conditions = [
        (df["firstBlip_in_polygon"] == True) & (df['lastBlip_polygon_id'].isna() == True),
        (df["firstBlip_in_polygon"] == False) & (df['lastBlip_polygon_id'].isna() == True),
        (df["lastBlip_polygon_id"].isna() == False)
    ]
    choices = ["not_ended", "False", "True"]
    df["lastBlip_in_polygon"] = np.select(conditions, choices)

    return df


def add_dist_from_nearest_port(df, ports_df):

    tmp_df = pd.DataFrame()
    tmp_df['lng'] = ports_df.center_coordinates.map(lambda x: x[0])
    tmp_df['lat'] = ports_df.center_coordinates.map(lambda x: x[1])
    tmp_df['name'] = ports_df.name
    tmp_df['country'] = ports_df.country
    ports_gdf = gpd.GeoDataFrame(
        tmp_df, geometry=gpd.points_from_xy(tmp_df.lng, tmp_df.lat))

    tree = BallTree(ports_gdf[['lat', 'lng']].values, leaf_size=2)

    df['distance_nearest_port'], _ = tree.query(df[['firstBlip_lat', 'firstBlip_lng']].values, k=1)

    return df


def load_and_process_vessels_file(import_path):

    vessels_df = pd.read_csv(import_path, compression='gzip',
                             usecols=VESSELS_RELEVANT_COLS, index_col='_id')

    vessels_size = vessels_df[['size']].dropna()
    vessels_size['size_category'] = pd.qcut(vessels_size['size'], 3, labels=["small", "medium", "big"])
    vessels_df = vessels_df.merge(vessels_size['size_category'], left_index=True, right_index=True, how='left')

    vessels_df['class_calc_updated'] = vessels_df['class_calc'].fillna('Other').replace({'MilitaryOrLaw': 'Other',
                                                                                         'Pleasure': 'Other',
                                                                                         'Unknown': 'Other',
                                                                                         'HighSpeedCraft': 'Other'})
    vessels_df = vessels_df.add_prefix('vessel_')

    return vessels_df


def main(import_path, export_path, debug=True):

    """
    This code will load the activities datasets, extract lat,lng, and calculate intersections with polygons
    :param import_path: path to directory with all relevant files
    :param export_path: path in which the output will be exported
    :param debug: if True, only a first 10K rows of each file will be processed
    :return:
    """

    files_list = os.listdir(import_path)

    nrows = 10000 if debug else None  # 10K rows per file if debug mode == True

    polygons_file_path = os.path.join(import_path, 'polygons.json')
    vessels_file_path = os.path.join(import_path, 'vessels.csv.gz')

    polygons_df = load_and_process_polygons_file(polygons_file_path)
    vessels_df = load_and_process_vessels_file(vessels_file_path)

    try:

        ports_df = pd.read_json(os.path.join(import_path, 'ports.json'), orient='index')
    except Exception as e:
        logging.info(f'failed to load ports_df using pd.read_json! (error: {e}), trying using json.load')
        import json

        with open(os.path.join(import_path, 'ports.json'), 'r') as portsfile:
            ports_df = json.load(portsfile)
        ports_df = pd.DataFrame.from_dict(ports_df, orient='index')

    logging.info(f'loading file vessels...')

    results_list = []

    for file_name in files_list:

        file_path = os.path.join(import_path, file_name)

        if file_name in ACTIVITIES_FILES:
            logging.info(f'loading file {file_name}...')
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)

            df = extract_coordinates(df, 'firstBlip')
            df = extract_coordinates(df, 'lastBlip')

            df = find_intersection_with_polygons(df, polygons_df, 'firstBlip')
            df = find_intersection_with_polygons(df, polygons_df, 'lastBlip')
            df = add_dist_from_nearest_port(df, ports_df)

            df = is_in_polygon_features(df)

            logging.info(f'merging vessels data...')
            df = df.merge(vessels_df, left_on='vesselId', right_index=True)

            logging.info(f'merging nextPort data...')
            df = df.merge(polygons_df.set_index('polygon_id'), left_on='nextPort', right_index=True, how='left').rename(
                columns={'name': 'nextPort_name'})
            df.nextPort_name.fillna('UNKNOWN', inplace=True)

            df['activity'] = file_name.replace('.csv.gz', '')

            results_list.append(df)

            df.to_csv(os.path.join(export_path, file_name), index=False, compression='gzip')

    results_df = pd.concat(results_list)

    results_df.to_csv(os.path.join(export_path, 'all_activities.csv.gz'), index=False, compression='gzip')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

