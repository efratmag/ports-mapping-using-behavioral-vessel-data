from shapely.geometry import Point

from shapely import ops
from area_snaptshot.area_snapshot import load_and_process_polygons_file, extract_coordinates

import os
import fire
import pandas as pd
import numpy as np
import geopandas as gpd

import logging

from geo_utils import geo_dist

ACTIVITIES_FILES = ['mooring.csv.gz', 'drifting.csv.gz', 'port_calls.csv.gz', 'anchoring.csv.gz']

VESSELS_RELEVANT_COLS = ['_id', 'class_calc', 'subclass_documented', 'built_year',
                         'name', 'deadweight', 'draught', 'size', 'age']


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

    df[col+'_distance_from_shore'] = df.apply(lambda x: geo_dist(x[col + '_lat'],
                                                                 x[col + '_lng'],
                                                                 x[col+'_nearest_shore_lat'],
                                                                 x[col+'_nearest_shore_lng']), axis=1)

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
    polygons_df = load_and_process_polygons_file(polygons_file_path)

    shorelines_file_path = os.path.join(import_path, 'shoreline_layer_merged.geojson')

    shore_lines = load_and_process_shorelines_df(shorelines_file_path)

    logging.info(f'loading file vessels...')
    vessels_df = pd.read_csv(os.path.join(import_path, 'vessels.csv.gz'), compression='gzip',
                             usecols=VESSELS_RELEVANT_COLS, index_col='_id')

    vessels_size = vessels_df[['size']].dropna()
    vessels_size['size_category'] = pd.qcut(vessels_size['size'], 3, labels=["small", "medium", "big"])
    vessels_df = vessels_df.merge(vessels_size['size_category'], left_index=True, right_index=True, how='left')

    vessels_df['class_calc_updated'] = vessels_df['class_calc'].fillna('Other').replace({'MilitaryOrLaw': 'Other',
                                                                                         'Pleasure': 'Other',
                                                                                         'Unknown': 'Other',
                                                                                         'HighSpeedCraft': 'Other'})

    vessels_df = vessels_df.add_prefix('vessel_')

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

            df = is_in_polygon_features(df)

            # df = calc_distance_from_shore(df, shore_lines, 'firstBlip') # TODO: improve logic / get updated layer
            # df = calc_distance_from_shore(df, shore_lines, 'lastBlip')

            logging.info(f'merging vessels data...')
            df = df.merge(vessels_df, left_on='vesselId', right_index=True)

            results_list.append(df)

            df.to_csv(os.path.join(export_path, file_name), index=False, compression='gzip')

    results_df = pd.concat(results_list)
    results_df.to_csv(os.path.join(export_path, 'all_activities.csv.gz'), index=False, compression='gzip')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

