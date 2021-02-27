from shapely.geometry import Point

from area_snaptshot.area_snapshot import load_and_process_polygons_file, extract_coordinates

import os
import fire
import pandas as pd

import geopandas as gpd

import logging

ACTIVITIES_FILES = ['mooring.csv.gz', 'drifting.csv.gz', 'port_calls.csv.gz', 'anchoring.csv.gz']

VESSELS_RELEVANT_COLS = ['_id', 'class_calc', 'subclass_documented', 'built_year', 'name', 'deadweight', 'size', 'age']


def find_intersection_with_polygons(df, polygons_df, col_prefix):

    logging.debug(f'converting to Point - {col_prefix}...')

    df['geometry'] = df.apply(lambda x: Point(x[col_prefix + '_lng'], x[col_prefix + '_lat']) if not pd.isna(
        x[col_prefix + '_lat']) else None, axis=1)

    df = gpd.GeoDataFrame(df)

    logging.debug(f'performing spatial join - {col_prefix}...')
    df = gpd.sjoin(df, polygons_df[['polygon_id', 'geometry']], how='left').drop('index_right', axis=1)

    df['polygon_id'] = df['polygon_id'].astype(str)

    logging.debug(f'aggregating polygons - {col_prefix}...')
    merged_polygons = df.groupby('_id', as_index=False).agg({'polygon_id': ', '.join}).replace({'nan': None})

    df = df.drop_duplicates('_id').drop('polygon_id', axis=1)
    df = df.merge(merged_polygons, on='_id')
    df = df.rename(columns={'polygon_id': col_prefix + '_polygon_id'})

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

    for file_name in files_list:

        file_path = os.path.join(import_path, file_name)

        if file_name in ACTIVITIES_FILES:
            logging.info(f'loading file {file_name}...')
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            df = extract_coordinates(df, 'firstBlip')
            df = extract_coordinates(df, 'lastBlip')

            df = find_intersection_with_polygons(df, polygons_df, 'firstBlip')
            df = find_intersection_with_polygons(df, polygons_df, 'lastBlip')

            df.to_csv(os.path.join(export_path, file_name), index=False, compression='gzip')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)

