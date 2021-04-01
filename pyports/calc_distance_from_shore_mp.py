import itertools

from shapely.geometry import Point
from multiprocessing import Pool
from datetime import datetime
from shapely import ops

import os
import fire
import pandas as pd
import geopandas as gpd

import logging

from pyports.geo_utils import polygons_to_multi_lines, merge_polygons, haversine


def calc_nearest_points(params):

    geo_layer_multi_line, geo_layer_multi_polygon, points = params

    results_list = []

    for record in points:

        index, point = record

        try:

            if point.within(geo_layer_multi_polygon):
                distance = 0

            else:
                nearest_point = ops.nearest_points(geo_layer_multi_line, point)[0]
                distance = haversine((nearest_point.y, nearest_point.x),
                                     (point.y, point.x))

            results_list.append({'index': index, 'distance': distance,
                                 'nearest_lat': nearest_point.y, 'nearest_lng': nearest_point.x})

        except Exception as e:
            logging.info(f'point at index {index} failed, reason - {e}')
            results_list.append({'index': index, 'distance': None,
                                 'nearest_lat': None, 'nearest_lng': None})

    logging.info('process finished calculating 1K points')
    return results_list


def chunker(iterable, n):

    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def main(df_import_path, geo_layer_import_path, export_path, col='firstBlip', debug=True):

    """
    This code will load the activities datasets, extract lat,lng, and calculate intersections with polygons
    :param df_import_path: path to directory with all relevant files
    :param export_path: path in which the output will be exported
    :param col: firstBlip / lastBlip
    :param debug: if True, only a first 10K rows of each file will be processed
    :return:
    """

    nrows = 1000 if debug else None

    df = pd.read_csv(df_import_path, nrows=nrows)

    logging.info('loading and processing geo_layer file - START')

    geo_layer = gpd.read_file(geo_layer_import_path)
    geo_layer_multi_line = polygons_to_multi_lines(geo_layer)
    geo_layer_multi_polygon = merge_polygons(geo_layer)

    logging.info('loading and processing geo_layer file - END')

    points = df[[f'{col}_lat', f'{col}_lng']].itertuples()

    params_list = [(index, Point(lng, lat)) for index, lat, lng in points]

    params_list = [(geo_layer_multi_line, geo_layer_multi_polygon, chunk) for chunk in chunker(params_list, 100)]

    pool = Pool(processes=8)

    start_time = datetime.now()

    logging.info('calc distance with multiprocessing - START')

    distances = pool.map(calc_nearest_points, params_list)

    logging.info('calc distance with multiprocessing - END')

    end_time = datetime.now()

    total_process_duration = end_time - start_time

    pool.close()

    pool.terminate()
    pool.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

