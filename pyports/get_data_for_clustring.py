import pandas as pd
import geopandas as gpd
import os
import logging

from shapely.geometry import Point

from pyports.geo_utils import merge_polygons


def get_data_for_clustering(import_path, activity, debug, sub_area_polygon_fname, blip):

    logging.info('loading data for clustering - START')

    df_for_clustering_fname = f'features/df_for_clustering_{activity}.csv'

    nrows = 10000 if debug else None  # will load first 10K rows if debug == True

    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False, nrows=nrows)

    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    ports_df = gpd.read_file(os.path.join(import_path, 'maps/ports.geojson'))  # WW ports
    ports_df.drop_duplicates(subset='name', inplace=True)
    polygons_df = gpd.read_file(os.path.join(import_path, 'maps/polygons.geojson'))  # WW polygons

    shoreline_df = gpd.read_file(os.path.join(import_path, 'maps/shoreline_layer.geojson'))  # shoreline layer
    main_land = merge_polygons(shoreline_df[:4])  # create multipolygon of the big continents
    shoreline_polygon = merge_polygons(shoreline_df)  # merging shoreline_df to one multipolygon

    logging.info('loading data for clustering - END')

    return df, ports_df, polygons_df, main_land, shoreline_polygon

