from shapely.geometry import shape, Point
from shapely import ops
from pyports.geo_utils import haversine
import os
import fire
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree
import pymongo
from bson import ObjectId
import json
import logging


def extract_coordinates(df, col='firstBlip'):

    """
    a function that extracts lat and lng from a Series with geometry dict
    :param df: Dataframe with coordinates dict columns
    :param col: name of column for coordinates extraction
    :return: df with lat lng coordinates
    """

    if col+'_lng' not in df.columns and col+'_lat' not in df.columns:

        logging.info(f'extracting coordinates for {col}...')
        coordinates_df = df[col].dropna().apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series)
        coordinates_df = coordinates_df.rename(columns={0: col+'_lng', 1: col+'_lat'})

        df = df.merge(coordinates_df, left_index=True, right_index=True, how='left')

    return df


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

        df = df.drop_duplicates('_id').drop(['polygon_id', 'polygon_area_type', 'geometry'], axis=1)
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

    ports_gdf = gpd.GeoDataFrame(
        ports_df, geometry=gpd.points_from_xy(ports_df.lng, ports_df.lat))

    tree = BallTree(ports_gdf[['lat', 'lng']].values, leaf_size=2)

    df['distance_nearest_port'], _ = tree.query(df[['firstBlip_lat', 'firstBlip_lng']].values, k=1)

    return df


def get_ports_wa_polygons(import_path=None, db=None):

    """
    :param import_path: path to the ports & waiting areas file location
    :param db:
    :return:
    """

    logging.info('get_ports_wa_polygons - START')

    # TODO: fill in mongo query here
    if not import_path:
        col = db["polygons"]

        col = col.find({'properties.areaType': {'$in': ['Port', 'PortWaitingArea']}},
                       {'_id': 1, 'properties.title':  1, 'geometry': 1})

        polygons_df = pd.DataFrame(list(col))
        polygons_df = polygons_df.rename(columns={'properties.areaType': 'areaType', 'properties.title': 'title'})

    else:
        polygons_file_path = os.path.join(import_path, 'polygons.json')
        with open(polygons_file_path, 'r') as polygons_file:
            polygons_file = json.load(polygons_file)
        polygons_df = pd.DataFrame.from_dict(polygons_file, orient='index')
        polygons_df['_id'] = polygons_df.index
        polygons_df.reset_index(drop=True, inplace=True)
        polygons_df['title'] = [d.get('title') for d in polygons_df.properties]  # add name column from nested dict
        polygons_df['areaType'] = [d.get('areaType') for d in
                                   polygons_df.properties]  # add areaType column from nested dict
        polygons_df = polygons_df[['_id', 'geometry', 'title', 'areaType']]

    polygons_df = polygons_df.rename(columns={'_id': 'polygon_id',
                                              'areaType': 'polygon_area_type',
                                              'title': 'name'})
    polygons_df['geometry'] = polygons_df['geometry'].apply(shape)  # convert to shape object

    polygons_df = gpd.GeoDataFrame(polygons_df)  # convert to GeoPandas

    logging.info('get_ports_wa_polygons - END')

    return polygons_df


def get_vessels_info(import_path, db, vessels_ids):

    logging.info('get_vessels_info - START')

    projection = {'vesselId': 1, "class_calc": 1, "subclass_documented": 1, "built_year": 1,
                  "name": 1, "deadweight": 1, "draught": 1, "size": 1, "age": 1}

    # TODO: fill in mongo query here
    if not import_path:
        col = db["vessels"]
        col = col.find({'_id': {'$in': [ObjectId(vessel) for vessel in vessels_ids]}}, projection)

        vessels_df = pd.DataFrame(list(col)).drop(['_id'], axis=1)

    else:

        vessels_file_path = os.path.join(import_path, 'vessels.json')
        with open(vessels_file_path, 'r') as vessels_file:
            vessels_file = json.load(vessels_file)
            vessels_df = pd.DataFrame.from_dict(vessels_file, orient='index')
            vessels_df = vessels_df[projection].drop('vesselId', axis=1)

    conditions = [(vessels_df["class_calc"] == 'Cargo') & (vessels_df["subclass_documented"] == 'Container Vessel'),
                  (vessels_df["class_calc"] == 'Cargo') & (vessels_df["subclass_documented"] != 'Container Vessel'),
                  (vessels_df["class_calc"] == 'Tanker')]

    choices = ["cargo_container", "cargo_other", "tanker"]
    vessels_df["class_new"] = np.select(conditions, choices)
    vessels_df["class_new"] = vessels_df["class_new"].replace({'0': 'other'})

    vessels_df = vessels_df.add_prefix('vessel_')
    vessels_df = vessels_df.reset_index().rename(columns={'index': 'vesselId'})
    if vessels_ids:
        vessels_df = vessels_df[vessels_df['vesselId'].isin(vessels_ids)]

    logging.info('get_vessels_info - END')

    return vessels_df


def get_ports_info(import_path, db):

    logging.info('get_ports_info - START')

    # TODO: fill in mongo query here
    if not import_path:
        col = db["ports"]
        col = col.find({}, {'_id': 1, 'country': 1, 'name': 1, 'center_coordinates.0': 1, 'center_coordinates.1': 1})
        ports_df = pd.DataFrame(list(col)).drop(['_id'], axis=1)
        ports_df = ports_df.rename(columns={'center_coordinates.0': 'lng', 'center_coordinates.1': 'lat'})

    else:
        ports_file_path = os.path.join(import_path, 'ports.json')
        with open(ports_file_path, 'r') as portsfile:

            ports_df = json.load(portsfile)
        ports_df = pd.DataFrame.from_dict(ports_df, orient='index')
        ports_df = ports_df[['country', 'name', 'center_coordinates']]

        ports_df['lat'] = ports_df.center_coordinates.map(lambda x: x[1])
        ports_df['lng'] = ports_df.center_coordinates.map(lambda x: x[0])

        ports_df = ports_df.reset_index().rename(columns={'index': 'PortId'}).drop('center_coordinates', axis=1)

        logging.info('get_ports_info - END')

    return ports_df


def get_activity_df(import_path, db, vessels_ids, activity='mooring', nrows=None):

    logging.info(F'get_activity_df ({activity}) - START')

    # TODO: fill in mongo query here
    if not import_path:
        query = {'vesselId': {'$in': [ObjectId(vessel) for vessel in vessels_ids]}}
        projection = {'_id': 1, 'vesselId': 1, 'startDate': 1, 'endDate': 1, 'duration': 1, 'nextPort': 1,
                      'firstBlip.geometry.coordinates.0': 1, 'firstBlip.geometry.coordinates.1': 1,
                      'lastBlip.geometry.coordinates.0': 1, 'lastBlip.geometry.coordinates.1': 1}

        col = db[activity]
        col = col.find(query, projection).limit(nrows) if nrows else col.find(query, projection)
        activity_df = pd.DataFrame(list(col))
        activity_df = activity_df.rename(columns={'firstBlip.geometry.coordinates.0': 'firstBlip_lng',
                                                  'firstBlip.geometry.coordinates.1': 'firstBlip_lat',
                                                  'lastBlip.geometry.coordinates.0': 'lastBlip_lng',
                                                  'lastBlip.geometry.coordinates.1': 'lastBlip_lat'})

    else:
        cols = ['_id', 'firstBlip', 'lastBlip', 'vesselId', 'startDate', 'endDate', 'duration', 'nextPort']
        activity_file_path = os.path.join(import_path, f'{activity}.csv.gz')
        activity_df = pd.read_csv(activity_file_path, nrows=nrows, usecols=cols)
        activity_df = extract_coordinates(activity_df, 'firstBlip')
        activity_df = extract_coordinates(activity_df, 'lastBlip')
        activity_df = activity_df.drop(['firstBlip', 'lastBlip'], axis=1)

    logging.info(F'get_activity_df ({activity}) - END')

    return activity_df


def main(export_path, vessels_ids=None, import_path=None, use_db=False, debug=True):

    """
    This code will get all relevant data and prepare it for clustering
    :param export_path: path in which the output will be exported
    :param import_path: path to directory with all relevant files
    :param vessels_ids: comma-separated list of vessels ids for mongo query
    :param use_db: if True, will use mongo db to query data
    :param debug: if True, only a first 10K rows of each file will be processed
    :return:
    """

    assert use_db or import_path, "use_db or import_path wasn't passed"

    if vessels_ids and isinstance(vessels_ids, str):
        vessels_ids = vessels_ids.split(',')

    db = None
    if use_db:
        myclient = pymongo.MongoClient("<your connection string here>")  # initiate MongoClient for mongo queries
        db = myclient["<DB name here>"]

    nrows = 10000 if debug else None  # 10K rows per file if debug mode == True

    polygons_df = get_ports_wa_polygons(import_path, db)
    vessels_df = get_vessels_info(import_path, db, vessels_ids)
    ports_df = get_ports_info(import_path, db)

    results_dict = {'anchoring': None, 'mooring': None}

    for activity in ['anchoring', 'mooring']:

        activity_df = get_activity_df(import_path, db, vessels_ids, activity, nrows)

        activity_df = find_intersection_with_polygons(activity_df, polygons_df, 'firstBlip')
        activity_df = find_intersection_with_polygons(activity_df, polygons_df, 'lastBlip')
        activity_df = add_dist_from_nearest_port(activity_df, ports_df)

        activity_df = is_in_polygon_features(activity_df)

        logging.info(f'merging vessels data...')
        activity_df = activity_df.merge(vessels_df, on='vesselId', how='left')

        logging.info(f'merging nextPort data...')
        activity_df = activity_df.merge(polygons_df.set_index('polygon_id').drop('geometry', axis=1),
                                        left_on='nextPort', right_index=True, how='left').rename(
            columns={'name': 'nextPort_name'})
        activity_df.nextPort_name.fillna('UNKNOWN', inplace=True)

        results_dict[activity] = activity_df
        activity_df.to_csv(os.path.join(export_path, f'{activity}.csv.gz'), index=False, compression='gzip')

    return results_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
