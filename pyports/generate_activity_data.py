"""
Get activity data from database/ import_path and extract needed features.
"""

from shapely.geometry import Point
import os
import fire
import pandas as pd
import geopandas as gpd
import pymongo
from bson import ObjectId
import logging
from typing import List

from pyports.constants import ACTIVITY
from pyports.get_metadata import get_vessels_info, get_ww_polygons


def extract_coordinates(df: pd.DataFrame, blip: str) -> pd.DataFrame:

    """
    a function that extracts lat and lng from a Series with geometry dict
    :param df: Dataframe with coordinates dict columns
    :param blip: "firstBlip" / "lastBlip"
    :return: df with lat lng coordinates
    """

    coordinates_df = df[blip].dropna().apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series) # extract geometry from nested dict
    coordinates_df = coordinates_df.rename(columns={0: blip + '_lng', 1: blip + '_lat'})

    df = df.merge(coordinates_df, left_index=True, right_index=True, how='left')

    return df


def find_intersection_with_polygons(df: pd.DataFrame, polygons_df: gpd.GeoDataFrame, blip: str) -> gpd.GeoDataFrame:

    """
    this function will find for each activity point its ww polygon id and type (if an intersection will be found)
    :param df: activity df
    :param polygons_df: ww polygons df
    :param blip: "firstBlip" / "lastBlip"
    :return: activity df with additional columns for ww polygon id and type
    """

    df['geometry'] = df.apply(lambda x: Point(x[blip + '_lng'], x[blip + '_lat']) if not pd.isna(
        x[blip + '_lat']) else None, axis=1)  # convert lat,lng to Point object

    df = gpd.GeoDataFrame(df)

    df = gpd.sjoin(df, polygons_df[['polygon_id', 'polygon_area_type', 'geometry'
                                    ]], how='left').drop('index_right', axis=1)  # spatial join (left: activity points & right: ww polygons)

    df['polygon_id'], df['polygon_area_type'] = df['polygon_id'].astype(str), df['polygon_area_type'].astype(str) # convert to string

    merged_polygons = df.groupby('_id', as_index=False).agg({'polygon_id': ', '.join,  # create df with aggregated to polygons_ids & types
                                                             'polygon_area_type': ', '.join}).replace({'nan': None})

    df = df.drop_duplicates('_id').drop(['polygon_id', 'polygon_area_type', 'geometry'], axis=1)  # drop duplicate rows caused by sjoin
    df = df.merge(merged_polygons, on='_id')  # merge polygons intersections
    df = df.rename(columns={'polygon_id': blip + '_polygon_id',
                            'polygon_area_type': blip + '_polygon_area_type'})  # rename in respect to blip

    return df


def get_activity_df(import_path: str, db: pymongo.MongoClient, vessels_ids: List[str] = None,
                    activity: ACTIVITY = ACTIVITY.MOORING, debug: bool = False) -> pd.DataFrame:

    """
    this function will load the activity data
    :param import_path: path to location of activity file: mooring.csv.gz / anchoring.csv.gz
    :param db: MongoDB object
    :param vessels_ids: filter data by specific vessels_ids
    :param activity: "mooring" / "anchoring"
    :param debug: if True, only first 10K rows of activity df will be loaded
    :return:
    """

    nrows = 10000 if debug else None

    # TODO: fill in mongo query
    if not import_path:
        query = {'vesselId': {'$in': [ObjectId(vessel) for vessel in vessels_ids]}} if vessels_ids else {}
        columns_projection = {'_id': 1, 'vesselId': 1, 'startDate': 1, 'endDate': 1, 'duration': 1, 'nextPort': 1,
                              'firstBlip.geometry.coordinates.0': 1, 'firstBlip.geometry.coordinates.1': 1,
                              'lastBlip.geometry.coordinates.0': 1, 'lastBlip.geometry.coordinates.1': 1}

        col = db[activity.value]
        col = col.find(query, columns_projection).limit(nrows) if nrows else col.find(query, columns_projection)
        activity_df = pd.DataFrame(list(col))
        activity_df = activity_df.rename(columns={'firstBlip.geometry.coordinates.0': 'firstBlip_lng',
                                                  'firstBlip.geometry.coordinates.1': 'firstBlip_lat',
                                                  'lastBlip.geometry.coordinates.0': 'lastBlip_lng',
                                                  'lastBlip.geometry.coordinates.1': 'lastBlip_lat'})

    else:
        cols = ['_id', 'firstBlip', 'lastBlip', 'vesselId', 'startDate', 'endDate', 'duration', 'nextPort']
        activity_file_path = os.path.join(import_path, f'{activity.value}.csv.gz')
        activity_df = pd.read_csv(activity_file_path, nrows=nrows, usecols=cols)

        if vessels_ids:
            activity_df = activity_df[activity_df['vesselId'].isin(vessels_ids)]  # filter data by specific vessels_ids

        activity_df = extract_coordinates(activity_df, 'firstBlip')
        activity_df = extract_coordinates(activity_df, 'lastBlip')
        activity_df = activity_df.drop(['firstBlip', 'lastBlip'], axis=1)  # drop nested locations columns

    logging.info(f'{activity.value} data extracted')

    return activity_df


def main(export_path: str, activity: ACTIVITY = ACTIVITY.MOORING, vessels_ids: str = None,
         import_path: str = None, use_db: bool = None, debug: bool = True):

    """
    This code will get all relevant data and prepare it for clustering
    :param export_path: path in which the output will be exported
    :param activity: "mooring" / "anchoring"
    :param import_path: path to directory with all relevant files: polygons.json, vessels.json, anchoring.csv.gz, mooring.csv.gz
    :param vessels_ids: comma-separated string of vessels ids for mongo query
    :param use_db: if True, will use mongo db to query data
    :param debug: if True, only a first 10K rows of each file will be processed for the activity file
    :return:
    """

    activity = activity.value if isinstance(activity, ACTIVITY) else activity  # parse activity value

    assert activity in ["mooring", "anchoring"], 'activity must be "mooring" or "anchoring"'
    assert use_db or import_path, "use_db or import_path wasn't passed"

    if vessels_ids:
        vessels_ids = vessels_ids.split(',')  # split comma-separated string to list

    # TODO: update with winward querying methods
    db = None
    if use_db:
        myclient = pymongo.MongoClient("<your connection string here>")  # initiate MongoClient for mongo queries
        db = myclient["<DB name here>"]

    polygons_df = get_ww_polygons(import_path, db)  # WW polygons
    vessels_df = get_vessels_info(import_path, db, vessels_ids)  # WW vessels info

    activity_df = get_activity_df(import_path, db, vessels_ids, activity, debug)  # vessels activity

    activity_df = find_intersection_with_polygons(activity_df, polygons_df, 'firstBlip')  # enrich activity_df with polygons info
    activity_df = find_intersection_with_polygons(activity_df, polygons_df, 'lastBlip') # enrich activity_df with polygons info

    activity_df = activity_df.merge(vessels_df, on='vesselId', how='left')  # enrich activity_df with vessels info

    activity_df = activity_df.merge(polygons_df.set_index('polygon_id').drop('geometry', axis=1), # enrich with activity_df df with nextPort info
                                    left_on='nextPort', right_index=True, how='left').rename(
        columns={'name': 'nextPort_name'})
    activity_df.nextPort_name.fillna('UNKNOWN', inplace=True)

    activity_df.to_csv(os.path.join(export_path, f'{activity.value}.csv.gz'), index=False, compression='gzip')

    return activity_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
