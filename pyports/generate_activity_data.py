"""
Get activity data from database/ import_path and extract needed features.
"""

import os
import fire
import pandas as pd
import pymongo
from bson import ObjectId
import logging
from typing import List, Union

from pyports.constants import ACTIVITY
from pyports.get_metadata import get_vessels_info, get_ww_polygons


def extract_coordinates(df: pd.DataFrame, blip: str) -> pd.DataFrame:

    """
    a function that extracts lat and lon from a Series with geometry dict
    :param df: Dataframe with coordinates dict columns
    :param blip: "firstBlip" / "lastBlip"
    :return: df with lat lon coordinates
    """

    coordinates_df = df[blip].dropna().apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series) # extract geometry from nested dict
    coordinates_df = coordinates_df.rename(columns={0: blip + '_lon', 1: blip + '_lat'})

    df = df.merge(coordinates_df, left_index=True, right_index=True, how='left')

    return df


def get_activity_df(import_path: str, db: pymongo.MongoClient, vessels_ids: List[str] = None,
                    activity: ACTIVITY = ACTIVITY.ANCHORING.value, debug: bool = False) -> pd.DataFrame:

    """
    Load the activity data and extract lat lon.
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

        col = db[activity]
        col = col.find(query, columns_projection).limit(nrows) if nrows else col.find(query, columns_projection)
        activity_df = pd.DataFrame(list(col))
        activity_df = activity_df.rename(columns={'firstBlip.geometry.coordinates.0': 'firstBlip_lon',
                                                  'firstBlip.geometry.coordinates.1': 'firstBlip_lat',
                                                  'lastBlip.geometry.coordinates.0': 'lastBlip_lon',
                                                  'lastBlip.geometry.coordinates.1': 'lastBlip_lat'})

    else:
        cols = ['_id', 'firstBlip', 'lastBlip', 'vesselId', 'startDate', 'endDate', 'duration', 'nextPort']
        activity_file_path = os.path.join(import_path, f'{activity}.csv.gz')
        activity_df = pd.read_csv(activity_file_path, nrows=nrows, usecols=cols)

        if vessels_ids:
            activity_df = activity_df[activity_df['vesselId'].isin(vessels_ids)]  # filter data by specific vessels_ids

        activity_df = extract_coordinates(activity_df, 'firstBlip')
        activity_df = extract_coordinates(activity_df, 'lastBlip')
        activity_df = activity_df.drop(['firstBlip', 'lastBlip'], axis=1)  # drop nested locations columns

    logging.info(f'{activity} data extracted')

    return activity_df


def main(export_path: str, activity: Union[ACTIVITY, str] = ACTIVITY.ANCHORING, vessels_ids: str = None,
         import_path: str = None, use_db: bool = None, debug: bool = False):

    """
    Generate dataframe for clustering and save it to export path.
    :param export_path: path in which the output will be exported
    :param activity: "mooring" / "anchoring"
    :param import_path: path to directory with all relevant files: polygons.json, vessels.json, anchoring.csv.gz, mooring.csv.gz
    :param vessels_ids: comma-separated string of vessels ids for mongo query
    :param use_db: if True, will use mongo db to query data
    :param debug: if True, only a first 10K rows of each file will be processed for the activity file
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
    # enrich activity_df with vessels info
    activity_df = activity_df.merge(vessels_df, on='vesselId', how='left')
    # enrich with activity_df df with nextPort info
    activity_df = activity_df.merge(polygons_df.set_index('polygon_id').drop('geometry', axis=1),
                                    left_on='nextPort', right_index=True, how='left').rename(columns={'name': 'nextPort_name'})
    # fill missing nextPort values with 'unknown'
    activity_df.nextPort_name.fillna('UNKNOWN', inplace=True)
    # save activity dataframe to csv
    activity_df.to_csv(os.path.join(export_path, f'df_for_clustering_{activity}.csv.gz'), index=False, compression='gzip')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
