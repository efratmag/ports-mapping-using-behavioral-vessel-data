import os
from enum import Enum
import pandas as pd
import numpy as np
import json
import logging
from bson import ObjectId
from shapely.geometry import shape
import geopandas as gpd
import pymongo


class VesselType(Enum):
    """constant parameter for vessels type"""
    CARGO_CONTAINER = "cargo_container"
    CARGO_OTHER = "cargo_other"
    TANKER = "tanker"


def get_ww_polygons(import_path: str, db: pymongo.MongoClient = None) -> gpd.GeoDataFrame:

    """
    function that extract ports & waiting areas polygons
    :param import_path: path to the ports & waiting areas polygons file location
    :param db: MongoDB object
    :return:
    """

    # TODO: fill in mongo query
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

    logging.info('polygons data extracted')

    return polygons_df


def get_vessels_info(import_path: str, db: pymongo.MongoClient, vessels_ids: list = None) -> pd.DataFrame:

    """
    function that extract vessels info
    :param import_path: path to the vessels info file location
    :param db: MongoDB object
    :param vessels_ids: filter data by specific vessels_ids
    :return:
    """

    columns_projection = {"vesselId": 1, "class_calc": 1, "subclass_documented": 1, "built_year": 1,
                          "name": 1, "deadweight": 1, "draught": 1, "size": 1, "age": 1}

    # TODO: fill in mongo query
    if not import_path:
        col = db["vessels"]
        col = col.find({'_id': {'$in': [ObjectId(vessel) for vessel in vessels_ids]}} if vessels_ids else {}, columns_projection)

        vessels_df = pd.DataFrame(list(col)).drop(['_id'], axis=1)

    else:

        vessels_file_path = os.path.join(import_path, 'vessels.json')
        with open(vessels_file_path, 'r') as vessels_file:
            vessels_file = json.load(vessels_file)
            vessels_df = pd.DataFrame.from_dict(vessels_file, orient='index')

            if vessels_ids:
                vessels_df = vessels_df[vessels_df.index.isin(vessels_ids)]

            vessels_df = vessels_df[columns_projection].drop('vesselId', axis=1)

    conditions = [(vessels_df["class_calc"] == "Cargo") & (vessels_df["subclass_documented"] == "Container Vessel"),
                  (vessels_df["class_calc"] == "Cargo") & (vessels_df["subclass_documented"] != "Container Vessel"),
                  (vessels_df["class_calc"] == "Tanker")]

    choices = [VesselType.CARGO_CONTAINER.value, VesselType.CARGO_OTHER.value, VesselType.TANKER.value]
    vessels_df["class_new"] = np.select(conditions, choices)
    vessels_df["class_new"] = vessels_df["class_new"].replace({'0': 'other'})

    vessels_df = vessels_df.add_prefix('vessel_')
    vessels_df = vessels_df.reset_index().rename(columns={'index': 'vesselId'})

    logging.info('vessels data extracted')

    return vessels_df
