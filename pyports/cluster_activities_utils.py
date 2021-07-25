import pathlib

from pyports.geo_utils import *
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import pymongo
from shapely.geometry import Point
import datetime
import os
from typing import Union, Tuple

from pyports.get_metadata import get_ww_polygons, get_ports_info, get_shoreline_layer
from pyports.constants import ACTIVITY, AreaType, VesselType


def get_data_for_clustering(import_path: str, type_of_area_mapped: Union[AreaType, str], activity: Union[ACTIVITY, str],
                            blip: str, only_container_vessels: bool, sub_area_polygon_fname: str = None,
                            use_db: bool = False, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                                gpd.GeoDataFrame, MultiPolygon,
                                                                                MultiPolygon]:

    """
    this function will get all needed data for clustering.
    :param import_path:path to all used files.
    :param type_of_area_mapped: "ports" / "pwa" (ports waiting areas).
    :param activity: "mooring" / "anchoring".
    :param blip: "first" / "last".
    :param only_container_vessels: use only container vessels for pwa mapping.
    :param sub_area_polygon_fname: path to geojson file with polygon for area sub-setting.
    :param use_db: if True, will use mongo db to query data.
    :param debug: if True, only a first 10K rows of each file will be processed for the activity file.
    :return: activity dataframe, ports dataframe, polygons dataframe, multipolygon of the main continents, multipolygon
     of shoreline
    """

    # parsing input values
    activity = activity.value if isinstance(activity, ACTIVITY) else activity
    type_of_area_mapped = type_of_area_mapped.value if isinstance(type_of_area_mapped, AreaType) else \
        type_of_area_mapped

    # TODO: add shoreline needed files to ww
    # TODO: update with winward querying methods
    db = None
    if use_db:
        myclient = pymongo.MongoClient("<your connection string here>")  # initiate MongoClient for mongo queries
        db = myclient["<DB name here>"]

    df_for_clustering_fname = f'df_for_clustering_{activity}.csv.gz'

    nrows = 10000 if debug else None  # will load first 10K rows if debug == True

    # TODO: need to add condition to generate df_for_clustering if not exist
    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False, nrows=nrows)

    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lon'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    if type_of_area_mapped == 'pwa':
        # if destination based clustering for pwa then include only activities with known destination
        df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
        df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
        df.reset_index(drop=True, inplace=True)  # reset index  #TODO: FIX!this line does not seem to work!

    if type_of_area_mapped == 'ports':
        df = df[df.vessel_class_new != 'other']
        # TODO: generalize vessel class selection

    if only_container_vessels:
        df = df[df.vessel_class_new == VesselType.CARGO_CONTAINER.value]  # take only container vessels
        df.reset_index(drop=True, inplace=True)

    df = df.drop_duplicates(subset=['firstBlip_lon', 'firstBlip_lat'])  # drop duplicates

    ports_df = get_ports_info(import_path, db)
    polygons_df = get_ww_polygons(import_path, db)  # WW polygons
    main_land, shoreline_polygon = get_shoreline_layer(import_path, db)
    # TODO: find out why still get error: WARNING:fiona.ogrext:Skipping field otherNames: invalid type 5

    return df, ports_df, polygons_df, main_land, shoreline_polygon


def polygenize_clusters_with_features(type_of_area_mapped: Union[AreaType, str], df_for_clustering: pd.DataFrame,
                                      polygons_df: gpd.GeoDataFrame, ports_df: pd.DataFrame, main_land: MultiPolygon,
                                      shoreline_polygon: MultiPolygon, blip: str, optimize_polygon: bool, alpha: int,
                                      polygon_type: str, shoreline_distance_method: str = 'haversine',
                                      only_container_vessels: bool = None) -> gpd.GeoDataFrame:

    """
    :param type_of_area_mapped: 'ports' or 'pwa' (ports waiting areas).
    :param df_for_clustering: activities dataframe with clustering results.
    :param polygons_df: dataframe of WW polygons.
    :param ports_df: dataframe of WW ports. only relevant if type_of_area_mapped=='ports'.
    :param main_land: multi-polygon of the main continents.
    :param shoreline_polygon: merged shoreline layer to one multipolygon. only relevant if type_of_area_mapped=='ports'.
    :param blip: as in main- 'first' or 'last'.
    :param optimize_polygon: if True, will apply optimize_polygon.
    :param alpha: as in main- parameter for alpha shape- degree of polygon segregation.
    :param polygon_type: 'alpha_shape' or 'convexhull'.
    :param shoreline_distance_method - "euclidean" / "haversine". only relevant if type_of_area_mapped=='ports'.
    :param only_container_vessels: boolean, only relevant if type_of_area_mapped=='pwa'.
    :return: geopandas dataframe of all polygenized clusters with their features.
    """

    # parsing input values
    type_of_area_mapped = type_of_area_mapped.value if isinstance(type_of_area_mapped, AreaType) else \
        type_of_area_mapped

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

    ww_polygons_centroids = np.array([polygons_df.geometry.centroid.y, polygons_df.geometry.centroid.x]).T

    cluster_polygons = []

    for cluster in tqdm(df_for_clustering.cluster_label.unique()):  # iterate over clusters
        record = {}
        cluster_df = df_for_clustering[df_for_clustering.cluster_label == cluster]  # sub-df for chosen cluster
        points = cluster_df[[f'{blip}Blip_lon', f'{blip}Blip_lat']].to_numpy()  # numpy array of lon/lat

        if optimize_polygon:
            probs = cluster_df['cluster_probability'].to_numpy()
            polygon, original_polygon, min_prob_knee, points_removed, metrics = \
                optimize_polygon_by_probs(points, probs, polygon_type, alpha=alpha, s=1)
            record['min_prob_knee'] = min_prob_knee
            record['points_removed'] = points_removed
            record['points_removed_pct'] = points_removed / len(points)
            record['original_polygon'] = gpd.GeoSeries(original_polygon).to_json()
            record['polygon_optimization_curve'] = ', '.join([str(i) for i in metrics])
        else:
            polygon = polygon_from_points(points, polygon_type, alpha)  # create polygon from points

        record['label'] = f'cluster_{int(cluster)}'  # cluster label
        if type_of_area_mapped == 'pwa':  # get destination port name
            record['destination_port'] = cluster_df['nextPort_name'].mode()[0]
        record['probs_of_belonging_to_cluster'] = \
            ', '.join(cluster_df['cluster_probability'].astype(str).to_list())  # list of all points probabilities of
        # belonging to the cluster
        record['mean_prob_of_belonging_to_cluster'] = cluster_df['cluster_probability'].mean()  # mean probability of
        # belonging to cluster
        record['geometry'] = polygon  # polygenized cluster
        record['geojson'] = gpd.GeoSeries(polygon).to_json()  # geojson format of the polygenized cluster
        record['num_points'] = cluster_df.shape[0]  # total number of points in cluster
        record['area_sqkm'] = calc_polygon_area_sq_unit(polygon)  # polygon area in sqkm
        record['density'] = calc_cluster_density(points)  # polygon density (1/mean squared distance)
        record['mean_duration'] = cluster_df['duration'].mean()  # mean duration from first blip to last blip
        record['median_duration'] = cluster_df['duration'].median()  # median duration from first blip to last blip
        record['n_unique_vesselID'] = cluster_df['vesselId'].nunique()  # number of unique vessel IDs in cluster
        record['percent_unique_vesselID'] = cluster_df['vesselId'].nunique() / len(points)  # percent of unique
        # vesselIDs in cluster
        record['vesselIDs'] = ', '.join(cluster_df['vesselId'].astype(str).to_list())  # list of all vessel IDs in
        # cluster
        if not only_container_vessels:
            # only measure vessel type variance if there is more than one vessel type
            record['most_freq_vessel_type'] = cluster_df['vessel_class_new'].mode()[0]  # most frequent vessel type in
            # cluster
            record['vessel_type_variance'] = calc_entropy(cluster_df['vessel_class_new'])  # variance of vessel type in
            # cluster
        record['is_in_river'] = polygon.within(main_land)  # is the polygon in rivers (True) or in the sea/ocean (False)
        record['centroid_lat'] = polygon.centroid.y  # latitude of polygon centroid
        record['centroid_lon'] = polygon.centroid.x  # longitude of polygon centroid
        record['pct_intersection'] = polygon_intersection(polygon, polygons_df, type_of_area_mapped)  # percent
        # intersection with WW polygons
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, ww_polygons_centroids)
        # distance from nearest WW polygon
        # TODO: find out why getting warning: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are
        #  likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
        if type_of_area_mapped == 'ports':
            # calculate distance from nearest port and from shoreline only for ports for later filtering
            record['distance_from_nearest_port'], record['name_of_nearest_port'] = \
                calc_polygon_distance_from_nearest_port(polygon, ports_df)
            distance_from_shore = calc_nearest_shore(polygon, shoreline_polygon, shoreline_distance_method)  # distance
            # in Km and point of nearest_shore
            record.update(distance_from_shore)
        record['link_to_google_maps'] = create_google_maps_link_to_centroid(polygon.centroid)  # link to google maps for
        # the polygon centroid

        cluster_polygons.append(record)

    cluster_polygons = gpd.GeoDataFrame(cluster_polygons)

    return cluster_polygons


def save_data(type_of_area_mapped: Union[AreaType, str], polygenized_clusters_geodataframe: gpd.GeoDataFrame,
              export_path: pathlib.Path):
    """
    saves geojson and csv versions of the mapped areas.

    :param type_of_area_mapped: 'ports' or 'pwa'
    :param polygenized_clusters_geodataframe: output of polygenize_clusters_with_features.
    :param export_path: the path to save the files.
    """

    fname = f'{type_of_area_mapped}_polygons_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    polygenized_clusters_geodataframe.to_file(os.path.join(export_path, fname + '.geojson'), driver="GeoJSON")
    polygenized_clusters_geodataframe.to_csv(os.path.join(export_path, fname + '.csv'))
