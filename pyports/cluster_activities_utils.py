from pyports.geo_utils import *
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime
import os
import logging

from pyports.get_metadata import VesselType


def get_data_for_clustering(import_path, type_of_area_mapped, activity, debug, sub_area_polygon_fname, blip, only_container_vessels):

    # TODO: insert communication with db for other dfs (ports_df, polygons_df, vessels_df)
    # TODO: add shoreline needed files to ww

    logging.info('loading data for clustering - START')

    df_for_clustering_fname = f'features/df_for_clustering_{activity}.csv'

    nrows = 10000 if debug else None  # will load first 10K rows if debug == True

    logging.info('loading activity data...')

    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False, nrows=nrows)

    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    if type_of_area_mapped == 'pwa':
        # if destination based clustering for pwa then include only activities with known destination
        df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
        df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
        df.reset_index(drop=True, inplace=True)  # reset index

    if only_container_vessels:
        df = df[df.vessel_class_new == VesselType.CARGO_CONTAINER.value]  # take only container vessels

    logging.info('loading ports data...')
    ports_df = gpd.read_file(os.path.join(import_path, 'maps/ports.geojson'))  # WW ports
    ports_df.drop_duplicates(subset='name', inplace=True)

    logging.info('loading polygons data...')
    polygons_df = gpd.read_file(os.path.join(import_path, 'maps/polygons.geojson'),
                                usecols=['_id', 'title', 'areaType', 'geometry'])  # WW polygons
    # TODO: find out why still get error: WARNING:fiona.ogrext:Skipping field otherNames: invalid type 5

    logging.info('loading shoreline data...')
    shoreline_df = gpd.read_file(os.path.join(import_path, 'maps/shoreline_layer.geojson'))  # shoreline layer
    main_land = merge_polygons(shoreline_df[:4])  # create multipolygon of the big continents
    shoreline_polygon = merge_polygons(shoreline_df)  # merging shoreline_df to one multipolygon

    logging.info('loading data for clustering - END')

    return df, ports_df, polygons_df, main_land, shoreline_polygon


def polygenize_clusters_with_features(type_of_area_mapped, df_for_clustering, polygons_df, main_land, blip,
                                      optimize_polygon, alpha, polygon_type, shoreline_distance_method='haversine',
                                      shoreline_polygon=None, ports_df=None, only_container_vessels=None):

    """
    :param type_of_area_mapped: 'ports' or 'pwa' (ports waiting areas).
    :param df_for_clustering: activities dataframe with clustering results.
    :param polygons_df: dataframe of WW polygons.
    :param main_land: multi-polygon of the main continents.
    :param blip: as in main- 'first' or 'last'.
    :param optimize_polygon: if True, will apply optimize_polygon.
    :param alpha: as in main- parameter for alpha shape- degree of polygon segregation.
    :param polygon_type: 'alpha_shape' or 'convexhull'.
    :param shoreline_distance_method - "euclidean" / "haversine". only relevant if type_of_area_mapped=='ports'.
    :param shoreline_polygon: merged shoreline layer to one multipolygon. only relevant if type_of_area_mapped=='ports'.
    :param ports_df: dataframe of WW ports. only relevant if type_of_area_mapped=='ports'.
    :param only_container_vessels: boolean, only relevant if type_of_area_mapped=='pwa'.
    :return: geopandas dataframe of all polygenized clusters with their features.
    """

    logging.info('starting feature extraction for clusters...')

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

    ww_polygons_centroids = np.array([polygons_df.geometry.centroid.y, polygons_df.geometry.centroid.x]).T

    cluster_polygons = []

    for cluster in tqdm(df_for_clustering.cluster_label.unique()):  # iterate over clusters
        record = {}
        cluster_df = df_for_clustering[df_for_clustering.cluster_label == cluster]  # sub-df for chosen cluster
        points = cluster_df[[f'{blip}Blip_lng', f'{blip}Blip_lat']].to_numpy()  # numpy array of lng/lat

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

        record['label'] = f'cluster_{cluster}'  # cluster label

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
        record['centroid_lng'] = polygon.centroid.x  # longitude of polygon centroid
        record['pct_intersection'] = polygon_intersection(polygon, polygons_df, type_of_area_mapped)  # percent intersection with WW polygons
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, ww_polygons_centroids)  # distance from
        # TODO: find out why getting warning: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are
        #  likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
        # nearest WW polygon
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

    logging.info('finished extracting polygons features!')

    return cluster_polygons


def save_data(type_of_area_mapped, polygenized_clusters_geodataframe, export_path):
    """
    saves geojson and csv versions of the mapped areas.

    :param type_of_area_mapped: 'ports' or 'pwa'
    :param polygenized_clusters_geodataframe: output of polygenize_clusters_with_features.
    :param export_path: the path to save the files.
    """

    logging.info('saving data...')

    fname = f'{type_of_area_mapped}_polygons_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    polygenized_clusters_geodataframe.to_file(os.path.join(export_path, fname + '.geojson'), driver="GeoJSON")
    polygenized_clusters_geodataframe.to_csv(os.path.join(export_path, fname + '.csv'))

    logging.info('finished saving data!')

