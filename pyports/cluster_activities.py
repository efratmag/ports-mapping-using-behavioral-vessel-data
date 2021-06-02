import os
import hdbscan
import pickle
import fire
from shapely.geometry import Point
from pyports.geo_utils import *
from pyports.rank_ports_candidates import main as rank_candidates
from tqdm import tqdm
from kneed import KneeLocator
import numpy as np


def optimize_polygon_by_probs(points, probs, alpha=4, s=1):

    # TODO: Optimize function to run more efficiently

    all_prob = np.linspace(min(probs), 1, 20)
    metrics = []

    for prob in all_prob:
        probs_mask = probs >= prob
        relevant_points = points[probs_mask]
        if len(relevant_points) > 0:
            poly, _, _ = alpha_shape(relevant_points, alpha=alpha)

            area_size = calc_polygon_area_sq_unit(poly)
            metrics.append(area_size)

    if metrics[0] == metrics[1]:
        metrics, all_prob = metrics[1:], all_prob[1:]

    kneedle = KneeLocator(all_prob, metrics, S=s, curve="convex", direction="decreasing")

    if kneedle.knee:

        final_points = points[probs >= kneedle.knee]
        points_removed = len(points) - len(final_points)
        poly, _, _ = alpha_shape(final_points, alpha=alpha)
    else:
        poly = polygon_from_points(points, 'alpha_shape', alpha)
        points_removed = 0

    return poly, kneedle.knee, points_removed


def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df, polygons_df, shoreline_polygon,
                                      main_land,
                                      activity,
                                      blip,
                                      optimize_polygon,
                                      alpha=4,
                                      shoreline_distance_method='euclidean',
                                      polygon_type='alpha_shape'):
    """
    :param df_for_clustering: activities dataframe with clustering results
    :param ports_df: dataframe of WW ports
    :param polygons_df: dataframe of WW polygons
    :param shoreline_polygon: merged shoreline layer to one multipolygon
    :param main_land: multi-polygon of the main continents
    :param activity: qs in main- activity type (i.e mooring/anchoring etc.)
    :param blip: as in main- 'first' or 'last'
    :param optimize_polygon: if True, will apply optimize_polygon
    :param alpha: as in main- parameter for alpha shape- degree of polygon segregation
    :param shoreline_distance_method - "euclidean" / "haversine"
    :param polygon_type: 'alpha_shape' or 'convexhull'
    :return: geopandas dataframe of all polygenized clusters with their features
    """

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

    cluster_polygons = []

    for cluster in tqdm(df_for_clustering.cluster_label.unique()):  # iterate over clusters
        record = {}
        cluster_df = df_for_clustering[df_for_clustering.cluster_label == cluster]  # sub-df for chosen cluster
        points = cluster_df[[f'{blip}Blip_lng', f'{blip}Blip_lat']].to_numpy()  # numpy array of lng/lat

        if optimize_polygon:
            probs = cluster_df['cluster_probability'].to_numpy()
            polygon, min_prob_knee, points_removed = optimize_polygon_by_probs(points, probs, alpha=alpha, s=1)
            record['min_prob_knee'] = min_prob_knee
            record['points_removed'] = points_removed
            record['points_removed_pct'] = points_removed / len(points)
        else:
            polygon = polygon_from_points(points, polygon_type, alpha)  # create polygon from points

        record['label'] = f'cluster_{cluster}'  # cluster label

        record['probs_of_belonging_to_clust'] =\
            ', '.join(cluster_df['cluster_probability'].astype(str).to_list())  # list of all points probabilities of belonging to the cluster
        record['mean_prob_of_belonging_to_cluster'] = cluster_df['cluster_probability'].mean()  # mean probability of belonging to cluster
        record['geometry'] = polygon  # polygenized cluster
        record['geojson'] = gpd.GeoSeries(polygon).to_json()  # geojson format of the polygenized cluster
        record['num_points'] = cluster_df.shape[0]  # total number of points in cluster
        record['area_sqkm'] = calc_polygon_area_sq_unit(polygon)  # polygon area in sqkm
        record['density'] = calc_cluster_density(points)  # polygon density (1/mean squared distance)
        record['mean_duration'] = cluster_df['duration'].mean()  # mean duration from first blip to last blip
        record['median_duration'] = cluster_df['duration'].median()  # median duration from first blip to last blip
        record['distance_from_nearest_port'], record['name_of_nearest_port'] =\
            calc_polygon_distance_from_nearest_port(polygon, ports_df)  # distance in km from nearest WW port and its name
        record['n_unique_vesselID'] = cluster_df['vesselId'].nunique()  # number of unique vessel IDs in cluster
        record['percent_unique_vesselID'] = cluster_df['vesselId'].nunique() / len(points)  # percent of unique vesselIDs in cluster
        record['vesselIDs'] = ', '.join(cluster_df['vesselId'].astype(str).to_list())  # list of all vessel IDs in cluster
        record['most_freq_vessel_type'] = cluster_df['vessel_class_new'].mode()[0]  # most frequent vessel type in cluster
        record['vessel_type_variance'] = calc_entropy(cluster_df['vessel_class_new'])  # variance of vessel type in cluster

        if activity == 'anchoring':
            record['most_freq_destination'] = cluster_df['nextPort_name'].mode()[0]  # most frequent destination in cluster
            record['destination_variance'] = calc_entropy(cluster_df['nextPort_name'])  # variance of destination in cluster

        record['is_in_river'] = polygon.within(main_land)  # is the polygon in rivers (True) or in the sea/ocean (False)
        record['centroid_lat'] = polygon.centroid.y  # latitude of polygon centroid
        record['centroid_lng'] = polygon.centroid.x  # longitude of polygon centroid
        record['pct_intersection'] = polygon_intersection(polygon, polygons_df)  # percent intersection with WW polygons
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, polygons_df)  # distance from nearest WW polygon
        record['link_to_google_maps'] = create_google_maps_link_to_centroid(polygon.centroid)  # link to google maps for the polygon centroid
        distance_from_shore = calc_nearest_shore(polygon, shoreline_polygon, shoreline_distance_method)  # distance in Km and point of nearest_shore
        record.update(distance_from_shore)

        cluster_polygons.append(record)

    cluster_polygons = gpd.GeoDataFrame(cluster_polygons)

    return cluster_polygons


# TODO: update missing params in documentation

def main(import_path, export_path, activity='anchoring', blip='first',
         hdbscan_min_cluster_zise=30, hdbscan_min_samples=5,
         hdbscan_distance_metric='euclidean', alpha=4,
         sub_area_polygon_fname=None, optimize_polygon=False, merge_near_polygons=False,
         shoreline_distance_method='euclidean',
         debug=False):

    """
    :param import_path: path to all used files
    :param export_path: path to save dataframe and model
    :param activity: 'mooring' (for ports) or 'anchoring' (for ports waiting areas)
    :param blip: 'first' or 'last'
    :param hdbscan_min_cluster_zise: hdbscan min_cluster_size hyperparameter
    :param hdbscan_min_samples: hdbscan min_samples hyperparameter
    :param distance_metric:hdbscan distance_metric hyperparameter
    :param alpha: parameter for 'alpha_shape'- degree of polygon segregation
    :param sub_area_polygon_fname: optional- add filname for sub area of interest
    :param optimize_polygon: if True, will apply optimize_polygon
    :param merge_near_polygons: merge adjacent clusters
    :param shoreline_distance_method: "euclidean" / "haversine"
    :param debug: take only subset of data for testing code
    """

    df_for_clustering_fname = f'features/df_for_clustering_{activity}.csv'

    # import df and clean it
    logging.info('Loading data...')

    nrows = 10000 if debug else None  # 10K rows per file if debug == True

    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False, nrows=nrows)
    df = df.drop_duplicates(subset=[f'{blip}Blip_lat', f'{blip}Blip_lng'])  # drop duplicates #  todo check if this operation is not problematic
    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    ports_df = gpd.read_file(os.path.join(import_path, 'maps/ports.json')) # WW ports
    polygons_df = gpd.read_file(os.path.join(import_path, 'maps/polygons.geojson')) # WW polygons

    logging.info('loading and processing shoreline file - START')
    shoreline_df = gpd.read_file(os.path.join(import_path, 'maps/shoreline_layer.geojson'))  # shoreline layer
    main_land = merge_polygons(shoreline_df[:4])  # create multipolygon of the big continents
    shoreline_polygon = merge_polygons(shoreline_df) # merging shoreline_df to one multipolygon
    logging.info('loading and processing shoreline file - END')

    logging.info('Finished loading data!')

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                min_samples=hdbscan_min_samples,
                                metric=hdbscan_distance_metric)

    logging.info('Starting clustering...')

    clusterer.fit(locations)
    logging.info('Finished fitting clusterer!')

    logging.info('Starting feature extraction for clusters...')

    # add for each point its cluster label and probability of belonging to it
    df['cluster_label'] = clusterer.labels_
    df['cluster_probability'] = clusterer.probabilities_

    # polygenize clusters and extract features of interest
    clust_polygons = polygenize_clusters_with_features(df, ports_df, polygons_df, shoreline_polygon,
                                                       main_land, activity, blip, optimize_polygon, alpha,
                                                       shoreline_distance_method)

    # merging adjacent polygons
    if merge_near_polygons:
        _, clust_polygons = merge_adjacent_polygons(clust_polygons, inflation_meter=1000, aggfunc='first')

    # add ports rank
    if activity == 'mooring':
        clust_polygons = rank_candidates(clust_polygons, debug)

    logging.info('finished extracting features')

    # save model and files
    logging.info('saving data and model...')
    pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_zise}mcs_{hdbscan_min_samples}ms_{activity}.pkl'
    pkl_model_fname = os.path.join(export_path, pkl_model_fname)

    clust_polygons_fname = pkl_model_fname + '_polygons.geojson'
    clust_polygons_fname = os.path.join(export_path, 'not_optimzie_'+clust_polygons_fname)

    with open(pkl_model_fname, 'wb') as file:
        pickle.dump(clusterer, file)
    clust_polygons.to_file(clust_polygons_fname, driver="GeoJSON")
    # TODO: add csv version for analysts with polygon in geojson form


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
