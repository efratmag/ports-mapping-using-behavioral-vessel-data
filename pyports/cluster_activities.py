import os
import hdbscan
import pickle
import fire
from pyports.geo_utils import *
from pyports.get_data_for_clustring import get_data_for_clustering
from pyports.rank_ports_candidates import main as rank_candidates
from tqdm import tqdm
import numpy as np


# TODO: generalize paths
def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df, polygons_df, shoreline_polygon,
                                      main_land,
                                      activity,
                                      blip,
                                      optimize_polygon,
                                      alpha=4,
                                      polygon_type='alpha_shape',
                                      shoreline_distance_method='haversine',
                                      destination_based_clustering=None):

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
    :param destination_based_clustering: boolean, containers only destination based clustering for pwa
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
            polygon, original_polygon, min_prob_knee, points_removed, metrics = optimize_polygon_by_probs(points, probs, polygon_type, alpha=alpha, s=1)
            record['min_prob_knee'] = min_prob_knee
            record['points_removed'] = points_removed
            record['points_removed_pct'] = points_removed / len(points)
            record['original_polygon'] = gpd.GeoSeries(original_polygon).to_json()
            record['polygon_optimization_curve'] = ', '.join([str(i) for i in metrics])
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

        if not destination_based_clustering:
            record['distance_from_nearest_port'], record['name_of_nearest_port'] = \
                calc_polygon_distance_from_nearest_port(polygon, ports_df)

            distance_from_shore = calc_nearest_shore(polygon, shoreline_polygon, shoreline_distance_method)  # distance in Km and point of nearest_shore
            record.update(distance_from_shore)

        cluster_polygons.append(record)

    cluster_polygons = gpd.GeoDataFrame(cluster_polygons)

    return cluster_polygons


# TODO: update missing params in documentation

def main(import_path, export_path, activity='anchoring', blip='first',
         hdbscan_min_cluster_size=30, hdbscan_min_samples=5,
         hdbscan_distance_metric='euclidean', alpha=4, destination_based_clustering=True,
         polygon_type='alpha_shape',
         sub_area_polygon_fname=None, optimize_polygon=False, merge_near_polygons=False,
         shoreline_distance_method='haversine', save_files_and_model=False,
         debug=False):

    """
    :param import_path: path to all used files
    :param export_path: path to save dataframe and model
    :param activity: 'mooring' (for ports) or 'anchoring' (for ports waiting areas)
    :param blip: 'first' or 'last'
    :param hdbscan_min_cluster_size: hdbscan min_cluster_size hyper parameter (30 for mooring 20 for anchoring)
    :param hdbscan_min_samples: hdbscan min_samples hyper parameter (5 for mooring 10 for anchoring)
    :param hdbscan_distance_metric: hdbscan distance_metric hyper parameter
    :param alpha: parameter for 'alpha_shape'- degree of polygon segregation
    :param polygon_type: 'alpha_shape' or 'convexhull'
    :param sub_area_polygon_fname: optional- add file name for sub area of interest
    :param optimize_polygon: if True, will apply optimize_polygon
    :param merge_near_polygons: merge adjacent clusters
    :param shoreline_distance_method: "euclidean" / "haversine"
    :param destination_based_clustering: boolean- take only container vessels for pwa destination based clustering
    :param save_files_and_model: boolean- whether to save results and model to output_path
    :param debug: take only subset of data for testing code

    """

    df, ports_df, polygons_df, main_land, shoreline_polygon = get_data_for_clustering(import_path, activity, debug,
                                                                                      sub_area_polygon_fname, blip)

    if destination_based_clustering:
        df = df[df.vessel_class_new == 'cargo_container']  # take only container vessels
        df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
        df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
        df.reset_index(drop=True, inplace=True)  # reset index

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

    logging.info('Starting clustering...')

    if destination_based_clustering:
        # cluster per port and create dataframe for feature generation
        num_clusters = 0
        for i, port in enumerate(df.nextPort_name.unique()):
            if port == 'Port Said East':
                port = 'Port Said'  # TODO: fix appropriately this bug in port name
            idxs = df.index[df.nextPort_name == port]
            locs = locations[idxs]
            locs, idxs = filter_points_far_from_port(ports_df, port, locs, idxs)
            if locs.shape[0] > 20:  # if enough points left
                clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                            min_samples=hdbscan_min_samples,
                                            metric=hdbscan_distance_metric)
                clusterer.fit(locs)
                df.loc[idxs, 'cluster_probability'] = clusterer.probabilities_

                if i == 0:
                    df.loc[idxs, 'cluster_label'] = clusterer.labels_
                    num_clusters = clusterer.labels_.max() + 1
                else:
                    cluster_labels = np.where(clusterer.labels_ > -1, clusterer.labels_ + num_clusters,
                                              clusterer.labels_)
                    df.loc[idxs, 'cluster_label'] = cluster_labels
                    num_clusters += clusterer.labels_.max() + 1
        df.cluster_label.fillna(value=-1, inplace=True)  # fill labels of missing values as noise

    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples,
                                    metric=hdbscan_distance_metric)
        clusterer.fit(locations)
        # add for each point its cluster label and probability of belonging to it
        df['cluster_label'] = clusterer.labels_
        df['cluster_probability'] = clusterer.probabilities_

    logging.info('Finished fitting clusterer!')

    logging.info('Starting feature extraction for clusters...')

    # polygenize clusters and extract features of interest
    clust_polygons = polygenize_clusters_with_features(df, ports_df, polygons_df, shoreline_polygon,
                                                       main_land, activity, blip, optimize_polygon, alpha, polygon_type,
                                                       shoreline_distance_method, destination_based_clustering)

    # merging adjacent polygons
    if merge_near_polygons:
        _, clust_polygons = merge_adjacent_polygons(clust_polygons, inflation_meter=1000, aggfunc='first')

    # add ports rank
    if activity == 'mooring':
        clust_polygons = rank_candidates(clust_polygons, debug)

    logging.info('finished extracting features')

    # save model and files

    if save_files_and_model:
        logging.info('saving data and model...')
        pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_size}mcs_{hdbscan_min_samples}ms_{activity}.pkl'
        pkl_model_fname = os.path.join(export_path, pkl_model_fname)

        clust_polygons_fname = pkl_model_fname + '_polygons.geojson'
        clust_polygons_fname = os.path.join(export_path, clust_polygons_fname)

        if not destination_based_clustering:
            with open(pkl_model_fname, 'wb') as file:
                pickle.dump(clusterer, file)
        clust_polygons.to_file(clust_polygons_fname, driver="GeoJSON")
        clust_polygons.to_csv(os.path.join(export_path, pkl_model_fname + '_polygons.csv'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
