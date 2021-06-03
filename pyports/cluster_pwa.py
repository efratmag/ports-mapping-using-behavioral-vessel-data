"""
Cluster ports waiting areas (pwa).
Takes only anchoring activity and only container vessels, and cluster ports waiting areas by destination.
For each destination port, first find anchoring container vessels that were heading to it (i.e. had this
port as their NextPort) and are less than 200km away from it. Then cluster them with hdbscan min_cluster_size:20,
min_samples:10. Lastly create polygons from these clusters and extract their features.
"""

import os
import hdbscan
import fire
from pyports.geo_utils import *
from pyports.get_data_for_clustring import get_data_for_clustering
from tqdm import tqdm
import datetime

# TODO: generalize paths
def polygenize_clusters_with_features(df_for_clustering,
                                      polygons_df,
                                      main_land,
                                      blip,
                                      optimize_polygon,
                                      only_container_vessels,
                                      alpha=4,
                                      polygon_type='alpha_shape'
                                      ):

    """
    :param df_for_clustering: activities dataframe with clustering results.
    :param polygons_df: dataframe of WW polygons.
    :param main_land: multi-polygon of the main continents.
    :param blip: as in main- 'first' or 'last'.
    :param only_container_vessels: as in main- use only container vessels for pwa mapping. default: True.
    :param optimize_polygon: if True, will apply optimize_polygon.
    :param alpha: as in main- parameter for alpha shape- degree of polygon segregation.
    :param polygon_type: 'alpha_shape' or 'convexhull'.
    :return: geopandas dataframe of all polygenized clusters with their features.
    """

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

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

        record['probs_of_belonging_to_clust'] =\
            ', '.join(cluster_df['cluster_probability'].astype(str).to_list())  # list of all points probabilities
        # of belonging to the cluster
        record['mean_prob_of_belonging_to_cluster'] = cluster_df['cluster_probability'].mean()  # mean probability
        # of belonging to cluster
        record['geometry'] = polygon  # polygenized cluster
        record['num_points'] = cluster_df.shape[0]  # total number of points in cluster
        record['area_sqkm'] = calc_polygon_area_sq_unit(polygon)  # polygon area in sqkm
        record['density'] = calc_cluster_density(points)  # polygon density (1/mean squared distance)
        record['mean_duration'] = cluster_df['duration'].mean()  # mean duration from first blip to last blip
        record['median_duration'] = cluster_df['duration'].median()  # median duration from first blip to last blip
        record['n_unique_vesselID'] = cluster_df['vesselId'].nunique()  # number of unique vessel IDs in cluster
        record['percent_unique_vesselID'] = cluster_df['vesselId'].nunique() / len(points)  # percent of unique
        # vesselIDs in cluster
        record['vesselIDs'] = ', '.join(cluster_df['vesselId'].astype(str).to_list())  # list of all vessel IDs
        # in cluster
        if not only_container_vessels:
            record['most_freq_vessel_type'] = cluster_df['vessel_class_new'].mode()[0]  # most frequent vessel type
            # in cluster
            record['vessel_type_variance'] = calc_entropy(cluster_df['vessel_class_new'])  # variance of vessel type
            # in cluster
        record['is_in_river'] = polygon.within(main_land)  # is the polygon in rivers (True) or in the sea/ocean (False)
        record['centroid_lat'] = polygon.centroid.y  # latitude of polygon centroid
        record['centroid_lng'] = polygon.centroid.x  # longitude of polygon centroid
        record['percent_intersection'] = polygon_intersection(polygon, polygons_df)  # percent intersection with WW
        # polygons
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, polygons_df)  # distance from
        # nearest WW polygon
        record['link_to_google_maps'] = create_google_maps_link_to_centroid(polygon.centroid)  # link to google maps for
        # the polygon centroid

        cluster_polygons.append(record)

    cluster_polygons = gpd.GeoDataFrame(cluster_polygons)

    return cluster_polygons


def main(import_path, export_path, activity='anchoring', blip='first',
         only_container_vessels=True,
         hdbscan_min_cluster_size=20, hdbscan_min_samples=10,
         hdbscan_distance_metric='haversine',
         polygon_type='alpha_shape', polygon_alpha=4,
         sub_area_polygon_fname=None, optimize_polygon=False,
         save_files=False,
         debug=False):

    """
    :param import_path: path to all used files.
    :param export_path: path to save dataframe.
    :param activity: default: 'anchoring' (for ports waiting areas).
    :param blip: 'first' or 'last'.
    :param only_container_vessels: use only container vessels for pwa mapping. default: True.
    :param hdbscan_min_cluster_size: hdbscan min_cluster_size hyper parameter (20 for anchoring).
    :param hdbscan_min_samples: hdbscan min_samples hyper parameter (10 for anchoring).
    :param hdbscan_distance_metric: hdbscan distance_metric hyper parameter.
    :param polygon_type: 'alpha_shape' or 'convexhull'.
    :param polygon_alpha: parameter for 'alpha_shape'- degree of polygon segregation.
    :param sub_area_polygon_fname: optional- add file name for sub area of interest.
    :param optimize_polygon: if True, will apply optimize_polygon.
    :param save_files: boolean- whether to save results and model to output_path
    :param debug: take only subset of data for testing code

    """

    logging.info('loading data...')

    df, ports_df, polygons_df, main_land, shoreline_polygon = \
        get_data_for_clustering(import_path, activity, debug,
                                sub_area_polygon_fname, blip)

    # TODO: add next 4 lines to get_data_for_clustering (+ the only_container_vessels param)
    df = df[df.vessel_class_new == 'cargo_container']  # take only container vessels
    df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
    df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
    df.reset_index(drop=True, inplace=True)  # reset index

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

    logging.info('finished loading data!')

    logging.info('starting clustering...')

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

    logging.info('finished clustering!')

    logging.info('starting feature extraction for clusters...')

    # polygenize clusters and extract features of interest
    clust_polygons = polygenize_clusters_with_features(df, ports_df, polygons_df, shoreline_polygon,
                                                       main_land, activity, blip, optimize_polygon,
                                                       polygon_alpha, polygon_type)

    logging.info('finished extracting polygons features!')

    if save_files:
        logging.info('saving data...')

        pwa_polygons_fname = f'ports_waiting_areas_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        clust_polygons.to_file(os.path.join(export_path, pwa_polygons_fname+'.geojson'), driver="GeoJSON")
        clust_polygons.to_csv(os.path.join(export_path, pwa_polygons_fname+'.csv'))

        logging.info('finished saving data!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
