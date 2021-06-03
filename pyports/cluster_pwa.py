"""
Cluster ports waiting areas (pwa).
Takes only anchoring activity and only container vessels, and cluster ports waiting areas by destination.
For each destination port, first find anchoring container vessels that were heading to it (i.e. had this
port as their NextPort) and are less than 200km away from it. Then cluster them with hdbscan min_cluster_size:20,
min_samples:10. Lastly create polygons from these clusters and extract their features.
"""

import hdbscan
import fire
from pyports.cluster_activities_utils import *


type_of_area_mapped = 'pwa'


def main(import_path, export_path, activity='anchoring', blip='first', only_container_vessels=True,
         hdbscan_min_cluster_size=20, hdbscan_min_samples=10, hdbscan_distance_metric='haversine',
         polygon_type='alpha_shape', polygon_alpha=4, sub_area_polygon_fname=None, optimize_polygon=False,
         save_files=False, debug=False
         ):

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
    :param save_files: boolean- whether to save results and model to output_path.
    :param debug: take only subset of data for testing code.

    """

    df, ports_df, polygons_df, main_land, shoreline_polygon = \
        get_data_for_clustering(import_path, activity, debug,
                                sub_area_polygon_fname, blip)

    # TODO: add next 4 lines to get_data_for_clustering (+ the only_container_vessels param)
    if only_container_vessels:
        df = df[df.vessel_class_new == 'cargo_container']  # take only container vessels
        df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
        df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
        df.reset_index(drop=True, inplace=True)  # reset index

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

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

    # polygenize clusters and extract features of interest
    ports_waiting_areas_polygons = polygenize_clusters_with_features(type_of_area_mapped, df, polygons_df, main_land,
                                                                     blip, optimize_polygon, polygon_alpha,
                                                                     polygon_type, only_container_vessels)

    # save results
    if save_files:
        save_data(type_of_area_mapped, ports_waiting_areas_polygons, export_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
