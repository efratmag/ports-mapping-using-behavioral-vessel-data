"""
Cluster ports waiting areas (pwa).
Takes only anchoring activity and only container vessels, and cluster ports waiting areas by destination.
For each destination port, first find anchoring container vessels that were heading to it (i.e. had this
port as their NextPort) and are less than 200km away from it. Then cluster them with hdbscan min_cluster_size:20,
min_samples:10. Lastly create polygons from these clusters and extract their features.
"""

import fire
from pyports.cluster_activities_utils import *
from pyports.constants import ACTIVITY, AreaType
from typing import Union


# TODO: fix issues with import/export paths
def main(import_path: str, export_path: str, activity: Union[ACTIVITY, str] = ACTIVITY.MOORING,
         blip: str = 'first', type_of_area_mapped: Union[AreaType, str] = AreaType.PORTS,
         only_container_vessels: bool = True, polygon_type: str = 'alpha_shape', polygon_alpha: int = 4,
         optimize_polygon: bool = False, sub_area_polygon_fname: str = None, save_files: bool = False,
         debug: bool = False):

    """
    :param import_path: path to all used files.
    :param export_path: path to save dataframe.
    :param activity: default: 'anchoring' (for ports waiting areas).
    :param blip: 'first' or 'last'.
    :param type_of_area_mapped: ports waiting areas
    :param only_container_vessels: use only container vessels for pwa mapping. default: True.
    :param hdbscan_min_cluster_size: hdbscan min_cluster_size hyper parameter (20 for anchoring).
    :param hdbscan_min_samples: hdbscan min_samples hyper parameter (10 for anchoring).
    :param hdbscan_distance_metric: hdbscan distance_metric hyper parameter.
    :param polygon_type: 'alpha_shape' or 'convexhull'.
    :param polygon_alpha: parameter for 'alpha_shape'- degree of polygon segregation.
    :param optimize_polygon: if True, will apply optimize_polygon.
    :param sub_area_polygon_fname: optional- add file name for sub area of interest.
    :param save_files: boolean- whether to save results and model to output_path.
    :param debug: take only subset of data for testing code.

    """

    logging.info('loading data...')
    # loading data
    df, ports_df, polygons_df, main_land, shoreline_polygon = \
        get_data_for_clustering(import_path, type_of_area_mapped, activity,  blip,
                                only_container_vessels, sub_area_polygon_fname, debug)

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lon']].to_numpy()  # points for clustering

    logging.info('starting clustering...')

    # cluster per port and create dataframe for feature generation

    logging.info('finished clustering!')

    # polygenize clusters and extract features of interest
    ports_waiting_areas_polygons = polygenize_clusters_with_features(type_of_area_mapped, df, polygons_df, main_land,
                                                                     blip, optimize_polygon, polygon_alpha,
                                                                     polygon_type, only_container_vessels)

    # save results
    if save_files:
        logging.info('saving files...')
        save_data(type_of_area_mapped, ports_waiting_areas_polygons, export_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
