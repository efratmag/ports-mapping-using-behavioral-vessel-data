"""
Cluster ports.
Takes only mooring activity and only cargo and tanker vessels, and cluster ports with growing connected components.
First, all points are projected to UTM zones and features regarding points location are extracted (e.g. whether a point
is in border zone - less than threshold (epsilon) away from utm border).
Then a cKDtree is built to estimate neighbors.
Then all points are initialized as noise (i.e. unmarked), and seeds are randomly selected iteratively and their
neighboring points are assigned to their cluster. The process finishes when all points are visited.
Lastly polygons are created from these clusters and their features are extracted.
"""

from scipy.spatial import cKDTree
import hdbscan
from pyports.cluster_activities_utils import *
from pyports.connected_components_utils import *
from pyports.constants import ACTIVITY, AreaType
from typing import Union
import time
import pathlib
import fire
from tqdm import tqdm
tqdm.pandas()


def main(import_path: pathlib.Path, export_path: pathlib.Path, activity: Union[ACTIVITY, str] = ACTIVITY.MOORING,
         blip: str = 'first', type_of_area_mapped: Union[AreaType, str] = AreaType.PORTS,
         filter_river_points: bool = True, epsilon: int = 10000, only_container_vessels: bool = False,
         hdbscan_min_cluster_size: int = 30, hdbscan_min_samples: int = 5, hdbscan_distance_metric: int = 'haversine',
         polygon_type: str = 'alpha_shape', polygon_alpha: int = 4, optimize_polygon: bool = False,
         sub_area_polygon_fname: str = None, save_files: bool = False, debug: bool = False):

    """
    :param import_path: path to all used files.
    :param export_path: path to saved dataframe.
    :param activity: default: 'mooring' (for ports).
    :param blip: 'first' or 'last'.
    :param type_of_area_mapped: ports.
    :param filter_river_points: filter points in rivers. default: True.
    :param epsilon: radius for neighbors definition. default: 10,000 (10 km).
    :param only_container_vessels: use only container vessels for pwa mapping. default: True.
    :param hdbscan_min_cluster_size: hdbscan min_cluster_size hyper parameter (30 for mooring).
    :param hdbscan_min_samples: hdbscan min_samples hyper parameter (5 for mooring).
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
        get_data_for_clustering(str(import_path), type_of_area_mapped, activity, blip,
                                only_container_vessels, sub_area_polygon_fname, debug)

    logging.info('preprocessing data...')

    locations_preprocessed = preprocess_for_connected_components(import_path, df, blip, main_land, filter_river_points,
                                                                 epsilon)

    logging.info('starting growing connected components...')

    # initializations
    locs = locations_preprocessed.copy()
    locs["component"] = -1  # initialize all points as unmarked (i.e noise by default)
    current_cid = -1
    components = {}
    timings = []

    # build kdTrees per zone

    zone_grp = locs.groupby(["zone_number", "zone_letter"])

    kdtrees = {}

    for zone, zone_locations in tqdm(zone_grp, total=len(zone_grp.groups)):
        zn, zl = zone
        zone_mask = (locations_preprocessed.zone_number == zn) & (locations_preprocessed.zone_letter == zl)
        tree = cKDTree(locations_preprocessed.loc[zone_mask, ["easting", "northing"]].values)
        kdtrees[zone] = [locations_preprocessed.loc[zone_mask].index, tree]

    # growing connected components
    global_start_t = time.time()

    while (locs.component == -1).sum() != 0:
        start_t = time.time()
        current_cid += 1
        zones_to_update = set()

        # TODO: add weights, use more connected locations first
        seed = locs[locs.component == -1].sample(1).index[0]

        zn, zl = locs.loc[seed, ["zone_number", "zone_letter"]]
        zone_mask = (locs.zone_number ==zn) & (locs.zone_letter == zl)

        component = ConnectedComponent(current_cid, locs, epsilon)
        component.add(seed)
        locs.loc[seed, "component"] = current_cid

        print(f"""Starting new component {component.cid} in {locs.loc[seed, 'zone_number']}
              {locs.loc[seed, 'zone_letter']}, """
              f"""unmarked locations: {(locs.component == -1).sum()} out of {locs.shape[0]}""")

        while not component.is_full():
            component_mask = locs.component == -1
            if not locs.loc[zone_mask & component_mask].empty:
                kdtrees[(zn, zl)] = [locs.loc[zone_mask & component_mask].index,
                                     cKDTree(locs.loc[zone_mask & component_mask,
                                                      ["easting", "northing"]].values)]

                for (lzn, lzl) in zones_to_update:
                    local_zone_mask = (locs.zone_number == lzn) & (locs.zone_letter == lzl)
                    zone_tree = cKDTree(locs.loc[local_zone_mask & component_mask,
                                                 ["easting", "northing"]].values)
                    kdtrees[(lzn, lzl)] = [locs.loc[local_zone_mask & component_mask].index, zone_tree]

                zones_to_update = component.grow(kdtrees=kdtrees)
            else:
                component.grow(locs=locs)
            locs.loc[component.elements, "component"] = current_cid
            print(f"\tGrowing {component.cid}: {component.size}")

        components[component.cid] = component.elements

        end_t = time.time()
        print(f"Elapsed: {(end_t - global_start_t)/60:.2f}m")
        timings.append([component.size, end_t - start_t])

    logging.info('finished growing connected components!')

    logging.info('starting post processing of components...')

    # remove small components
    locs_pp = locs.groupby('component').filter(lambda x: len(x) > 20).reset_index(drop=True)

    # hdbscan clustering on each component
    num_clusters = 0
    for i, comp in enumerate(tqdm(locs_pp.component.unique())):
        idxs = locs_pp.index[locs_pp.component == comp]
        sub_locs = locs_pp.loc[idxs, ['lat', 'lon']].to_numpy()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples,
                                    metric=hdbscan_distance_metric)
        clusterer.fit(sub_locs)
        locs_pp.loc[idxs, 'cluster_probability'] = clusterer.probabilities_
        if i == 0:
            locs_pp.loc[idxs, 'cluster_label'] = clusterer.labels_
            num_clusters = clusterer.labels_.max() + 1
        else:
            cluster_labels = np.where(clusterer.labels_ > -1, clusterer.labels_ + num_clusters,
                                      clusterer.labels_)
            locs_pp.loc[idxs, 'cluster_label'] = cluster_labels
            num_clusters += clusterer.labels_.max() + 1

    logging.info('finished post processing of components!')

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
