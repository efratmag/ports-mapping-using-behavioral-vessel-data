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
from pyports.cluster_activities_utils import *
from pyports.connected_components_utils import *
from pyports.geo_utils import is_in_river
from pyports.constants import ACTIVITY, AreaType
from typing import Union
import time
import pathlib
import fire
from tqdm import tqdm
tqdm.pandas()


def main(import_path: pathlib.Path, export_path: pathlib.Path, activity: Union[ACTIVITY, str] = ACTIVITY.MOORING,
         blip: str = 'first', type_of_area_mapped: Union[AreaType, str] = AreaType.PORTS,
         filter_river_points: bool = True, epsilon: str = 10000, only_container_vessels: bool = False,
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

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lon']].rename({f'{blip}Blip_lat': 'lat', f'{blip}Blip_lon': 'lon'},
                                                                  axis=1).reset_index(drop=True)

# TODO: for all exists file inspections- make sure the file checked is per running time
    # filter out points in rivers
    if filter_river_points:
        if not import_path.joinpath("river_mask_mooring.csv").exists():
            river_mask = is_in_river(locations, main_land)
            river_mask.to_csv(os.path.join(import_path, 'river_mask_mooring.csv'))
        else:
            river_mask = pd.read_csv(import_path.joinpath('river_mask_mooring.csv'))
        locations = locations[np.invert(river_mask)]  # take only ports where in_river == False
        print(f'removed {np.sum(river_mask)} points that lay in rivers ('
              f'{np.sum(river_mask) / locations.shape[0] *100:.2f}% of the data).')

    # get locations_utm - projections of lat lon to utm coordinates
    if not import_path.joinpath("locations_utm.csv").exists():
        locations_utm = locations.progress_apply(lambda row: get_utm(row.lat, row.lon), axis=1)
        locations_utm.to_csv(import_path.joinpath("locations_utm.csv"), index=False)
    else:
        locations_utm = pd.read_csv(import_path.joinpath("locations_utm.csv"))

    # get zone feature by combining utm number and letter
    locations_utm["zone"] = locations_utm.apply(lambda row: f"{row.zone_number}{row.zone_letter}", axis=1)

    # get borders- a boolean per location indicating if it occurs close to a utm zone border (<epsilon away from border)
    if not import_path.joinpath("locations_utm_border.csv").exists():
        border_statuses = locations_utm.progress_apply(lambda row: is_border(row.lat,
                                                                             row.lon,
                                                                             row.zone_number,
                                                                             row.zone_letter, epsilon),
                                                       axis=1)
        locations_utm = locations_utm.join(border_statuses)
        locations_utm.to_csv(import_path.joinpath("locations_utm_border.csv"), index=False)
    else:
        locations_utm = pd.read_csv(import_path.joinpath("locations_utm_border.csv"))


    logging.info('starting growing connected components...')

    # initializations

    locs = locations_utm.copy()
    locs["component"] = -1  # initialize all points as unmarked (i.e noise by default)
    # get sub zones by dividing easting/northing by epsiloneshold
    locs["cell_x"] = (locs["easting"] / epsilon).astype(int)
    locs["cell_y"] = (locs["northing"] / epsilon).astype(int)
    # sum up all four borders' boolean to one general indicator of whether a point is in border zone
    locs["border"] = (locs[["N", "E", "S", "W"]].sum(axis=1)!=0).astype(int)

    current_cid = -1
    components = {}
    timings = []

    # build kdTrees

    zone_grp = locs.groupby(["zone_number", "zone_letter"])

    kdtrees = {}

    for zone, zone_locations in tqdm(zone_grp, total=len(zone_grp.groups)):
        zn, zl = zone
        zone_mask = (locations_utm.zone_number == zn) & (locations_utm.zone_letter == zl)
        tree = cKDTree(locations_utm.loc[zone_mask, ["easting", "northing"]].values)
        kdtrees[zone] = [locations_utm.loc[zone_mask].index, tree]

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


    # TODO: cluster on top

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
