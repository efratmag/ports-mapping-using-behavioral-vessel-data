import os
import hdbscan
import pickle
import fire
from shapely.geometry import Point
from pyports.geo_utils import *
from pyports.rank_ports_candidates import main as rank_candidates
from tqdm import tqdm


# TODO: generalize paths
def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df, polygons_df,
                                      main_land, activity,
                                      blip, alpha=4,
                                      polygon_type='alpha_shape',
                                      only_containers=None):
    """
    :param df_for_clustering: activities dataframe with clustering results
    :param ports_df: dataframe of WW ports
    :param polygons_df: dataframe of WW polygons
    :param main_land: multi-polygon of the main continents
    :param activity: qs in main- activity type (i.e mooring/anchoring etc.)
    :param blip: as in main- 'first' or 'last'
    :param alpha: as in main- parameter for alpha shape- degree of polygon segregation
    :param polygon_type: 'alpha_shape' or 'convex_hull'
    :return: geopandas dataframe of all polygenized clusters with their features
    """

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

    cluster_polygons = []

    for cluster in tqdm(df_for_clustering.cluster_label.unique()):  # iterate over clusters
        record = {}
        cluster_df = df_for_clustering[df_for_clustering.cluster_label == cluster]  # sub-df for chosen cluster
        points = cluster_df[[f'{blip}Blip_lng', f'{blip}Blip_lat']].to_numpy()  # numpy array of lng/lat
        polygon = polygon_from_points(points, polygon_type, alpha)  # create polygon from points

        # cluster label
        record['label'] = f'cluster {cluster}'
        # list of all points probabilities of belonging to the cluster
        record['probs_of_belonging_to_clust'] =\
            ', '.join(cluster_df['cluster_probability'].astype(str).to_list())
        # mean probability of belonging to cluster
        record['mean_prob_of_belonging_to_cluster'] = cluster_df['cluster_probability'].mean()
        # polygenized cluster
        record['geometry'] = polygon
        # geojson format of the polygenized cluster
        record['geojson'] = gpd.GeoSeries(polygon).to_json()
        # total number of points in cluster
        record['num_points'] = cluster_df.shape[0]
        # polygon area in sqkm
        record['area_sqkm'] = calc_polygon_area_sq_unit(polygon)
        # polygon density (1/mean squared distance)
        record['density'] = calc_cluster_density(points)
        # mean duration from first blip to last blip
        record['mean_duration'] = cluster_df['duration'].mean()
        # median duration from first blip to last blip
        record['median_duration'] = cluster_df['duration'].median()
        # distance in km from nearest WW port and its name
        # number of unique vessel IDs in cluster
        record['n_unique_vesselID'] = cluster_df['vesselId'].nunique()
        # percent of unique vesselIDs in cluster
        record['percent_unique_vesselID'] = cluster_df['vesselId'].nunique() / len(points)
        # list of all vessel IDs in cluster
        record['vesselIDs'] = ', '.join(cluster_df['vesselId'].astype(str).to_list())
        # most frequent vessel type in cluster
        record['most_freq_vessel_type'] = cluster_df['class_new'].mode()[0]
        # variance of vessel type in cluster
        record['vessel_type_variance'] = calc_entropy(cluster_df['class_new'])
        if activity == 'anchoring':
            # most frequent destination in cluster
            record['most_freq_destination'] = cluster_df['nextPort_name'].mode()[0]
            # variance of destination in cluster
            record['destination_variance'] = calc_entropy(cluster_df['nextPort_name'])
        # is the polygon in rivers (True) or in the sea/ocean (False)
        record['is_in_river'] = polygon.within(main_land)
        # latitude of polygon centroid
        record['centroid_lat'] = polygon.centroid.y
        # longitude of polygon centroid
        record['centroid_lng'] = polygon.centroid.x
        # percent intersection with WW polygons
        record['percent_intersection'] = polygon_intersection(polygon, polygons_df)
        # distance from nearest WW polygon
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, polygons_df)
        # link to google maps for the polygon centroid
        record['link_to_google_maps'] = create_google_maps_link_to_centroid(polygon.centroid)
        if not only_containers:
            record['distance_from_nearest_port'], record['name_of_nearest_port'] = \
                calc_polygon_distance_from_nearest_port(polygon, ports_df)
        cluster_polygons.append(record)

    cluster_polygons = gpd.GeoDataFrame(cluster_polygons)

    return cluster_polygons


# TODO: call clean_data_and_extract_features at start
def main(import_path, export_path, activity='anchoring', blip='first',
         hdbscan_min_cluster_zise=20, hdbscan_min_samples=10,
         hdbscan_distance_metric='euclidean',
         sub_area_polygon_fname=None, merge_near_polygons=False,
         only_containers=True,
         debug=False, save_files_and_model=False):

    """
    :param import_path: path to all used files
    :param export_path: path to save dataframe and model
    :param activity: 'mooring' (for ports) or 'anchoring' (for ports waiting areas)
    :param blip: 'first' or 'last'
    :param hdbscan_min_cluster_zise: hdbscan min_cluster_size hyper parameter (30 for mooring 20 for abchoring)
    :param hdbscan_min_samples: hdbscan min_samples hyper parameter (5 for mooring 10 for anchoring)
    :param hdbscan_distance_metric: hdbscan distance_metric hyper parameter
    :param alpha: parameter for 'alpha_shape'- degree of polygon segregation
    :param sub_area_polygon_fname: optional- add file name for sub area of interest
    :param merge_near_polygons: merge adjacent clusters
    :param only_containers: boolean- take only container vessels
    :param debug: take first 1000 samples for debugging
    :param save_files_and_model: boolean- whether to save results and model to output_path
    param debug: take only subset of data for testing code
    """

    df_for_clustering_fname = f'features/df_for_clustering_{activity}.csv'

    # import df and clean it
    logging.info('Loading data...')
    # TODO: link to database, read raw files and run 'clean_data_and_extract_features' on it
    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False)
    df = df.drop_duplicates(subset=[f'{blip}Blip_lat', f'{blip}Blip_lng'])  # drop duplicates
    # TODO: move to clean_data_and_extract_features
    df.nextPort_name.fillna('UNKNOWN', inplace=True)  # fill empty next port names #
    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]
    if only_containers:
        df = df[df.class_new == 'cargo_container']  # take only container vessels
        df = df[df.nextPort_name != 'UNKNOWN']  # remove missing values
        df = df.groupby("nextPort_name").filter(lambda x: len(x) > 20)  # take only ports with at least 20 records
        df.reset_index(drop=True, inplace=True)  # reset index
    if debug:
        df = df[:1000]

    ports_df = gpd.read_file(os.path.join(import_path, 'maps/ports.geojson'))  # WW ports
    ports_df.drop_duplicates(subset='name', inplace=True)
    polygons_df = gpd.read_file(os.path.join(import_path, 'maps/polygons.geojson'))  # WW polygons
    shoreline_df = gpd.read_file(os.path.join(import_path, 'maps/shoreline_layer.geojson'))  # shoreline layer

    main_land = merge_polygons(shoreline_df[:4])  # create multipolygon of the big continents

    logging.info('Finished loading data!')

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

    logging.info('Starting clustering...')

    if only_containers:
        # cluster per port and create dataframe for feature generation
        num_clusters = 0
        for i, port in enumerate(df.nextPort_name.unique()):
            if port == 'Port Said East':
                port = 'Port Said'  # TODO: fix appropriately this bug in port name
            idxs = df.index[df.nextPort_name == port]
            locs = locations[idxs]
            locs, idxs = filter_points_far_from_port(ports_df, port, locs, idxs)
            if locs.shape[0] > 20:  # if enough points left
                clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                            min_samples=hdbscan_min_samples,
                                            metric=hdbscan_distance_metric)
                clusterer.fit(locs)
                df.loc[idxs, 'cluster_probability'] = clusterer.probabilities_

                if i == 0:
                    df.loc[idxs, 'cluster_label'] = clusterer.labels_
                    num_clusters = clusterer.labels_.max() + 1
                else:
                    cluster_labels = np.where(clusterer.labels_ > -1, clusterer.labels_ + num_clusters, clusterer.labels_)
                    df.loc[idxs, 'cluster_label'] = cluster_labels
                    num_clusters += clusterer.labels_.max() + 1
        df.cluster_label.fillna(value=-1, inplace=True)  # fill labels of missing values as noise

    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                    min_samples=hdbscan_min_samples,
                                    metric=hdbscan_distance_metric)
        clusterer.fit(locations)
        # add for each point its cluster label and probability of belonging to it
        df['cluster_label'] = clusterer.labels_
        df['cluster_probability'] = clusterer.probabilities_

    logging.info('Finished fitting clusterer!')

    logging.info('Starting feature extraction for clusters...')

    # polygenize clusters and extract features of interest
    clust_polygons = polygenize_clusters_with_features(df, ports_df, polygons_df, main_land, activity, blip, only_containers=True)
    # TODO: change function to operate on polygon level and add to polygenize_clusters_with_features
    clust_polygons = calc_nearest_shore(clust_polygons, shoreline_df, method='haversine')

    # merging adjacent polygons
    if merge_near_polygons:
        _, clust_polygons = merge_adjacent_polygons(clust_polygons, inflation_meter=1000, aggfunc='first')

    # add ports rank
    if activity == 'mooring':
        clust_polygons = rank_candidates(clust_polygons)

    logging.info('finished extracting features')

    # save model and files
    if save_files_and_model:
        logging.info('saving data and model...')
        pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_zise}mcs_{hdbscan_min_samples}ms_{activity}.pkl'
        pkl_model_fname = os.path.join(export_path, pkl_model_fname)

        clust_polygons_fname = pkl_model_fname + '_polygons.geojson'
        clust_polygons_fname = os.path.join(export_path, clust_polygons_fname)

        with open(pkl_model_fname, 'wb') as file:
            pickle.dump(clusterer, file)
        clust_polygons.to_file(clust_polygons_fname, driver="GeoJSON")
        # TODO: add csv version for analysts with polygon in geojson form


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
