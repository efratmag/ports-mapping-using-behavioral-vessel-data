import os
import hdbscan
import pickle
import fire
from pyports.geo_utils import *
from tqdm import tqdm
from pyports.rank_ports_candidates import main as rank_candidates

# TODO: generlize paths
# ACTIVITY = 'anchoring'
# FILE_NAME = f'df_for_clustering_{ACTIVITY}.csv'  # df with lat lng of all anchoring activities
# SHORELINE_FNAME = 'shoreline_layer.geojson'
# path_to_shoreline_file = os.path.join('/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/maps/', SHORELINE_FNAME)


def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df, main_land,
                                      polygons_df, activity,
                                      blip, alpha=4):

    df_for_clustering = df_for_clustering[df_for_clustering.cluster_label != -1]  # remove clustering outlier points

    clust_polygons = []

    for cluster in tqdm(df_for_clustering.cluster_label.unique()):
        record = {}
        cluster_df = df_for_clustering[df_for_clustering.cluster_label == cluster]
        points = cluster_df[[f'{blip}Blip_lng', f'{blip}Blip_lat']].to_numpy()
        polygon = alpha_shape(points, alpha)[0]

        # polygon = MultiPoint(points).convex_hull
        record['label'] = f'cluster {cluster}'

        record['probs_of_belonging_to_clust'] = ', '.join(cluster_df['cluster_probability'].astype(str).to_list())
        record['mean_prob_of_belonging_to_cluster'] = cluster_df['cluster_probability'].mean()
        record['geometry'] = polygon
        record['num_points'] = cluster_df.shape[0]
        record['area_sqkm'] = calc_polygon_area_sq_unit(polygon)
        record['density'] = calc_cluster_density(points)
        record['mean_duration'] = cluster_df['duration'].mean()
        record['median_duration'] = cluster_df['duration'].median()
        record['distance_from_nearest_port'], record['name_of_nearest_port'] = calc_polygon_distance_from_nearest_port(polygon, ports_df)
        record['n_unique_vesselID'] = cluster_df['vesselId'].nunique()
        record['percent_unique_vesselID'] = cluster_df['vesselId'].nunique() / len(points)
        record['vesselIDs'] = ', '.join(cluster_df['vesselId'].astype(str).to_list())
        record['most_freq_vessel_type'] = cluster_df['class_new'].mode()[0]
        record['vessel_type_variance'] = calc_entropy(cluster_df['class_new'])
        record['is_in_river'] = calc_entropy(cluster_df['class_new'])
        record['is_in_river'] = polygon.within(main_land)
        record['centroid_lat'] = polygon.centroid.y
        record['centroid_lng'] = polygon.centroid.x
        record['pct_intersection'] = polygon_intersection(polygon, polygons_df)
        record['dist_to_ww_poly'] = calc_polygon_distance_from_nearest_ww_polygon(polygon, polygons_df)
        record['link_to_google_maps'] = create_google_maps_link_to_centroid(polygon.centroid)

        if activity == 'anchoring':

            record['most_freq_destination'] = cluster_df['nextPort_name'].mode()[0]
            record['destination_variance'] = calc_entropy(cluster_df['nextPort_name'])

        clust_polygons.append(record)

    clust_polygons = gpd.GeoDataFrame(clust_polygons)

    return clust_polygons


#TODO call clean_data_and_extract_features at start

def main(import_path, export_path, activity='anchoring', blip='first',
         hdbscan_min_cluster_zise=30, hdbscan_min_samples=5,
         distance_metric='euclidean', alpha=4,
         sub_area_polygon_fname=None, merge_near_polygons=False):

    """

    :param import_path: path to all relevant files
    :param export_path: path to
    :param activity:
    :param blip:
    :param hdbscan_min_cluster_zise:
    :param hdbscan_min_samples:
    :param distance_metric:
    :param alpha:
    :param sub_area_polygon_fname:
    :param merge_near_polygons:
    :return:
    """

    df_for_clustering_fname = f'df_for_clustering_{activity}.csv'

    # import df and clean it
    logging.info('Loading data...')
    df = pd.read_csv(os.path.join(import_path, df_for_clustering_fname), low_memory=False)
    df = df.drop_duplicates(subset=[f'{blip}Blip_lat', f'{blip}Blip_lng'])  # drop duplicates
    df.nextPort_name.fillna('UNKNOWN', inplace=True)  # fill empty next port names #TODO move to clean_data_and_extract_features
    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(import_path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    ports_df = gpd.read_file('maps/ports.json')
    polygons_df = gpd.read_file('maps/polygons.geojson')  # WW polygons
    shoreline_df = gpd.read_file('maps/shoreline_layer.geojson')

    main_land = merge_polygons(shoreline_df[:4])  # the big continents

    logging.info('Finished loading data!')

    locations = df[[f'{blip}Blip_lat', f'{blip}Blip_lng']].to_numpy()  # points for clustering

    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                min_samples=hdbscan_min_samples,
                                metric=distance_metric)

    logging.info('Starting clustering...')

    clusterer.fit(locations)
    logging.info('Finished fitting clusterer!')

    logging.info('Starting feature extraction for clusters...')

    # add for each point its cluster label and probability of belonging to it
    df['cluster_label'] = clusterer.labels_
    df['cluster_probability'] = clusterer.probabilities_

    clust_polygons = polygenize_clusters_with_features(df, ports_df, polygons_df, main_land, activity, blip, alpha)

    clust_polygons = calc_nearest_shore(clust_polygons, shoreline_df, method='haversine')

    if merge_near_polygons:

        _, clust_polygons = merge_adjacent_polygons(clust_polygons, inflation_meter=1000, aggfunc='first')

    if activity == 'mooring':
        clust_polygons = rank_candidates(clust_polygons)

    logging.info('finished extracting features')

    # save model and files
    logging.info('saving data and model...')
    pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_zise}mcs_{hdbscan_min_samples}ms_{activity}.pkl'
    pkl_model_fname = os.path.join(export_path, pkl_model_fname)

    clust_polygons_fname = pkl_model_fname + '_polygons.geojson'
    clust_polygons_fname = os.path.join(export_path, clust_polygons_fname)

    with open(pkl_model_fname, 'wb') as file:
        pickle.dump(clusterer, file)
    clust_polygons.to_file(clust_polygons_fname, driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
