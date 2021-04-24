import os
import hdbscan
import pickle
import fire
import geopandas as gpd
from pyports.geo_utils import *
from tqdm import tqdm
from sklearn import preprocessing


# TODO: generlize paths
ACTIVITY = 'anchoring'
FILE_NAME = f'df_for_clustering_{ACTIVITY}.csv'  # df with lat lng of all anchoring activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features/'  # features folder
SHORELINE_FNAME = 'shoreline_layer.geojson'
path_to_shoreline_file = os.path.join('/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/maps/', SHORELINE_FNAME)

aggfunc = {'num_points': 'sum',
           'area_sqkm': 'sum',
           'density': 'mean',
           'mean_duration': 'mean',
           'median_duration': 'mean',
           'distance_from_nearest_port': 'min',
           'n_unique_vesselID': 'mean',
           'percent_unique_vesselID': 'mean',
           'vesselIDs': 'first',
           'intersection': 'min',
           'distance_from_shore_euclidean': 'min',
           'nearest_shore_lat': 'mean',
           'nearest_shore_lng': 'mean',
           'nearest_point_lat': 'mean',
           'nearest_point_lng': 'mean',
           'mean_prob': 'mean'}


def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df,
                                      alpha, blip):

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
        clust_polygons.loc[cluster, 'mean_duration'] = \
            df_for_clustering.loc[clusters == cluster, 'duration'].mean()
        clust_polygons.loc[cluster, 'median_duration'] = \
            df_for_clustering.loc[clusters == cluster, 'duration'].median()
        clust_polygons.loc[cluster, 'distance_from_nearest_port'] = \
            calc_polygon_distance_from_nearest_port(polygon, ports_df)[0]
        clust_polygons.loc[cluster, 'name_of_nearest_port'] = \
            calc_polygon_distance_from_nearest_port(polygon, ports_df)[1]
        clust_polygons.loc[cluster, 'n_unique_vesselID'] = \
            df_for_clustering.loc[clusters == cluster, 'vesselId'].nunique()
        clust_polygons.loc[cluster, 'percent_unique_vesselID'] = \
            clust_polygons.loc[cluster, 'n_unique_vesselID'] / len(points)
        clust_polygons.at[cluster, 'vesselIDs'] = \
            ','.join(df_for_clustering.loc[clusters == cluster, 'vesselId'].to_numpy())
        clust_polygons.loc[cluster, 'most_freq_vessel_type'] = \
            df_for_clustering.loc[clusters == cluster, 'class_new'].mode()[0]
        clust_polygons.loc[cluster, 'vessel_type_variance'] = \
            calc_entropy(df_for_clustering.loc[clusters == cluster, 'class_new'])
        if ACTIVITY == 'anchoring':
            clust_polygons.loc[cluster, 'most_freq_destination'] = \
                df_for_clustering.loc[clusters == cluster, 'nextPort_name'].mode()[0]
            clust_polygons.loc[cluster, 'destination_variance'] = \
                calc_entropy(df_for_clustering.loc[clusters == cluster, 'nextPort_name'])

    return clust_polygons


def rank_candidates(geo_df_clust_polygons):

    geo_df_clust_polygons['rank'] = geo_df_clust_polygons['density'] * \
                                    geo_df_clust_polygons['n_unique_vesselID'] * \
                                    np.exp(geo_df_clust_polygons['dist_to_ww_poly'])

    x = geo_df_clust_polygons['rank'].values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    geo_df_clust_polygons['rank_scaled'] = min_max_scaler.fit_transform(x)

    geo_df_clust_polygons = geo_df_clust_polygons.sort_values('rank_scaled', ascending=False)

    return geo_df_clust_polygons


def main(path, activity='anchoring', blip='first',
         hdbscan_min_cluster_zise=30, hdbscan_min_samples=5,
         distance_metric='euclidean', alpha=4,
         sub_area_polygon_fname=None):

    df_for_clustering_fname = f'df_for_clustering_{activity}.csv'

    # import df and clean it
    # TODO: take raw data and add features- seperate lat lng, vessel type new class, within polygon etc.
    logging.info('Loading data...')
    df = pd.read_csv(os.path.join(path, df_for_clustering_fname), low_memory=False)
    df = df.drop_duplicates(subset=[f'{blip}Blip_lat', f'{blip}Blip_lng'])  # drop duplicates
    df.nextPort_name.fillna('UNKNOWN', inplace=True)  # fill empty next port names
    if sub_area_polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        logging.info('Calculating points within sub area...')
        sub_area_polygon = gpd.read_file(os.path.join(path, sub_area_polygon_fname)).loc[0, 'geometry']
        df = df[df.apply(lambda x: Point(x[f'{blip}Blip_lng'], x[f'{blip}Blip_lat']).within(sub_area_polygon), axis=1)]

    ports_df = gpd.read_file('maps/ports.json')
    polygons_df = gpd.read_file('maps/polygons.geojson')  # WW polygons

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

    df['cluster_label'] = clusterer.labels_
    df['cluster_probability'] = clusterer.probabilities_

    clust_polygons = polygenize_clusters_with_features(df, ports_df, alpha, blip)

    clust_polygons = polygon_intersection(clust_polygons, polygons_df)
    clust_polygons = calc_nearest_shore(clust_polygons, path_to_shoreline_file, method='haversine')

    _, clust_polygons = merge_adjacent_polygons(clust_polygons, inflation_meter=1000, aggfunc=aggfunc)

    if ACTIVITY == 'mooring':
        clust_polygons['dist_to_ww_poly'] = clust_polygons.geometry.apply(
            lambda x: calc_polygon_distance_from_nearest_ww_polygon(x, polygons_df))

    geo_df_clust_polygons = gpd.GeoDataFrame(clust_polygons)
    geo_df_clust_polygons.geometry = geo_df_clust_polygons.geometry.apply(lambda x: shapely.wkt.loads(x))

    geo_df_clust_polygons = rank_candidates(geo_df_clust_polygons)

    # save model and files
    pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_zise}mcs_{hdbscan_min_samples}ms_{ACTIVITY}'
    clust_polygons_fname = pkl_model_fname + '_polygons.json'

    with open('models/' + pkl_model_fname + '.pkl', 'wb') as file:
        pickle.dump(clusterer, file)
    geo_df_clust_polygons.to_file('maps/' + clust_polygons_fname, driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
