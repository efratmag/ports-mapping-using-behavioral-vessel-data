import os
import hdbscan
from shapely.geometry import MultiPoint, Polygon
import geopandas as gpd
import pickle
import fire
from pyports.geo_utils import *
from tqdm import tqdm


# TODO: generlize paths
ACTIVITY = 'anchoring'
FILE_NAME = f'df_for_clustering_{ACTIVITY}.csv'  # df with lat lng of all anchoring activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features/'  # features folder
SHORELINE_FNAME = 'shoreline_layer.geojson'
path_to_shoreline_file = os.path.join('/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/maps/', SHORELINE_FNAME)


def transform_numbers_array_to_string(array):
    """converts list of numbers to delimited string"""
    x_arrstr = np.char.mod('%f', array)
    x_str = ",".join(x_arrstr)
    return x_str


def polygenize_clusters_with_features(df_for_clustering,
                                      ports_df,
                                      locations,
                                      clusterer):

    clusters = clusterer.labels_

    clust_polygons = pd.DataFrame()

    for cluster in tqdm(range(clusters.max() + 1), position=0, leave=True):
        points = locations[clusters == cluster]
        polygon = alpha_shape(points, 4)[0]
        polygon = ops.transform(flip, polygon)  # flip lat lng # TODO: make the order right from the start

        # polygon = MultiPoint(points).convex_hull

        clust_polygons.loc[cluster, 'label'] = f'cluster {cluster}'
        clust_polygons.at[cluster, 'probs_of_belonging_to_clust'] = \
            transform_numbers_array_to_string(clusterer.probabilities_[clusters == cluster])
        clust_polygons.loc[cluster,'mean_prob_of_belonging_to_cluster'] = \
            clusterer.probabilities_[clusters == cluster].mean()
        clust_polygons.at[cluster, 'geometry'] = polygon
        clust_polygons.loc[cluster, 'num_points'] = len(points)
        clust_polygons.loc[cluster, 'area_sqkm'] = calc_polygon_area_sq_unit(polygon)
        clust_polygons.loc[cluster, 'density'] = calc_cluster_density(points)
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

    return clust_polygons


def main(path=PATH,
         df_for_clustering_fname=FILE_NAME,
         hdbscan_min_cluster_zise=30, hdbscan_min_samples=5,
         distance_metric='euclidean',
         polygon_fname=None):

    # import df and clean it
    df = pd.read_csv(os.path.join(path, df_for_clustering_fname))
    df = df.drop_duplicates(subset=['firstBlip_lat', 'firstBlip_lng'])  # drop duplicates
    if polygon_fname:  # take only area of the data, e.g. 'maps/mediterranean.geojson'
        df = df[df.apply(lambda x: is_in_polygon(x['firstBlip_lng'], x['firstBlip_lat'], polygon_fname), axis=1)]

    locations = df[['firstBlip_lat', 'firstBlip_lng']].to_numpy()

    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                min_samples=hdbscan_min_samples,
                                metric=distance_metric)
    clusterer.fit(locations)

    # TODO: generalize path
    ports_df = gpd.read_file('maps/ports.json')
    polygons_df = gpd.read_file('maps/polygons.geojson')  # WW polygons

    clust_polygons = polygenize_clusters_with_features(df, ports_df, locations, clusterer)
    clust_polygons = polygon_intersection(clust_polygons, polygons_df)
    if ACTIVITY == 'mooring':
        clust_polygons = calc_nearest_shore(clust_polygons, path_to_shoreline_file, method='euclidean')
        clust_polygons['dist_to_ww_poly'] = clust_polygons.geometry.apply(
            lambda x: calc_polygon_distance_from_nearest_ww_polygon(x, polygons_df))

    geo_df_clust_polygons = gpd.GeoDataFrame(clust_polygons)

    # save model and files
    pkl_model_fname = f'hdbscan_{hdbscan_min_cluster_zise}mcs_{hdbscan_min_samples}ms_{ACTIVITY}'
    clust_polygons_fname = pkl_model_fname + '_polygons.json'

    with open('models/' + pkl_model_fname + '.pkl', 'wb') as file:
        pickle.dump(clusterer, file)
    geo_df_clust_polygons.to_file('maps/' + clust_polygons_fname, driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
