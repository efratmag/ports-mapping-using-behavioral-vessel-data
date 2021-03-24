import pandas as pd
import os
import hdbscan
from shapely.geometry import MultiPoint
from shapely.ops import transform
import geopandas as gpd
import pickle
import logging
import fire
from pyports.geo_utils import *


FILE_NAME = 'df_for_clustering.csv'  # df with lat lng of all anchoring activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features/'  # features folder


def polygenize_clusters(polygons_df, ports_df, df_for_clustering, locations, clusterer):

    """
    :param df_for_clustering: df used for clustering
    :param locations: lat lng of all relevant vessels behavior for clustering
    :param clusterer: fitted clustering model
    :return: df of polygons
    """

    ports_centroids = ports_df.loc[:, ['lng', 'lat']].to_numpy()

    clusters = clusterer.labels_

    clust_polygons = pd.DataFrame()

    for cluster in tqdm(range(clusters.max() + 1), position=0, leave=True):
        points = locations[clusters == cluster]
        polygon = MultiPoint(points).convex_hull

        clust_polygons.loc[cluster, 'label'] = f'cluster {cluster}'
        clust_polygons.at[cluster, 'probs_of_belonging_to_clust'] = \
            transform_numbers_array_to_string(clusterer.probabilities_[clusters == cluster])
        clust_polygons.at[cluster, 'geometry'] = polygon
        clust_polygons.loc[cluster, 'num_points'] = len(points)
        clust_polygons.loc[cluster, 'area_sqkm'] = calc_polygon_area_sq_unit(polygon)
        clust_polygons.loc[cluster, 'density'] = calc_cluster_density(points)
        clust_polygons.loc[cluster, 'mean_duration'] = \
            df_for_clustering.loc[clusters == cluster, 'duration'].mean()
        clust_polygons.loc[cluster, 'median_duration'] = \
            df_for_clustering.loc[clusters == cluster, 'duration'].median()
        clust_polygons.loc[cluster, 'distance_from_nearest_port'] = \
            calc_polygon_distance_from_nearest_port(polygon, ports_centroids)
        clust_polygons.loc[cluster, 'n_unique_vesselID'] = \
            df_for_clustering.loc[clusters == cluster, 'vesselId'].nunique()
        clust_polygons.loc[cluster, 'percent_unique_vesselID'] = \
            clust_polygons.loc[cluster, 'n_unique_vesselID'] / len(points)
        clust_polygons.at[cluster, 'vesselIDs'] = \
            ','.join(df_for_clustering.loc[clusters == cluster, 'vesselId'].to_numpy())

    clust_polygons = polygon_intersection(clust_polygons, polygons_df)

    return clust_polygons


def main(path, df_for_clustering_fname, hdbscan_min_cluster_zise=15, hdbscan_min_samples=1, polygon_fname=None, sub_area_name=None):

    # import df and clean it
    df = pd.read_csv(os.path.join(path, df_for_clustering_fname))
    df = df.drop_duplicates(subset=['firstBlip_lat', 'firstBlip_lng']) # drop duplicates
    if polygon_fname: # take only area of the data, e.g. 'maps/mediterranean.geojson'
        df = df[df.apply(lambda x: is_in_polygon(x['firstBlip_lng'], x['firstBlip_lat'], polygon_fname), axis=1)]

    locations = df[['firstBlip_lat', 'firstBlip_lng']].to_numpy()

    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_zise,
                                min_samples=hdbscan_min_samples,
                                metric='euclidean')
    clusterer.fit(locations)

    clust_polygons = polygenize_clusters(df, locations, clusterer)

    geo_df_clust_polygons = gpd.GeoDataFrame(clust_polygons.drop(columns = ['polygon','probs_of_belonging_to_clust']))

    # fix lat lng #TODO: do it right from the start
    for poly in range(geo_df_clust_polygons.shape[0]):
        geo_df_clust_polygons.loc[poly, 'geometry'] = transform(flip, geo_df_clust_polygons.loc[poly, 'geometry'])

    geo_df_clust_polygons = polygon_intersection(geo_df_clust_polygons)

    # save model and files
    #pickle.dump(clusterer, open('models/hdbscan_15mcs_1ms'), 'wb')
    #clust_polygons.to_csv(os.path.join(path, 'clust_polygons.csv'))
    geo_df_clust_polygons.to_file(os.path.join(path, 'hdbscan_polygons.json'), driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

