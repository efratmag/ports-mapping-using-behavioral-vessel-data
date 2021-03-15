import pandas as pd
import os
import hdbscan
from shapely.geometry import MultiPoint
from shapely.ops import transform
import geopandas as gpd
import pickle
import logging
import fire
from pyports.geo_utils import is_in_polygon, calc_polygon_area_sq_unit


FILE_NAME = 'df_for_clustering.csv'  # df with lat lng of all anchoring activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features/'  # features folder

geod = Geod(ellps="WGS84")


def polygenize_clusters(df_for_clustering, locations, clusterer):

    """
    :param df_for_clustering: df used for clustering
    :param locations: lat lng of all relevant vessels behavior for clustering
    :param clusterer: fitted clustering model
    :return: df of polygons
    """

    clusters = clusterer.labels_

    clust_polygons = pd.DataFrame(columns=['id', 'num_points', 'mean_duration',
                                           'probs_of_belonging_to_clust','polygon',
                                           'geometry', 'area_sqkm', 'density'])

    clust_polygons['probs_of_belonging_to_clust'] = clust_polygons['probs_of_belonging_to_clust'].astype(object)
    clust_polygons['polygon'] = clust_polygons['polygon'].astype(object)

    for clust in range(clusters.max()+1):
        points = locations[clusters == clust]
        polygon = MultiPoint(points).convex_hull

        clust_polygons.loc[clust, 'id'] = clust
        clust_polygons.loc[clust, 'num_points'] = len(points)
        clust_polygons.loc[clust, 'mean_duration'] = df_for_clustering.loc[clusters == clust, 'duration'].mean()
        clust_polygons.at[clust, 'probs_of_belonging_to_clust'] = clusterer.probabilities_[clusters == clust]
        clust_polygons.at[clust, 'polygon'] = gpd.GeoSeries([polygon]).__geo_interface__['features'][0]['geometry']
        clust_polygons.at[clust, 'geometry'] = polygon
        clust_polygons.loc[clust, 'area_sqkm'] = calc_polygon_area_sq_unit(polygon)
        clust_polygons.loc[clust, 'density'] = clust_polygons.loc[clust, 'area_sqkm'] / len(points)

    #clust_polygons['geometry'] = clust_polygons.apply(lambda x: shape(x['polygon']), axis=1)

    return clust_polygons


def flip(x, y):
    """Flips the x and y coordinate values"""
    return y, x


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

    # save model and files
    #pickle.dump(clusterer, open('models/hdbscan_15mcs_1ms'), 'wb')
    #clust_polygons.to_csv(os.path.join(path, 'clust_polygons.csv'))
    #geo_df_clust_polygons.to_file(os.path.join(path, 'hdbscan_polygons.json'), driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

