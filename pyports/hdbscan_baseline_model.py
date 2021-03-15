import pandas as pd
import os
import hdbscan
import folium
from folium.vector_layers import CircleMarker
from colour import Color
from shapely.geometry import Point, MultiPoint, shape, Polygon
from shapely.ops import transform
import geopandas as gpd
import pickle
import logging
import fire
from pyports.geo_utils import is_in_polygon


FILE_NAME = 'df_for_clustering.csv'  # df with lat lng of all anchoring activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features/'  # features folder


def polygenize_clusters(locations, clusterer):

    """
    :param locations: lat lng of all relevant vessels behavior for clustering
    :param clusterer: fitted clustering model
    :return: df of polygons
    """

    clusters = clusterer.labels_

    clust_polygons = pd.DataFrame(columns=['id', 'num_points', 'probs_of_belonging_to_clust','polygon', 'geometry'])

    clust_polygons['probs_of_belonging_to_clust'] = clust_polygons['probs_of_belonging_to_clust'].astype(object)
    clust_polygons['polygon'] = clust_polygons['polygon'].astype(object)

    for clust in range(clusters.max()+1):
        points = locations[clusters == clust]
        polygon = MultiPoint(points).convex_hull

        clust_polygons.loc[clust, 'id'] = clust
        clust_polygons.loc[clust, 'num_points'] = len(points)
        clust_polygons.at[clust, 'probs_of_belonging_to_clust'] = clusterer.probabilities_[clusters == clust]
        clust_polygons.at[clust, 'polygon'] = gpd.GeoSeries([polygon]).__geo_interface__['features'][0]['geometry']

    clust_polygons['geometry'] = clust_polygons.apply(lambda x: shape(x['polygon']), axis=1)

    return clust_polygons


def show_cluster_map(cluster_id, clusterer, locations):

    blue = Color("blue")
    red = Color("red")
    color_range = list(blue.range_to(red, 10))

    my_map = folium.Map(prefer_canvas=True, tiles='CartoDB positron')

    clusters = clusterer.labels_
    outlier_scores = clusterer.outlier_scores_

    points = locations[clusters == cluster_id]
    scores = outlier_scores[clusters == cluster_id]

    for i in range(points.shape[0]):
        point = points[i]
        color = color_range[int(scores[i] * 10)]
        CircleMarker(point, radius=1, color=color.hex, tooltip="{:.2f}".format(scores[i])).add_to(map)

    min_lat, max_lat = points[:, 0].min(), points[:, 0].max()
    min_lon, max_lon = points[:, 1].min(), points[:, 1].max()
    my_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return my_map


def flip(x, y):
    """Flips the x and y coordinate values"""
    return y, x


def main(path, df_for_clustering_fname, hdbscan_min_cluster_zise=15, hdbscan_min_samples=1, polygon_fname=None, sub_area_name=None):


    # import df
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

    clust_polygons = polygenize_clusters(locations, clusterer)

    geo_df_clust_polygons = gpd.GeoDataFrame(clust_polygons.loc[:, ['id', 'num_points', 'geometry']])

    # fix lat lng #TODO: do it right from the start
    for poly in range(geo_df_clust_polygons.shape[0]):
        geo_df_clust_polygons.loc[poly, 'geometry'] = transform(flip, geo_df_clust_polygons.loc[poly, 'geometry'])

    # save model and files
    pickle.dump(clusterer, open('models/hdbscan_15mcs_1ms'), 'wb')
    clust_polygons.to_csv(os.path.join(path, 'clust_polygons.csv'))
    geo_df_clust_polygons.to_file(os.path.join(path, 'hdbscan_polygons.json'), driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

