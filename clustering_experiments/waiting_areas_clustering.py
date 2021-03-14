from pyports.area_snaptshot.area_snapshot import today_str
from pyports.area_snaptshot.map_configs import MAP_CONFIG_CLUSTERING
from pyports.geo_utils import haversine, alpha_shape, calc_polygon_area_sqmi
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
import fire
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from keplergl import KeplerGl
import logging


def remove_outliers(df, std=3.5, col='firstBlip'):
    """
    a function that removes far points from the cluster by
    calc distance from center and dropping faraway points

    :param df: data frame
    :param std: points with higher x std distance from cluster center will be dropped
    :param col: firstBlip / lastBlip
    :return:
    """
    points = df[[col+'_lng', col+'_lat']].to_numpy()
    distances = distance_matrix(points, np.expand_dims(points.mean(axis=0), axis=0))
    df['dist_from_c'] = distances

    df = df.loc[(df['dist_from_c'] <= (df['dist_from_c'].std() * std))]

    return df


def points_to_polygon(df, outliers_std, col, alpha):

    df = remove_outliers(df, outliers_std, col)
    poly, _, _ = alpha_shape(df[[col + '_lng', col + '_lat']].to_numpy(), alpha=alpha)

    return poly


def main(import_path, export_path, ports=False, col='firstBlip', dbscan_eps=2, dbscan_min_samples=10, outliers_std=3.5, alpha=2):

    """

    :param import_path:
    :param export_path:
    :param ports: if True, will generate polygons to ports as well as waiting areas
    :param col: FirstBlip / LastBlip
    :param dbscan_eps:
    :param dbscan_min_samples:
    :param outliers_std:
    :param alpha: alpha
    :return:
    """

    map_file_name = f'clustering_map_{today_str()}.html'

    clustering_file_name = f'clustering_points_{today_str()}.csv'
    polygons_file_name = f'clustering_polygons_{today_str()}.geojson'

    kepler_map = KeplerGl()

    logging.info(f'loading file...')
    df = pd.read_csv(import_path)

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[col+'_lng'],
                                                          df[col+'_lat']))

    activities = ['anchoring']

    if ports:
        activities.append('mooring')

    activities_df = df[(df['activity'].isin(activities)) & (df['vessel_class_calc'].isin(['Tanker', 'Cargo']))]

    avg_lat = activities_df[col+'_lat'].mean()
    avg_lng = activities_df[col + '_lng'].mean()

    #  define the map center by the given coordinates
    MAP_CONFIG_CLUSTERING['config']['mapState']['latitude'] = avg_lat
    MAP_CONFIG_CLUSTERING['config']['mapState']['longitude'] = avg_lng

    logging.info(f'performing clustering, total points {len(activities_df)}...')
    blips_distance_matrix = squareform(pdist(activities_df[[col+'_lat', col+'_lng']], (lambda u, v: haversine(u, v))))
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="precomputed").fit(blips_distance_matrix)

    activities_df['labels'] = dbscan.labels_
    activities_df['labels'] = 'cluster_' + activities_df['labels'].astype(str)

    poly_list = []

    logging.info(f'generating polygons...')
    for label in activities_df[(activities_df['labels'] != 'cluster_-1')]['labels'].unique():

        cluster_df = activities_df[activities_df['labels'] == label]

        anchoring = cluster_df[(cluster_df['activity'] == 'anchoring')]
        wa_poly = points_to_polygon(anchoring, outliers_std, col, alpha)
        poly_list.append({'cluster': label,
                          'type': 'Waiting Area',
                          'geometry': wa_poly,
                          'area_sqmi': calc_polygon_area_sqmi(wa_poly),
                          'mean_duration (hours)': anchoring['duration'].mean()})

        if ports:
            try:
                mooring = cluster_df[(cluster_df['activity'] == 'mooring')]
                mooring = mooring[~mooring['geometry'].within(wa_poly)]
                port_poly = points_to_polygon(mooring, 1.1, col, alpha)

                poly_list.append({'cluster': label,
                                  'type': 'Port',
                                  'geometry': port_poly,
                                  'area_sqmi': calc_polygon_area_sqmi(port_poly),
                                  'mean_duration (hours)': mooring['duration'].mean()})

            except Exception as e:
                logging.info(f'failed to generate polygons for port {label}, reason - {e}')

    polygons_df = gpd.GeoDataFrame(poly_list)

    activities_df.to_csv(os.path.join(export_path, clustering_file_name), index=False)
    polygons_df.to_file(os.path.join(export_path, polygons_file_name), driver='GeoJSON')

    kepler_map.add_data(activities_df, 'clusters')
    kepler_map.add_data(df, 'all_activities')
    kepler_map.add_data(polygons_df, 'polygons')
    kepler_map.save_to_html(file_name=os.path.join(export_path, map_file_name), config=MAP_CONFIG_CLUSTERING)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

