import numpy as np
from sklearn import preprocessing
import logging
logging.basicConfig(level=logging.INFO)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_rank(candidate):

    rank = candidate['density'] * candidate['n_unique_vesselID'] * \
           sigmoid(candidate['dist_to_ww_poly'] / 100 - candidate['dist_to_ww_poly'] / 100)

    return rank


def filter_candidates(geo_df_clust_polygons, min_nunique_vessels=20, min_distance_from_ww_polygons_km=30,
                      max_distance_from_shore_km=5, remove_intersection_polygon=True):

    """

    :param geo_df_clust_polygons: geopandas df (output of the clustering)
    :param min_nunique_vessels: polygons with less then this threshold will be filtered
    :param min_distance_from_ww_polygons_km: polygons with distance below this threshold to other winward polygons will be filtered
    :param max_distance_from_shore_km: polygons with distance above this threshold to shoreline will be filtered
    :param remove_intersection_polygon: if True, polygons who's intersect with winward polygons will be filtered
    :return:
    """

    geo_df_clust_polygons = geo_df_clust_polygons[geo_df_clust_polygons['n_unique_vesselID'] >= min_nunique_vessels]

    geo_df_clust_polygons = geo_df_clust_polygons[
        geo_df_clust_polygons['dist_to_ww_poly'] >= min_distance_from_ww_polygons_km]

    if remove_intersection_polygon:
        geo_df_clust_polygons = geo_df_clust_polygons[geo_df_clust_polygons['pct_intersection'] == 0]

    if 'distance_from_shore_haversine' in geo_df_clust_polygons.columns:

        geo_df_clust_polygons = geo_df_clust_polygons[
            geo_df_clust_polygons['distance_from_shore_haversine'] <= max_distance_from_shore_km]
    else:
        logging.info('distance_from_shore_haversine is not exists, max_distance_from_shore_km filter skipped')

    return geo_df_clust_polygons


def main(geo_df_clust_polygons, debug=False, min_nunique_vessels=20, min_distance_from_ww_polygons_km=30,
         max_distance_from_shore_km=5, remove_intersection_polygon=True):

    if not debug:
        geo_df_clust_polygons = filter_candidates(geo_df_clust_polygons, min_nunique_vessels,
                                                  min_distance_from_ww_polygons_km, max_distance_from_shore_km,
                                                  remove_intersection_polygon)

    geo_df_clust_polygons['rank'] = geo_df_clust_polygons.apply(lambda row: calc_rank(row), axis=1)

    x = geo_df_clust_polygons['rank'].values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    geo_df_clust_polygons['rank_scaled'] = min_max_scaler.fit_transform(x)

    geo_df_clust_polygons = geo_df_clust_polygons.sort_values('rank_scaled', ascending=False)

    return geo_df_clust_polygons
