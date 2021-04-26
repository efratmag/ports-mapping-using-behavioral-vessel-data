import numpy as np
from sklearn import preprocessing


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_rank(candidate):

    rank = candidate['density'] * candidate['n_unique_vesselID'] * \
           sigmoid(candidate['dist_to_ww_poly'] / 100 - candidate['dist_to_ww_poly'] / 100)

    return rank

#TODO add filters functions

def main(geo_df_clust_polygons):

    geo_df_clust_polygons['rank'] = geo_df_clust_polygons.apply(lambda row: calc_rank(row), axis=1)

    x = geo_df_clust_polygons['rank'].values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    geo_df_clust_polygons['rank_scaled'] = min_max_scaler.fit_transform(x)

    geo_df_clust_polygons = geo_df_clust_polygons.sort_values('rank_scaled', ascending=False)

    return geo_df_clust_polygons
