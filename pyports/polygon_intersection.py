import geopandas as gpd


def polygon_intersection(geodf_polygons):
    """
    :param geodf_polygons:
    :return: geopandas dataframe with intersection of polygons with windward's polygons
    """

    pre_defined_polygons = gpd.read_file('maps/polygons.geojson')

    for i, clust_poly in enumerate(geodf_polygons.geometry):
        for j, ww_poly in enumerate(pre_defined_polygons.geometry):
            if clust_poly.intersects(ww_poly):
                geodf_polygons.loc[i,'intersection'] = clust_poly.intersection(ww_poly).area/clust_poly.area * 100

    return geodf_polygons
