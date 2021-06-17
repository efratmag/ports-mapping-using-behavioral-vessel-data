import math
import pandas as pd
from shapely.geometry import shape, MultiLineString, Polygon, MultiPolygon, MultiPoint
from scipy.spatial import Delaunay
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
from shapely import ops
from sklearn.metrics.pairwise import haversine_distances
import logging
from tqdm import tqdm
from kneed import KneeLocator

# TODO: verify all lat lngs are in right order

R = 6378.1  # Radius of the Earth
SQUARE_FOOT_IN_SQUARE_METRE = 10.7639

BRNG_N_E = 1.0472  # 60 degrees converted to radians.
BRNG_S_W = 4.18879  # 240 degrees converted to radians.

METERS_IN_DEG = 2 * math.pi * 6371000.0 / 360

UNIT_RESOLVER = {'sqmi': 1609.34, 'sqkm': 1000.0}
AREA_TYPE_RESOLVER = {'pwa': 'PortWaitingArea', 'ports': 'Port'}


def haversine(lonlat1, lonlat2):

    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return c * R


def polygon_from_points(points, polygenize_method, alpha=None):
    """ takes plat/lng array of points and create polygon from them """

    assert polygenize_method in ['alpha_shape', 'convex_hull'], \
        f'expect polygon_type="alpha_shape" or "convex_hull", got {polygenize_method}'

    poly = None
    if polygenize_method == 'alpha_shape':
        poly = alpha_shape(points, alpha)[0]
    elif polygenize_method == 'convex_hull':
        poly = MultiPoint(points).convex_hull
    return poly


def alpha_shape(points, alpha, only_outer=True):
    # if len(points) < 4:
    #     # When you have a triangle, there is no sense
    #     # in computing an alpha shape.
    #     return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = points  # np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        # area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        # circum_r = a*b*c/(4.0*area)
        area = math.sqrt(abs(s * (s - a) * (s - b) * (s - c)))
        if area != 0:
            circum_r = a * b * c / (4.0 * area)
        else:
            circum_r = np.Inf

        # Here's the radius filter.
        # print(circum_r)
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(ops.polygonize(m))
    return ops.cascaded_union(triangles), edge_points, edges


def calc_polygon_area_sq_unit(polygon, unit='sqkm'):

    """
    this function will calculate the square area of a polygon (Kilometers/ Miles)
    :param polygon: Polygon object
    :param unit: sqkm / sqmi
    :return:
    """

    avg_lat, polygon = polygon_to_meters(polygon)
    polygon_area = np.sqrt(polygon.area) / UNIT_RESOLVER[unit]
    polygon_area *= polygon_area

    return polygon_area


def calc_cluster_density(points):
    """
    :param points: all points in cluster
    :return: cluster density
    """

    distances = haversine_distances(np.radians(points))
    mean_squared_distane = np.square(distances).mean()
    mean_squared_distane_km = mean_squared_distane * R

    return 1 / mean_squared_distane_km


def polygon_intersection(clust_polygon, ww_polygons, type_of_area_mapped):
    """
    :param clust_polygon: df of clustering polygons
    :param ww_polygons: df of windward polygons
    :param type_of_area_mapped: ports vs pwa
    :return: geopandas dataframe with extra feature of intersection of polygons with windward's polygons
    """
    # choose relevant type of polygons
    ww_polygons = ww_polygons[ww_polygons.areaType == AREA_TYPE_RESOLVER[type_of_area_mapped]]

    intersection_value = 0
    temp_df = ww_polygons[ww_polygons.intersects(clust_polygon)]
    if not temp_df.empty:
        intersection_value = clust_polygon.intersection(temp_df.iloc[0]['geometry']).area / clust_polygon.area * 100

    return intersection_value


def calc_polygon_distance_from_nearest_port(polygon, ports_df):
    """takes a polygon and ports df,
     calculate haversine distances from ports to polygon,
     returns: the name of nearest port and distance from it"""
    ports_centroids = ports_df.loc[:, ['lng', 'lat']].to_numpy()
    polygon_centroid = (polygon.centroid.y, polygon.centroid.x)
    dists = [haversine(port_centroid, polygon_centroid) for port_centroid in ports_centroids]
    min_dist = np.min(dists)
    name_of_nearest_port = ports_df.loc[dists.index(min_dist), 'name']
    return min_dist, name_of_nearest_port


def filter_points_far_from_port(ports_df, port_name, points, idxs):
    """ calculate distance between port and the activity points related to it
    filters out points that are more than 200km away.
    used for destination based port waiting area clustering"""

    if port_name == 'Port Said East':  # fix specific bug in port said port
        port_name = 'Port Said'  # TODO: fix appropriately this bug in port name

    port_data = ports_df[ports_df.name == port_name]
    if port_data.shape[0] > 1:  # fix bug for duplicate port entries
        port_data = port_data[:1]

    port_centroid = [port_data.lat.item(), port_data.lon.item()]
    dists = np.asarray([haversine(port_centroid, loc) for loc in points])
    good_idxs = idxs[np.where(dists < 200)]
    points = points[np.where(dists < 200)]
    return points, good_idxs


def merge_polygons(geo_df):

    merged_polygons = gpd.GeoSeries(ops.cascaded_union(geo_df['geometry'])).loc[0]

    return merged_polygons


def calc_nearest_shore(poly, shoreline_polygon, method='euclidean'):

    """
    this function will calculate nearest point to shoreline layer for a given polygon
    :param poly: Polygon Object
    :param shoreline_polygon: shoreline layer polygon
    :param method: euclidean / haversine
    :return:
    """

    if poly.intersects(shoreline_polygon):
        distance = 0
        nearest_shore = {f'distance_from_shore_{method}': distance}

    else:
        nearest_polygons_points = nearest_points(shoreline_polygon, poly)
        point1, point2 = (nearest_polygons_points[0].y, nearest_polygons_points[0].x), \
                         (nearest_polygons_points[1].y, nearest_polygons_points[1].x)

        if method == 'euclidean':
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
        elif method == 'haversine':
            distance = haversine(point1, point2)
        else:
            raise ValueError('method must be "euclidean" or "haversine"')

        nearest_shore = {f'distance_from_shore_{method}': distance,
                         'nearest_shore_lat': point1[0],
                         'nearest_shore_lng': point1[1],
                         'nearest_point_lat': point2[0],
                         'nearest_point_lng': point2[1]}

    return nearest_shore


def calc_nearest_shore_bulk(df, shoreline_polygon, method='euclidean'):

    """
    this function will iterate over df with polygons and calculate nearest point to shoreline layer
    :param df: geopandas df with polygons
    :param shoreline_polygon: shoreline layer polygon
    :param method: euclidean / haversine
    :return:
    """

    results_list = []

    for row in tqdm(df['geometry'].iteritems()):
        index, poly = row

        if index % 100 == 0 and index != 0:
            logging.info(f'{index} instances was calculated')

        nearest_shore = calc_nearest_shore(poly, shoreline_polygon, method)
        results_list.append(nearest_shore)

    results_df = pd.DataFrame(results_list)
    shared_columns = set(results_df.columns).intersection(set(df.columns))
    df = df.drop(shared_columns, axis=1)
    df = pd.concat([df, results_df], axis=1)
    return df


def calc_polygon_distance_from_nearest_ww_polygon(polygon, ww_polygons_centroids):
    """takes a polygon and an array of ports centroids
    and returns the distance in km from the nearest port from the array"""
    polygon_centroid = (polygon.centroid.y, polygon.centroid.x)
    dists = [haversine(ww_poly_centroid, polygon_centroid) for ww_poly_centroid in ww_polygons_centroids]
    return np.min(dists)


def get_multipolygon_exterior(multipolygon):
    """
    this function will return exterior points of a multipolygon
    :param multipolygon:
    :return:
    """
    coordinates = []

    polygons_list = list(multipolygon)

    for polygon in polygons_list:
        coordinates.extend([(x[0], x[1]) for x in list(polygon.exterior.coords)])

    return coordinates


def polygon_to_wgs84(polygon, avg_lat=None):

    """
    this function will return polygon with wgs84 crs
    :param polygon: Polygon object
    :param avg_lat: average latitude value
    :return:
    """

    if not avg_lat:
        if isinstance(polygon, Polygon):

            avg_lat = get_avg_lat(polygon.exterior.coords)

        elif isinstance(polygon, MultiPolygon):

            avg_lat = get_avg_lat(get_multipolygon_exterior(polygon))

    def shape_to_wgs84(x, y, avg_lat):
        lng = x / math.cos(math.radians(avg_lat)) / METERS_IN_DEG
        lat = y / METERS_IN_DEG
        return lat, lng

    def to_wgs84(x, y):
        return tuple(reversed(shape_to_wgs84(x, y, avg_lat)))

    return avg_lat, shape(ops.transform(to_wgs84, polygon))


def polygon_to_meters(polygon, avg_lat=None):

    """
    this function will return polygon with meters crs
    :param polygon: Polygon object
    :param avg_lat: average latitude value
    :return:
    """

    if not avg_lat:

        if isinstance(polygon, Polygon):

            avg_lat = get_avg_lat(polygon.exterior.coords)

        elif isinstance(polygon, MultiPolygon):
            avg_lat = get_avg_lat(get_multipolygon_exterior(polygon))

    def shape_to_meters(lat, lng, avg_lat):
        x = lng * math.cos(math.radians(avg_lat)) * METERS_IN_DEG
        y = lat * METERS_IN_DEG
        return x, y

    def to_meters(lng, lat):
        return shape_to_meters(lat, lng, avg_lat)

    return avg_lat, shape(ops.transform(to_meters, polygon))


def get_avg_lat(coordinates):
    s = sum(c[1] for c in coordinates)
    return float(s) / len(coordinates)


def inflate_polygon(polygon, meters, resolution=4):

    """
    This function will inflate polygon by meters
    :param polygon: Polygon object
    :param meters: meters for the polygon to be inflated
    :param resolution: resolution value determines the number of segments used to approximate a quarter circle around a point.
    :return:
    """

    avg_lat, polygon = polygon_to_meters(polygon)
    polygon = polygon.buffer(meters, resolution=resolution)

    _, polygon = polygon_to_wgs84(polygon, avg_lat)

    return polygon


def merge_adjacent_polygons(geo_df, inflation_meter=1000, aggfunc='mean'):

    """
    This function will merge proximate polygons.
    First, polgons will be inflating by "inflation_meter", then intersected polygons will be merged
    :param geo_df: geopandas df with polygons
    :param inflation_meter:
    :param aggfunc: aggfunc for the polygons attributes aggregation
    :return:
    """

    logging.info('merge_adjacent_polygons - START')

    inflated_df = geo_df.apply(lambda x: inflate_polygon(x['geometry'], inflation_meter), axis=1)

    inflated_df = gpd.GeoDataFrame(inflated_df, columns=['geometry'])

    merged_inflated = merge_polygons(inflated_df)

    merged_inflated = [merged_inflated] if isinstance(merged_inflated, Polygon) else list(merged_inflated)

    merged_inflated = gpd.GeoDataFrame(merged_inflated, columns=['geometry'])

    merged_df = gpd.sjoin(geo_df, merged_inflated, how='left')

    merged_df = merged_df.dissolve(by='index_right', aggfunc=aggfunc)

    logging.info('merge_adjacent_polygons - END')

    return merged_inflated, merged_df


def calc_entropy(feature):
    """ takes categorical feature values and returns the column's entropy
    The maximum value of entropy is log𝑘, where 𝑘 is the number of categories you are using."""
    vc = pd.Series(feature).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc) / np.log(math.e)).sum()


def create_google_maps_link_to_centroid(centroid):

    """

    :param centroid: Point object
    :return: google maps link with the location
    """

    centroid_lat, centroid_lng = centroid.y, centroid.x
    return f'https://maps.google.com/?ll={centroid_lat},{centroid_lng}'


def optimize_polygon_by_probs(points, probs, polygon_type, alpha=4, s=1):

    all_prob = np.linspace(min(probs), 1, 20)

    metrics = []

    for prob in all_prob:
        probs_mask = probs >= prob
        relevant_points = points[probs_mask]
        if len(relevant_points) > 0:

            poly = polygon_from_points(relevant_points, polygon_type, alpha)

            area_size = calc_polygon_area_sq_unit(poly)
            metrics.append(area_size)

    kneedle = KneeLocator(all_prob, metrics, S=s, curve="convex", direction="decreasing")

    original_polygon = polygon_from_points(points, polygon_type, alpha)

    if kneedle.knee:

        final_points = points[probs >= kneedle.knee]
        points_removed = len(points) - len(final_points)
        poly = polygon_from_points(final_points, polygon_type, alpha)

    else:
        poly = original_polygon
        points_removed = 0

    return poly, original_polygon, kneedle.knee, points_removed, metrics
