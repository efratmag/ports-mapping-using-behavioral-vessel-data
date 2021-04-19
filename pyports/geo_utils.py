import math
import pandas as pd
import json
from shapely.geometry import shape, Point, MultiLineString, Polygon, MultiPolygon
from scipy.spatial import Delaunay
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
from shapely import ops
from geopy.distance import distance, great_circle
from scipy.spatial.distance import pdist
from numba import jit, prange
from scipy.sparse import dok_matrix
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)


R = 6378.1  # Radius of the Earth
SQUARE_FOOT_IN_SQUARE_METRE = 10.7639

BRNG_N_E = 1.0472  # 60 degrees converted to radians.
BRNG_S_W = 4.18879  # 240 degrees converted to radians.

METERS_IN_DEG = 2 * math.pi * 6371000.0 / 360

UNIT_RESOLVER = {'sqmi': 1609.34, 'sqkm': 1000.0}


def calc_dest_point(lat, lng, brng, d=15):
    """
    Calculate destination lat,lng for a given location, direction and distance
    :param lat: latitude
    :param lng: longitude
    :param brng: degrees converted to radians
    :param d: distance in Km
    :return:
    """

    lat = math.radians(lat)
    lng = math.radians(lng)

    dest_lat = math.asin(math.sin(lat) * math.cos(d / R) +
                     math.cos(lat) * math.sin(d / R) * math.cos(brng))

    dest_lng = lng + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat),
                             math.cos(d / R) - math.sin(lat) * math.sin(dest_lat))

    dest_lat = math.degrees(dest_lat)
    dest_lng = math.degrees(dest_lng)

    return dest_lat, dest_lng


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


def get_bounding_box(lat, lng, d=15):

    """
    Calculate the bounding box for given a point and distance
    :param lat: latitude
    :param lng: longitude
    :param d: distance in Km
    :return:
    """

    lat_n_e, lng_n_e = calc_dest_point(lat, lng, BRNG_N_E, d=d)
    lat_s_w, lng_s_w = calc_dest_point(lat, lng, BRNG_S_W, d=d)

    return lng_s_w, lat_s_w, lng_n_e, lat_n_e


def isin_box(lat, lng, bounds):

    """
    Check if a point located within a given bounding box
    :param lat: latitude
    :param lng: longitude
    :param bounds: bounding box coordinates
    :return:
    """

    x1, x2, x3, x4 = bounds

    within = False

    if x2 < lat < x4:
        if x1 < lng < x3:
            within = True

    return within


def is_in_polygon(lng, lat, polygon_fname):

    """
    checks if a point is inside a polygon
    :param lng: long of point
    :param lat: latitude of point
    :param polygon_fname: the polygon file name for which to test if the point is inside of. can take manually defined in geojson.io
    :return: boolean
    """

    with open(polygon_fname) as f:
        polygon = json.load(f)

    point = Point(lng, lat)

    for feature in polygon['features']:
        poly = shape(feature['geometry'])
        verdict = poly.contains(point)

    return verdict


def is_in_polygon_features(df):
    df['firstBlip_in_polygon'] = df['firstBlip_polygon_id'].notna()

    conditions = [
        (df['firstBlip_in_polygon'] == True) & (df['lastBlip_polygon_id'].isna() == True),
        (df['firstBlip_in_polygon'] == False) & (df['lastBlip_polygon_id'].isna() == True),
        (df['lastBlip_polygon_id'].isna() == False)
    ]

    choices = ['not_ended', 'False', 'True']
    df['lastBlip_in_polygon'] = np.select(conditions, choices)

    return df


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

    polygon = polygon_to_meters(polygon)
    polygon_area = np.sqrt(polygon[1].area) / UNIT_RESOLVER[unit]
    polygon_area *= polygon_area

    return polygon_area


def calc_cluster_density(points):
    """
    :param points: all points in cluster
    :return: cluster density
    """

    distances = pdist(points)
    total_distance = distances.sum()
    n_pairwise_comparisons = len(distances)

    return n_pairwise_comparisons / total_distance


def polygon_intersection(clust_polygons, ww_polygons):
    """
    :param clust_polygons: df of clustering polygons
    :param ww_polygons: df of windward polygons
    :return: geopandas dataframe with extra feature of intersection of polygons with windward's polygons
    """
    for i, clust_poly in enumerate(clust_polygons.geometry):
        clust_poly = shapely.wkt.loads(clust_poly)
        for j, ww_poly in enumerate(ww_polygons.geometry):
            if clust_poly.intersects(ww_poly):
                clust_polygons.loc[i, 'intersection'] = clust_poly.intersection(ww_poly).area/clust_poly.area * 100
    return clust_polygons


def flip(x, y):
    """Flips the x and y coordinate values"""
    return y, x


def get_ports_centroid_array(ports_df):
    """ Returns array of ports centroids"""
    ports_centroids = np.array(
        [ports_df.center_coordinates.map(lambda x: x[0]),
         ports_df.center_coordinates.map(lambda x: x[1])]).transpose()
    return ports_centroids


def calc_polygon_distance_from_nearest_port(polygon, ports_df):
    """takes a polygon and ports df,
     calculate haversine distances from ports to polygon,
     returns: the name of nearest port and distance from it"""
    ports_centroids = ports_df.loc[:, ['lng', 'lat']].to_numpy()
    polygon_centroid = (polygon.centroid.x, polygon.centroid.y)
    dists = [haversine(port_centroid, polygon_centroid) for port_centroid in ports_centroids]
    min_dist = np.min(dists)
    name_of_nearest_port = ports_df.loc[dists.index(min_dist), 'name']
    return min_dist, name_of_nearest_port


def geodesic_distance(x, y):
    """ distance metric using geopy geodesic metric. points need to be ordered (lat,lng)"""
    geo_dist = distance((x[0], x[1]), (y[0], y[1]))
    return geo_dist.kilometers


def great_circle_distance(x,y):
    """ distance metric using geopy great circle metric.
    it is much faster than geodesic but a bit less accurate.
    points need to be ordered (lat,lng)"""
    circle_dist = great_circle((x[0], x[1]), (y[0], y[1]))
    return circle_dist.kilometers


@jit(parallel=True)
def haversine_distances_parallel(d):
    """Numba version of haversine distance."""

    dist_mat = np.zeros((d.shape[0], d.shape[0]))

    # We parallelize outer loop to keep threads busy
    for i in prange(d.shape[0]):
        for j in range(i+1, d.shape[0]):
            sin_0 = np.sin(0.5 * (d[i, 0] - d[j, 0]))
            sin_1 = np.sin(0.5 * (d[i, 1] - d[j, 1]))
            cos_0 = np.cos(d[i, 0]) * np.cos(d[j, 0])
            dist_mat[i, j] = 2 * np.arcsin(np.sqrt(sin_0 * sin_0 + cos_0 * sin_1 * sin_1))
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


def polygons_to_multi_lines(polygons_df):

    polygons_multi_line = ops.linemerge(polygons_df['geometry'].boundary.values)

    return polygons_multi_line


def merge_polygons(geo_df):

    merged_polygons = gpd.GeoSeries(ops.cascaded_union(geo_df['geometry'])).loc[0]

    return merged_polygons


def calc_nearest_shore(df, path_to_shoreline_file, method='euclidean'):

    logging.info('loading and processing shoreline file - START')
    shoreline_df = gpd.read_file(path_to_shoreline_file)
    shoreline_multi_polygon = merge_polygons(shoreline_df)
    logging.info('loading and processing shoreline file - END')

    results_list = []

    for row in tqdm(df['geometry'].iteritems()):
        index, poly = row

        if index % 100 == 0 and index != 0:
            logging.info(f'{index} instances was calculated')
        if poly.intersects(shoreline_multi_polygon):
            distance = 0
            results_list.append({f'distance_from_shore_{method}': distance})
        else:
            nearest_polygons_points = nearest_points(shoreline_multi_polygon, poly)
            point1, point2 = (nearest_polygons_points[0].y, nearest_polygons_points[0].x), \
                             (nearest_polygons_points[1].y, nearest_polygons_points[1].x)
            if method == 'euclidean':
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
            elif method == 'haversine':
                distance = haversine(point1, point2)
            else:
                raise ValueError('method must be "euclidean" or "haversine"')

            results_list.append({f'distance_from_shore_{method}': distance,
                                 'nearest_shore_lat': point1[0],
                                 'nearest_shore_lng': point1[1],
                                 'nearest_point_lat': point2[0],
                                 'nearest_point_lng': point2[1]})

    results_df = pd.DataFrame(results_list)
    shared_columns = set(results_df.columns).intersection(set(df.columns))
    df = df.drop(shared_columns, axis=1)
    df = pd.concat([df, results_df], axis=1)
    return df


def calc_polygon_distance_from_nearest_ww_polygon(polygon, polygons_df):
    """takes a polygon and an array of ports centroids
    and returns the distance in km from the nearest port from the array"""
    ww_polygons_centroids = np.array([polygons_df.geometry.centroid.x, polygons_df.geometry.centroid.y]).T
    polygon_centroid = (polygon.centroid.x, polygon.centroid.y)
    dists = [haversine(ww_poly_centroid, polygon_centroid) for ww_poly_centroid in ww_polygons_centroids]
    return np.min(dists)


@jit(parallel=True)
def haversine_distances_parallel_sparse(d, threshold=7):
    """Numba version of haversine distance.
        Generates sparse matrix of pairwise distances by only adding points which are less than x (threshold) km afar"""

    dist_mat = dok_matrix((d.shape[0], d.shape[0]))

    for i in prange(d.shape[0]):
        for j in range(i+1, d.shape[0]):
            sin_0 = np.sin(0.5 * (d[i, 0] - d[j, 0]))
            sin_1 = np.sin(0.5 * (d[i, 1] - d[j, 1]))
            cos_0 = np.cos(d[i, 0]) * np.cos(d[j, 0])
            hav_dist = 2 * np.arcsin(np.sqrt(sin_0 * sin_0 + cos_0 * sin_1 * sin_1))
            # only add distance if below the threshold
            if hav_dist * R < threshold:  # R is the radius of earth in kilometers
                dist_mat[i, j] = hav_dist

    dist_mat = dist_mat.tocoo()

    return dist_mat


def get_multipolygon_exterior(multipolygon):
    coordinates = []

    polygons_list = list(multipolygon)

    for polygon in polygons_list:
        coordinates.extend([(x[0], x[1]) for x in list(polygon.exterior.coords)])

    return coordinates


def polygon_to_wgs84(polygon, avg_lat=None):

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

    avg_lat, polygon = polygon_to_meters(polygon)
    polygon = polygon.buffer(meters, resolution=resolution)

    _, polygon = polygon_to_wgs84(polygon, avg_lat)

    return polygon


def merge_adjacent_polygons(geo_df, inflation_meter=1000, aggfunc='mean'):

    inflated_df = geo_df.apply(lambda x: inflate_polygon(x['geometry'], inflation_meter), axis=1)

    inflated_df = gpd.GeoDataFrame(inflated_df, columns=['geometry'])

    merged_inflated = merge_polygons(inflated_df)

    merged_inflated = [merged_inflated] if isinstance(merged_inflated, Polygon) else list(merged_inflated)

    merged_inflated = gpd.GeoDataFrame(merged_inflated, columns=['geometry'])

    merged_df = gpd.sjoin(geo_df, merged_inflated, how='left')

    merged_df = merged_df.dissolve(by='index_right', aggfunc=aggfunc)

    return merged_inflated, merged_df


def calc_entropy(feature):
    """ takes categorical feature values and returns the column's entropy"""
    vc = pd.Series(feature).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc) / np.log(math.e)).sum()


