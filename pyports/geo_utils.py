from pyports.constants import R, UNIT_RESOLVER, AREA_TYPE_RESOLVER, METERS_IN_DEG
import math
import pandas as pd
from shapely.geometry import shape, MultiLineString, Polygon, MultiPolygon, MultiPoint, Point
from scipy.spatial import Delaunay
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
from shapely import ops
from sklearn.metrics.pairwise import haversine_distances
import logging
from kneed import KneeLocator
from typing import Tuple, Union
from tqdm import tqdm
tqdm.pandas()


def haversine(lonlat1: Tuple[float, float], lonlat2: Tuple[float, float]) -> float:

    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
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


def polygon_from_points(points: np.array, polygenize_method: str, alpha: int = None) -> Polygon:

    """
    Takes lat/lng array of points and creates polygon from them, using alpha_shape or convex_hull method.
    """

    assert polygenize_method in ['alpha_shape', 'convex_hull'], \
        f'expect polygon_type="alpha_shape" or "convex_hull", got {polygenize_method}'

    poly = None
    if polygenize_method == 'alpha_shape':
        poly = alpha_shape(points, alpha)[0]
    elif polygenize_method == 'convex_hull':
        poly = MultiPoint(points).convex_hull

    return poly


def alpha_shape(points: np.array, alpha: int, only_outer: bool = True):

    # TODO: add type hints - abir's function
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


def calc_polygon_area_sq_unit(polygon: Polygon, unit: str = 'sqkm') -> float:

    """
    Calculate the square area of a polygon (Kilometers/ Miles)
    :param polygon: Polygon object
    :param unit: sqkm / sqmi
    :return: area in squared unit
    """

    avg_lat, polygon = polygon_to_meters(polygon)  # convert to meters
    polygon_area = np.sqrt(polygon.area) / UNIT_RESOLVER[unit]
    polygon_area *= polygon_area  # squared

    return polygon_area


def calc_cluster_density(points: np.array) -> float:

    """
    Takes all points in cluster and compute their density as the inverse of the mean squared distance between them.
    :param points: all points in cluster
    :return: cluster density
    """

    distances = haversine_distances(np.radians(points))  # pairwise haversine distances between all points in cluster
    mean_squared_distance = np.square(distances).mean()  # mean squared of all distances in cluster
    mean_squared_distance_km = mean_squared_distance * R  # convert to meters

    return 1 / mean_squared_distance_km


def polygon_intersection(cluster_polygon: Union[Polygon, MultiPolygon], ww_polygons: gpd.GeoDataFrame,
                         type_of_area_mapped: str) -> float:
    """
    Calculate percent intersection between new cluster polygon and existing ww polygons.
    :param cluster_polygon: polygon from clustering
    :param ww_polygons: df of windward polygons
    :param type_of_area_mapped: ports vs pwa
    :return: geopandas dataframe with extra feature of intersection of polygons with windward's polygons
    """
    # choose relevant type of polygons
    ww_polygons = ww_polygons[ww_polygons.polygon_area_type == AREA_TYPE_RESOLVER[type_of_area_mapped]]

    intersection_value = 0
    temp_df = ww_polygons[ww_polygons.intersects(cluster_polygon)]  # find intersection
    if not temp_df.empty:  # if there is intersection with ww polygons
        intersection_value = cluster_polygon.intersection(temp_df.iloc[0]['geometry']).area \
                             / cluster_polygon.area * 100  # calculate % of intersection

    return intersection_value


def calc_polygon_distance_from_nearest_port(polygon: Union[Polygon, MultiPolygon], ports_df: pd.DataFrame) -> \
        Tuple[float, str]:

    """
    Takes a polygon and ports df, calculate haversine distances from ports to polygon,
    returns the name of nearest port and distance from it.
    """

    ports_centroids = ports_df.loc[:, ['lng', 'lat']].to_numpy()  # find ports centroids
    polygon_centroid = (polygon.centroid.y, polygon.centroid.x)  # get polygon centroid
    dists = [haversine(port_centroid, polygon_centroid) for port_centroid in ports_centroids]  # calculate haversine
    # distances between polygon and ports
    min_dist = np.min(dists)  # find minimal distance
    name_of_nearest_port = ports_df.loc[dists.index(min_dist), 'name']  # find name of port in minimal distance

    return min_dist, name_of_nearest_port


def filter_points_far_from_port(ports_df: pd.DataFrame, port_name: str, points: np.array, idxs: list,
                                max_dist: int = 200) -> Tuple[np.array, list]:

    """
    Calculate distance between port and the activity points related to it, and filter out points that are more than
    max_dist km (deafult = 200km) away from it.
    Used for destination based port waiting area clustering.
    """

    if port_name == 'Port Said East':  # fix specific bug in port said port
        port_name = 'Port Said'  # TODO: fix appropriately this bug in port name

    port_data = ports_df[ports_df.name == port_name]
    if port_data.shape[0] > 1:  # fix bug for duplicate port entries
        port_data = port_data[:1]

    port_centroid = (port_data.lat.item(), port_data.lon.item())  # get port centroid
    dists = np.asarray([haversine(port_centroid, loc) for loc in points])  # get array of all haversine distances
    good_idxs = idxs[np.where(dists < max_dist)]  # specify indices of points that are less than max_dist away
    points = points[np.where(dists < max_dist)]  # get points

    return points, good_idxs


def merge_polygons(geo_df: gpd.GeoDataFrame) -> Union[Polygon, MultiPolygon]:

    """
    Merge GeoDataFrame into one Polygon/MultiPolygon Object.
    """

    merged_polygons = gpd.GeoSeries(ops.cascaded_union(geo_df['geometry'])).loc[0]

    return merged_polygons


def calc_nearest_shore(cluster_polygon: Polygon, shoreline_polygon: MultiPolygon, method: str = 'haversine') -> dict:

    """
    Calculate nearest point to shoreline layer for a given polygon.
    :param cluster_polygon: Polygon Object from clustering
    :param shoreline_polygon: shoreline layer polygon
    :param method: euclidean / haversine
    :return:
    """

    if cluster_polygon.intersects(shoreline_polygon):  # check for intersection
        distance = 0
        nearest_shore = {f'distance_from_shore_{method}': distance}

    else:
        nearest_polygons_points = nearest_points(shoreline_polygon, cluster_polygon)  # find nearest point
        point1, point2 = (nearest_polygons_points[0].y, nearest_polygons_points[0].x), \
                         (nearest_polygons_points[1].y, nearest_polygons_points[1].x)

        if method == 'euclidean':
            distance = np.linalg.norm(np.array(point1) - np.array(point2)) # calculates nearest point distance euclidean
        elif method == 'haversine':
            distance = haversine(point1, point2)  # calculates nearest point distance haversine
        else:
            raise ValueError('method must be "euclidean" or "haversine"')

        nearest_shore = {f'distance_from_shore_{method}': distance,
                         'nearest_shore_lat': point1[0],
                         'nearest_shore_lng': point1[1],
                         'nearest_point_lat': point2[0],
                         'nearest_point_lng': point2[1]}

    return nearest_shore


def calc_polygon_distance_from_nearest_ww_polygon(cluster_polygon: Union[Polygon, MultiPolygon], ww_polygons_centroids: np.array) -> float:

    """
    Takes a polygon and an array of ports centroids
    and returns the distance in km from the nearest port from the array
    """
    polygon_centroid = (cluster_polygon.centroid.y, cluster_polygon.centroid.x)
    dists = [haversine(ww_poly_centroid, polygon_centroid) for ww_poly_centroid in ww_polygons_centroids]

    return np.min(dists)


def get_multipolygon_exterior(multipolygon: MultiPolygon) -> list:

    """
    Get exterior points of a multipolygon.
    """
    coordinates = []

    polygons_list = list(multipolygon)  # multipolygon to list of polygons

    for polygon in polygons_list:
        coordinates.extend([(x[0], x[1]) for x in list(polygon.exterior.coords)])

    return coordinates


def polygon_to_wgs84(polygon: Union[Polygon, MultiPolygon], avg_lat: float = None) -> Tuple[float, Union[Polygon,MultiPolygon]]:

    """
    Return polygon with wgs84 crs (coordinate system)
    :param polygon: Polygon object
    :param avg_lat: average latitude value
    :return:
    """

    if not avg_lat:
        if isinstance(polygon, Polygon):  # get average latitude of the polygon

            avg_lat = get_avg_lat(polygon.exterior.coords)  # get average latitude of the polygon

        elif isinstance(polygon, MultiPolygon):

            avg_lat = get_avg_lat(get_multipolygon_exterior(polygon))

    def shape_to_wgs84(x, y, avg_lat):  # convert to wgs84
        lng = x / math.cos(math.radians(avg_lat)) / METERS_IN_DEG
        lat = y / METERS_IN_DEG
        return lat, lng

    def to_wgs84(x, y):
        return tuple(reversed(shape_to_wgs84(x, y, avg_lat)))  # apply wgs84 conversion

    return avg_lat, shape(ops.transform(to_wgs84, polygon))


def polygon_to_meters(polygon: Union[Polygon, MultiPolygon], avg_lat: float = None) -> Tuple[float, Union[Polygon,MultiPolygon]]:

    """
    Return polygon with meters crs
    :param polygon: Polygon object
    :param avg_lat: average latitude value
    :return:
    """

    if not avg_lat:

        if isinstance(polygon, Polygon):

            avg_lat = get_avg_lat(polygon.exterior.coords)  # get average latitude of the polygon

        elif isinstance(polygon, MultiPolygon):
            avg_lat = get_avg_lat(get_multipolygon_exterior(polygon))  # get average latitude of the multipolygon

    def shape_to_meters(lat, lng, avg_lat):  # convert to meters
        x = lng * math.cos(math.radians(avg_lat)) * METERS_IN_DEG
        y = lat * METERS_IN_DEG
        return x, y

    def to_meters(lng, lat):
        return shape_to_meters(lat, lng, avg_lat)  # apply meter conversion

    return avg_lat, shape(ops.transform(to_meters, polygon))


def get_avg_lat(coordinates: list) -> float:

    """
    Calculate average latitude for a list of coordinates
    """
    s = sum(c[1] for c in coordinates)

    return float(s) / len(coordinates)


def inflate_polygon(polygon: Union[Polygon, MultiPolygon], meters: Union[int, float], resolution: int = 4) \
        -> Union[Polygon, MultiPolygon]:

    """
    This function will inflate polygon by meters
    :param polygon: Polygon / MultiPolygon object
    :param meters: meters for the polygon to be inflated
    :param resolution: resolution value determines the number of segments used to approximate a quarter circle around a point.
    :return:
    """

    avg_lat, polygon = polygon_to_meters(polygon)
    polygon = polygon.buffer(meters, resolution=resolution)

    _, polygon = polygon_to_wgs84(polygon, avg_lat)

    return polygon


def merge_adjacent_polygons(geo_df: gpd.GeoDataFrame, inflation_meter: Union[float, int] = 1000,
                            aggfunc: Union[str, dict] = 'mean') -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

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


def calc_entropy(feature: pd.Series) -> float:

    """
    Takes categorical feature values and returns the column's entropy.
    The maximum value of entropy is log𝑘, where 𝑘 is the number of categories you are using.
    """
    vc = pd.Series(feature).value_counts(normalize=True, sort=False)

    return -(vc * np.log(vc) / np.log(math.e)).sum()


def create_google_maps_link_to_centroid(centroid: Point) -> str:

    """
    Get google maps link with the location
    """
    centroid_lat, centroid_lng = centroid.y, centroid.x

    return f'https://maps.google.com/?ll={centroid_lat},{centroid_lng}'


def optimize_polygon_by_probs(points: np.array, probs: np.array, polygon_type: str = 'alpha_shape', alpha: int = 4,
                              s: int = 1,  n_polygons: int = 20) -> Tuple[Union[Polygon, MultiPolygon],
                                                                          Union[Polygon, MultiPolygon],
                                                                          Union[float, None], int, list]:

    """
    this function will optimize the polygons shapes by filtering out points with low probabilities.
    it will create multiple polygons (n_polygons) for each probability threshold, calculate area, and save the value.
    then it will look for an elbow point (using KneeLocator) to the probability threshold

    :param points: numpy array of lng, lat
    :param probs: hdbscan probabilities
    :param polygon_type: 'alpha_shape'/ 'convex_hull'
    :param alpha: alpha value for alpha_shape
    :param s: sensitivity value for KneeLocator
    :param n_polygons: n polygon will be created, and the optimal one will be chosen
    :return: optimal polygon, non-optimized polygon, elbow point, # of point removed, polygon optimization curve
    """

    prob_threshold = np.linspace(min(probs), 1, n_polygons)  # initiate array with probabilities thresholds

    metrics = []

    for threshold in prob_threshold:
        threshold_mask = probs >= threshold  # create mask for filter by probability threshold
        relevant_points = points[threshold_mask]  # filter by probability threshold
        if len(relevant_points) > 0:

            poly = polygon_from_points(relevant_points, polygon_type, alpha) # create polygon out of the filtered points

            area_size = calc_polygon_area_sq_unit(poly)  # calculate the area of the polygon
            metrics.append(area_size)  # keep polygon area in "metrics" list

    # search for elbow/knee in the metrics list
    kneedle = KneeLocator(prob_threshold, metrics, S=s, curve="convex", direction="decreasing")

    original_polygon = polygon_from_points(points, polygon_type, alpha)  # create polygon with no optimizations

    if kneedle.knee:
        # create optimized polygon using kneedle.knee as threshold
        final_points = points[probs >= kneedle.knee]
        points_removed = len(points) - len(final_points)  # calculate number of points removed
        poly = polygon_from_points(final_points, polygon_type, alpha)

    else:
        poly = original_polygon
        points_removed = 0

    return poly, original_polygon, kneedle.knee, points_removed, metrics


def is_in_river(locations: pd.DataFrame, main_land: MultiPolygon) -> list:
    """
    gets dataframe of lat lon locations and extracts for each point if in river (boolean).
    return a list of booleans.
    used for later filtering, i.e. to remove river moorings from ports clustering.
    NOTE: this is a highly time consuming step
    """
    # TODO: optimize to run faster with parallelization
    locs_gpd = gpd.GeoDataFrame(locations, geometry=gpd.points_from_xy(locations.lon, locations.lat, crs="EPSG:4326"))
    return locs_gpd.loc[:, 'geometry'].progress_apply(lambda x: main_land.contains(x)).to_list()

