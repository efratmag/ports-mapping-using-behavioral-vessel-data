import math
import json
from shapely.geometry import shape, Point, MultiLineString
from scipy.spatial import Delaunay
import numpy as np
from shapely import ops
import geopandas as gpd


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


def polygon_to_meters(polygon):

    avg_lat = polygon.centroid.y

    def shape_to_meters(lat, lng, avg_lat):
        x = lng * math.cos(math.radians(avg_lat)) * METERS_IN_DEG
        y = lat * METERS_IN_DEG
        return x, y

    def to_meters(lng, lat):
        return shape_to_meters(lat, lng, avg_lat)

    return shape(ops.transform(to_meters, polygon))


def calc_polygon_area_sq_unit(polygon, unit='sqkm'):

    polygon = polygon_to_meters(polygon)
    polygon_area = np.sqrt(polygon.area) / UNIT_RESOLVER[unit]
    polygon_area *= polygon_area

    return polygon_area


def polygons_to_multi_lines(polygons_df):

    polygons_multi_line = ops.linemerge(polygons_df['geometry'].boundary.values)

    return polygons_multi_line


def merge_polygons(geo_df):

    merged_polygons = gpd.GeoSeries(ops.cascaded_union(geo_df['geometry'])).loc[0]

    return merged_polygons
