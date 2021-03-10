import math
import json
from shapely.geometry import shape, Point, MultiLineString
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from shapely.ops import cascaded_union, polygonize


R = 6378.1  # Radius of the Earth
brng_n_e = 1.0472  # 60 degrees converted to radians.
brng_s_w = 4.18879  # 240 degrees converted to radians.


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


def get_bounding_box(lat, lng, d=15):

    """
    Calculate the bounding box for given a point and distance
    :param lat: latitude
    :param lng: longitude
    :param d: distance in Km
    :return:
    """

    lat_n_e, lng_n_e = calc_dest_point(lat, lng, brng_n_e, d=d)
    lat_s_w, lng_s_w = calc_dest_point(lat, lng, brng_s_w, d=d)

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
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points, edges

