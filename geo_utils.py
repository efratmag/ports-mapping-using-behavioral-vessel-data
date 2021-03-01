import math

R_KM = 6378.1  # Radius of the Earth
brng_n_e = 1.0472  # 60 degrees converted to radians.
brng_s_w = 4.18879  # 240 degrees converted to radians.
AVG_EARTH_RADIUS_M = 6371000.0

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

    dest_lat = math.asin(math.sin(lat) * math.cos(d / R_KM) +
                         math.cos(lat) * math.sin(d / R_KM) * math.cos(brng))

    dest_lng = lng + math.atan2(math.sin(brng) * math.sin(d / R_KM) * math.cos(lat),
                                math.cos(d / R_KM) - math.sin(lat) * math.sin(dest_lat))

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


def geo_dist(lat1, lng1, lat2, lng2):
    """
    Return distance in meters between the two points calculated with haversine formula
    """
    # convert decimal degrees to radians
    lat_1 = math.radians(lat1)
    lng_1 = math.radians(lng1)
    lat_2 = math.radians(lat2)
    lng_2 = math.radians(lng2)

    # haversine formula
    dlon = lng_2 - lng_1
    dlat = lat_2 - lat_1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat_1) * math.cos(lat_2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    dist_mr = AVG_EARTH_RADIUS_M * c

    return dist_mr
