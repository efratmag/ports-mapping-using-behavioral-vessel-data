import math

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
