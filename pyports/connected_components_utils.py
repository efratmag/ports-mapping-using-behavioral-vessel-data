import numpy as np
import pandas as pd
import utm


R = 6378.1e3
THR = 10000
ZONE_EXCEPTIONS = ["31V", "32V", "31X", "33X", "35X", "37X"]
ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"
ALLOWED_BORDERS = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
SPECIAL_BORDERS = {"31V": {"NE": {(31, "W"): ["S"]},
                           "SE": {(31, "U"): ["N"]}},
                   "32V": {"NW": {(31, "W"): ["S"]},
                           "SW": {(31, "U"): ["N"]},
                           "N": {(31, "W"): ["S"], (32, "W"): ["S"]},
                           "S": {(31, "U"): ["N"], (32, "U"): ["N"]}},
                   "31U": {"N": {(31, "V"): ["S"], (32, "V"): ["S"]},
                           "NE": {(32, "V"): ["S"]}},
                   "32U": {"NW": {(32, "V"): ["S"]}},
                   "31W": {"NE": {(31, "X"): ["S"]},
                           "S": {(31, "V"): ["N"], (32, "V"): ["N"]},
                           "SE": {(32, "V"): ["N"]}},
                   "32W": {"SW": {(32, "V"): ["N"]},
                           "N": {(31, "X"): ["S"], (33, "X"): ["S"]},
                           "NE": {(33, "X"): ["S"]}},
                   "33W": {"NW": {(33, "X"): ["S"]},
                           "N": {(33, "X"): ["S"]},
                           "NE": {(33, "X"): ["S"]}},
                   "34W": {"NW": {(33, "X"): ["S"]},
                           "N": {(33, "X"): ["S"], (35, "X"): ["S"]},
                           "NE": {(35, "X"): ["S"]}},
                   "35W": {"NW": {(35, "X"): ["S"]},
                           "N": {(35, "X"): ["S"]},
                           "NE": {(35, "X"): ["S"]}},
                   "36W": {"NW": {(35, "X"): ["S"]},
                           "N": {(35, "X"): ["S"], (37, "X"): ["S"]},
                           "NE": {(37, "X"): ["S"]}},
                   "37W": {"NW": {(35, "X"): ["S"]},
                           "N": {(37, "X"): ["S"]}},
                   "31X": {"S": {(31, "W"): ["N"]},
                           "SE": {(32, "W"): ["S"]},
                           "E": {(33, "X"): ["W"]}},
                   "33X": {"S": {(32, "W"): ["N"], (33, "W"): ["N"], (34, "W"): ["N"]},
                           "SW": {(32, "W"): ["N"]},
                           "SE": {(34, "W"): ["N"]},
                           "E": {(35, "X"): ["W"]}},
                   "35X": {"S": {(34, "W"): ["N"], (35, "W"): ["N"], (36, "W"): ["N"]},
                           "SW": {(34, "W"): ["N"]},
                           "SE": {(36, "W"): ["N"]},
                           "E": {(37, "X"): ["W"]}},
                   "37X": {"S": {(36, "W"): ["N"], (37, "W"): ["N"]},
                           "SW": {(36, "W"): ["N"]}}
                   }


def get_utm(lat, lon):
    """Calculate UTM coordinates latitude and longitude."""

    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return pd.Series([easting, northing, zone_number, zone_letter],
                     index=["easting", "northing", "zone_number", "zone_letter"])


def check_zone(zone_number, zone_letter):
    """Check if zone number and zone letter are valid."""

    if zone_number < 0 or zone_number > 60: raise ValueError(f"zone number is invalid: {zone_number}")
    if zone_letter not in ZONE_LETTERS: raise ValueError(f"zone letter is invalid: {zone_letter}")
    if zone_letter == "X" and zone_number in [32, 34, 36]:
        raise ValueError(f"there are no zone {zone_number}{zone_letter}")
    return True


def validate_zone(*decargs):
    if len(decargs)!=2: raise ValueError("zone designator contains exactly two elements")

    zone_number_idx = decargs[0]
    zone_letter_idx = decargs[1]

    def decorator(f):
        def wrapper(*args, **kwargs):
            if "zone_number" not in kwargs.keys() or "zone_letter" not in kwargs.keys():
                zone_number = args[zone_number_idx]
                zone_letter = args[zone_letter_idx]
            else:
                zone_number = kwargs["zone_number"]
                zone_letter = kwargs["zone_letter"]
            check_zone(zone_number, zone_letter)
            return f(*args, **kwargs)
        return wrapper
    return decorator


@validate_zone(0, 1)
def get_neighboring_zone_generic(zone_number, zone_letter, border):
    """Get generic neighboring zone."""

    if (zone_letter == "X" and "N" in border) or (zone_letter == "C" and "S" in border):
        return {}

    neighbor_borders = []
    neighbor_number = zone_number
    neighbor_letter = zone_letter

    if "N" in border:
        neighbor_letter = ZONE_LETTERS[ZONE_LETTERS.index(zone_letter.upper())+1]
        neighbor_borders.append("S")

    if "S" in border:
        neighbor_letter = ZONE_LETTERS[ZONE_LETTERS.index(zone_letter.upper())-1]
        neighbor_borders.append("N")

    if "W" in border:
        neighbor_number = 60 if zone_number == 1 else zone_number - 1
        neighbor_borders.append("E")

    if "E" in border:
        neighbor_number = (zone_number + 1) % 60
        neighbor_borders.append("W")

    return {(neighbor_number, neighbor_letter): neighbor_borders}


@validate_zone(0, 1)
def get_neighboring_zones(zone_number, zone_letter, borders):
    """Get zones neighboring to `borders` of `zone_number, zone_letter`."""

    zone = f"{zone_number}{zone_letter}"

    borders = [border.upper() for border in borders]

    if "".join(borders) not in ALLOWED_BORDERS: raise ValueError(f"border tuple {borders} is invalid")

    neigboring_zones = {}

    all_borders = set(borders)
    all_borders.add("".join(borders))

    for border in all_borders:
        if zone in SPECIAL_BORDERS.keys() and border in SPECIAL_BORDERS[zone].keys():
            update = SPECIAL_BORDERS[zone][border]
        else:
            update = get_neighboring_zone_generic(zone_number, zone_letter, border)
        neigboring_zones.update(update)
    return neigboring_zones


@validate_zone(0, 1)
def get_zone_border(zone_number, zone_letter):
    """Convenience routine to calculate zone border along latitude and longitude."""

    zone = f"{zone_number}{zone_letter}"

    # generic path
    if zone not in ZONE_EXCEPTIONS:
        lon_min = (zone_number - 1) * 6 - 180
        lon_max = lon_min + 6
    # Special grid zones
    elif zone == "31V":
        lon_min, lon_max = 0, 3
    elif zone == "32V":
        lon_min, lon_max = 3, 12
    elif zone == "31X":
        lon_min, lon_max = 0, 9
    elif zone == "33X":
        lon_min, lon_max = 9, 21
    elif zone == "35X":
        lon_min, lon_max = 21, 33
    elif zone == "37X":
        lon_min, lon_max = 33, 42

    zone_letter_idx = ZONE_LETTERS.index(zone_letter.upper())
    lat_min = -80 + 8 * zone_letter_idx
    lat_max = lat_min + 8

    if lat_max==80: lat_max = 84

    return lon_min, lon_max, lat_min, lat_max


@validate_zone(2, 3)
def is_border(lat, lon, zone_number, zone_letter, thr):
    """Calculate border code for a location."""

    border_status = {}
    lon_min, lon_max, lat_min, lat_max = get_zone_border(zone_number, zone_letter)

    if (lat < lat_min) or (lat > lat_max) or (lon < lon_min) or (lon > lon_max):
        raise ValueError(f"location {lat}, {lon} is outside the zone {zone_number}{zone_letter}")

    lon_min_dist = R * np.cos(lat * np.pi / 180) * np.abs(lon - lon_min) * np.pi / 180
    lon_max_dist = R * np.cos(lat * np.pi / 180) * np.abs(lon - lon_max) * np.pi / 180
    lat_min_dist = R * np.abs(lat - lat_min) * np.pi / 180
    lat_max_dist = R * np.abs(lat - lat_max) * np.pi / 180

    border_status["W"] = lon_min_dist <= thr
    border_status["E"] = lon_max_dist <= thr
    border_status["S"] = lat_min_dist <= thr
    border_status["N"] = lat_max_dist <= thr

    return pd.Series(border_status).astype(int)


def get_in_zone_distances(loc, locs):
    """Calculate distances between `loc` and all location in `locs` in the same zone."""
    return np.sqrt(np.square(locs[["easting", "northing"]] - loc[["easting", "northing"]]).sum(axis=1))


def get_cross_zone_distances(loc, locs):
    """Calculate distances between `loc` and all location in `locs` in the same zone."""

    dphi_sqr = np.square(locs["lat"] - loc["lat"])
    dlambda_sqr = np.square(locs["lon"] - loc["lon"])
    return R * np.sqrt(dphi_sqr + np.cos(loc.lat * np.pi / 180) * dlambda_sqr) * np.pi / 180


def get_in_zone_neighbors_kdtree(loc, tree, thr):
    """Calculate neighbors of `loc` in the same zone."""

    tree_elements, tree = tree
    neighbors = tree.query_ball_point(loc[["easting", "northing"]], THR)

    return tree_elements[neighbors]


def get_in_zone_neighbors(loc, locs, thr):
    """Calculate neighbors of `loc` in the same zone."""
    # TODO: vectorize

    zone_mask = (locs.zone_number==loc.zone_number) & (locs.zone_letter==loc.zone_letter)
    cand_mask_x = (locs.cell_x==loc.cell_x) | (locs.cell_x==(loc.cell_x-1)) | (locs.cell_x==(loc.cell_x+1))
    cand_mask_y = (locs.cell_y==loc.cell_y) | (locs.cell_y==(loc.cell_y-1)) | (locs.cell_y==(loc.cell_y+1))
    cand_mask = cand_mask_x & cand_mask_y & zone_mask & (locs.component==-1)
    candidates = locs[cand_mask]

    dist = get_in_zone_distances(loc, candidates)
    return candidates[dist<=thr].index


def get_cross_zone_neighbors(loc, locs, thr):
    """Calculate neighbors of `loc` in the same zone."""

    updated_zones = []
    border_status = loc[["N", "S", "E", "W"]]
    border_list = border_status[border_status!=0].index.tolist()
    neigboring_zones = get_neighboring_zones(loc.zone_number, loc.zone_letter, border_list)

    neighbors = []

    for (zn, zl), borders in neigboring_zones.items():
        zone_mask = (locs.zone_number==zn) & (locs.zone_letter==zl)
        border_mask = locs[borders].sum(axis=1)!=0
        candidates = locs[zone_mask & border_mask & (locs.component==-1)]

        dist = get_cross_zone_distances(loc, candidates)
        neighbors.append(candidates[dist<=thr])

        if not candidates[dist<=thr].empty:
            updated_zones.append((zn, zl))

    return pd.concat(neighbors).index, updated_zones


class ConnectedComponent(object):
    """Connected component entity. New elements can be added, and component can check if it's finished."""
    def __init__(self, cid, all_locations, thr):
        self.members = set()
        self.visited = set()
        self.cid = cid
        self.all_locations = all_locations
        self.thr = thr

    def add(self, element):
        self.members.add(element)

    def visit(self, element):
        if element not in self.members:
            raise ValueError(f"element {element} is not in this component")
        self.visited.add(element)

    def is_full(self):
        return self.members==self.visited

    def grow(self, locs=None, kdtrees=None):
        """Grow this component by elements from `locs`."""

        subset = self.members.difference(self.visited)
        locs = locs if locs is not None else self.all_locations
        all_updated_zones = set()

        for element in subset:
            zn, zl = self.all_locations.loc[element, ["zone_number", "zone_letter"]]

            if (kdtrees is not None):
                kdtree = kdtrees[(zn, zl)]
                neighbors = get_in_zone_neighbors_kdtree(self.all_locations.loc[element],
                                                         kdtree,
                                                         self.thr)
            else:
                neighbors = get_in_zone_neighbors(self.all_locations.loc[element], locs, self.thr)

            if self.all_locations.loc[element, "border"]:
                cross_zone_neighbors, updated_zones = get_cross_zone_neighbors(self.all_locations.loc[element], locs, self.thr)
                neighbors = neighbors.union(cross_zone_neighbors)
                all_updated_zones.update(updated_zones)

            self.members.update(neighbors)
            self.visited.add(element)
        return all_updated_zones

    @property
    def elements(self):
        return list(self.members)

    @property
    def size(self):
        return len(self.members)

