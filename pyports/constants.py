from enum import Enum
import math


class ACTIVITY(Enum):
    """constant parameter for activity type"""
    MOORING = 'mooring'
    ANCHORING = 'anchoring'


class VesselType(Enum):
    """constant parameter for vessels type"""
    CARGO_CONTAINER = "cargo_container"
    CARGO_OTHER = "cargo_other"
    TANKER = "tanker"
    OTHER = "other"


class AreaType(Enum):
    """constant parameter for type of area mapped"""
    PORTS_WAITING_AREA = 'pwa'
    PORTS = 'ports'


R = 6378.1  # Radius of the Earth

METERS_IN_DEG = 2 * math.pi * 6371000.0 / 360
UNIT_RESOLVER = {'sqmi': 1609.34, 'sqkm': 1000.0}
AREA_TYPE_RESOLVER = {AreaType.PORTS_WAITING_AREA.value: 'PortWaitingArea', AreaType.PORTS.value: 'Port'}
PI_180 = math.pi / 180
PI_90 = math.pi / 90

# utm stuff
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

