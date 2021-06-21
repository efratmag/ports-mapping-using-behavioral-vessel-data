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
