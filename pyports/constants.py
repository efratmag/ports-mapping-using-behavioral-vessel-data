from enum import Enum


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


class BLIP(Enum):
    # todo - do we want to use blip as enum?
    """constant parameter for first/last blip lat/lng"""
    FirstBlipLat = ''
    FirstBlipLng = ''
    LastBlipLat = ''
    LastBlipLng = ''
