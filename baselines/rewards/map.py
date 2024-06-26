# rewards/map.py

from constants.map_locations import map_locations


def get_map_location(env, map_idx):
    return map_locations.get(map_idx, "Unknown Location")
