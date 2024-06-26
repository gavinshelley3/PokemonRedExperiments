from constants.event_constants import *
from constants.map_locations import *
from constants.map_constants import *
from constants.pokedex_constants import *
from constants.pokemon_constants import *
from constants.type_constants import *
from constants.move_constants import *
from constants.battle_constants import *
from constants.player_constants import *
from constants.item_constants import *
from constants.opponent_trainer_constants import *
from constants.type_effectiveness_matrix import *
from constants.event_constants import (
    EVENT_000_FOLLOWED_OAK_INTO_LAB,
    EVENT_9FF,
)
from rewards.utils import bit_count


def get_all_events_reward(env):
    event_flags_start_addr = EVENT_000_FOLLOWED_OAK_INTO_LAB[0]  # Start address
    event_flags_end_addr = EVENT_9FF[0]  # End address
    base_event_flags = 13  # Base event flags to subtract
    total_bits_set = 0

    # Iterate over each address in the range
    for address in range(event_flags_start_addr, event_flags_end_addr + 1):
        # Read the value at the current address
        value = env.read_m(address)
        # Count the number of bits set in the current value
        total_bits_set += bit_count(value)

    # Subtract the base event flags and ensure non-negative result
    total_events_completed = max(total_bits_set - base_event_flags, 0)

    return total_events_completed
