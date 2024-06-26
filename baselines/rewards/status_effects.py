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


def get_status_effect_reward(env):
    status_effects = [
        ENEMY_PARALYZED_BIT,
        ENEMY_FROZEN_BIT,
        ENEMY_BURNED_BIT,
        ENEMY_POISONED_BIT,
    ]
    reward = 0
    for effect in status_effects:
        if env.read_bit(ENEMY_STATUS, effect):
            reward += 1  # Adjust the multiplier as needed
    sleep_counter = env.read_m(ENEMY_STATUS) & 0x07  # Bits 0-2
    if sleep_counter > 0:
        reward += 1  # Adjust the multiplier as needed
    return reward
