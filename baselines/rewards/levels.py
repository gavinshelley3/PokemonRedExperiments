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


def get_levels_reward(env):
    explore_thresh = 22
    scale_factor = 4
    level_sum = env.get_levels_sum()
    if level_sum < explore_thresh:
        scaled = level_sum
    else:
        scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
    env.max_level_rew = max(env.max_level_rew, scaled)
    return env.max_level_rew
