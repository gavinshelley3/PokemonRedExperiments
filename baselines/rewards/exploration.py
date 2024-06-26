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


def get_knn_reward(env):
    pre_rew = env.explore_weight * 0.005
    post_rew = env.explore_weight * 0.01
    cur_size = (
        env.knn_index.get_current_count()
        if env.use_screen_explore
        else len(env.seen_coords)
    )
    base = (env.base_explore if env.levels_satisfied else cur_size) * pre_rew
    post = (cur_size if env.levels_satisfied else 0) * post_rew
    return base + post


def get_explore_reward(env):
    # Use CURRENT_PLAYER_X_POSITION and CURRENT_PLAYER_Y_POSITION to get the player's current position
    # Reward the agent for exploring new areas of the map
    # Use the seen_coords attribute to keep track of the coordinates the agent has visited
    # X_POSITION = env.read_m(CURRENT_PLAYER_X_POSITION)
    # Y_POSITION = env.read_m(CURRENT_PLAYER_Y_POSITION)
    # coords = (X_POSITION, Y_POSITION)
    # if coords not in env.seen_coords:
    #     env.seen_coords.add(coords)
    #     return 1
    return 0
