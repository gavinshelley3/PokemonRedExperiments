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
