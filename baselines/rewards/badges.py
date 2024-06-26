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
from constants.event_constants import BADGES
from rewards.utils import bit_count


def get_badges(env):
    return bit_count(env.read_m(BADGES))


def get_badge_reward(env):
    badges = get_badges(env)
    return badges * 10
