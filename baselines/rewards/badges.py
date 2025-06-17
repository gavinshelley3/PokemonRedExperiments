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


def get_badges(env):
    badges = 0
    for badge in BADGES:
        if env.read_bit(badge[0], badge[1]):
            badges += 1
    return badges


def get_badge_reward(env):
    badges = get_badges(env)
    return badges * 10
