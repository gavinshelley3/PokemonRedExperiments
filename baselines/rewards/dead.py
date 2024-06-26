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
from rewards.utils import read_hp_fraction


def get_dead_reward(env):
    dead_reward = 0
    for addr in PARTY_POKEMON_HP:
        if env.read_hp(addr) == 0:
            dead_reward -= 1
    return dead_reward
