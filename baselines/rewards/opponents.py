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


def get_total_enemy_pokemon(env):
    return env.read_m(TOTAL_ENEMY_POKEMON)


def get_enemy_pokemon_data(env, pokemon_index):
    base_address = env.read_m(ENEMY_PARTY_POKEMON[pokemon_index])
    return base_address


def get_enemy_pokemon_hp(env, pokemon_index):
    hp_addr = ENEMY_PARTY_POKEMON_HP[pokemon_index]
    return env.read_m(hp_addr[0]) * 256 + env.read_m(hp_addr[1])


def get_enemy_pokemon_level(env, pokemon_index):
    level = env.read_m(ENEMY_PARTY_POKEMON_LEVEL[pokemon_index])
    return level


def initialize_enemy_hp(env):
    env.enemy_hp = [env.read_hp(env, addr) for addr in ENEMY_PARTY_POKEMON_HP]


def get_enemy_pokemon_defeated_reward(env):
    total_reward = 0
    for i in range(env.total_enemy_pokemon):
        try:
            current_hp = env.read_hp(ENEMY_PARTY_POKEMON_HP[i])
            if current_hp == 0 and env.enemy_hp[i] > 0:
                total_reward += 1
            env.enemy_hp[i] = current_hp
        except IndexError:
            # If the index is out of range, continue to the next iteration
            continue
    return total_reward


def get_op_level_reward(env):
    total_reward = 0
    for i in range(env.total_enemy_pokemon):
        try:
            current_level = env.read_m(ENEMY_PARTY_POKEMON_LEVEL[i])
            if current_level > env.enemy_levels[i]:
                total_reward += 1
            env.enemy_levels[i] = current_level
        except IndexError:
            # If the index is out of range, continue to the next iteration
            continue
    return total_reward
