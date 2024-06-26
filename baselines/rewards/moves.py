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
from rewards.opponents import get_enemy_pokemon_hp, get_total_enemy_pokemon


def get_move_effectiveness(env):
    move_type = env.read_m(PLAYER_MOVE_TYPE)
    enemy_type1 = env.read_m(ENEMY_TYPE1)
    enemy_type2 = env.read_m(ENEMY_TYPE2)

    effectiveness1 = type_effectiveness.get((move_type, enemy_type1), 1)
    effectiveness2 = type_effectiveness.get((move_type, enemy_type2), 1)

    effectiveness = effectiveness1 * effectiveness2

    if effectiveness > 1:
        return 1  # Super effective
    elif effectiveness < 1:
        return -1  # Not very effective
    else:
        return 0  # Neutral


def get_type_effectiveness_reward(env):
    effectiveness = get_move_effectiveness(env)
    return effectiveness * 2  # Adjust reward scaling as needed


def get_powerful_move_reward(env):
    total_enemy_pokemon = get_total_enemy_pokemon(env)
    total_reward = 0

    for i in range(total_enemy_pokemon):
        enemy_hp = get_enemy_pokemon_hp(env, i)
        max_enemy_hp = get_enemy_pokemon_max_hp(env, i)

        if max_enemy_hp > 0:
            hp_fraction = (max_enemy_hp - enemy_hp) / max_enemy_hp
            total_reward += hp_fraction * 3  # Adjust the multiplier as needed

    return total_reward


def get_enemy_pokemon_max_hp(env, pokemon_index):
    max_hp_addresses = ENEMY_PARTY_POKEMON_MAX_HP[pokemon_index]
    max_hp = env.read_hp(max_hp_addresses)
    return max_hp
