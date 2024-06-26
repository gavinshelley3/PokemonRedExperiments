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
    env.enemy_hp = [env.read_hp(addr) for addr in ENEMY_PARTY_POKEMON_HP]


def get_enemy_pokemon_defeated_reward(env):
    total_defeated = 0
    for i in range(get_total_enemy_pokemon(env)):
        current_hp = get_enemy_pokemon_hp(env, i)
        if current_hp == 0 and env.enemy_hp[i] > 0:
            total_defeated += 1
        env.enemy_hp[i] = current_hp  # Update the HP state
    reward_for_round = total_defeated * 2  # Reward for defeating enemy Pokémon
    env.total_enemy_defeated_reward += reward_for_round
    return env.total_enemy_defeated_reward
