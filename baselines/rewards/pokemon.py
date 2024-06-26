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
from constants.pokedex_constants import pokedex_own_addresses


def get_total_caught_pokemon(env):
    total_caught = 0
    num_entries = (
        len(pokedex_own_addresses) * 8
    )  # Calculate the total number of bits available
    for i in range(
        0, min(152, num_entries)
    ):  # Ensure we do not exceed the available entries
        byte_index = i // 8
        bit_index = i % 8
        address = list(pokedex_own_addresses.values())[byte_index]
        if env.read_bit(address, bit_index):
            total_caught += 1
    return total_caught


def get_unique_pokemon_caught_reward(env):
    total_unique = get_total_unique_pokemon_caught(env)
    return total_unique * 4


def get_total_unique_pokemon_caught(env):
    unique_pokemon = set()
    num_entries = (
        len(pokedex_own_addresses) * 8
    )  # Calculate the total number of bits available
    for i in range(
        0, min(152, num_entries)
    ):  # Ensure we do not exceed the available entries
        byte_index = i // 8
        bit_index = i % 8
        address = list(pokedex_own_addresses.values())[byte_index]
        if env.read_bit(address, bit_index):
            unique_pokemon.add(i)
    return len(unique_pokemon)


def get_unique_caught_pokemon(env):
    unique_caught = set()
    for idx in range(1, len(pokemon_constants) + 1):
        byte_index = (idx - 1) // 8
        bit_index = (idx - 1) % 8
        address = list(pokedex_own_addresses.values())[byte_index]
        if env.read_m(address) & (1 << bit_index):
            unique_caught.add(idx)
    return len(unique_caught)


def get_pokemon_caught_reward(env):
    total_caught = get_total_caught_pokemon(env)
    return total_caught
