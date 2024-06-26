import logging
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


def get_total_items(env):
    total_items = env.read_m(TOTAL_ITEMS)
    return total_items


def get_unique_items(env):
    unique_items = set()
    for address in ITEMS:
        item = env.read_m(address)
        if item != 0:
            unique_items.add(item)
    return len(unique_items)


def update_item_collection_reward(env):
    current_item_count = get_total_items(env)
    current_unique_items = get_unique_items(env)

    new_items = current_item_count - env.previous_item_count
    new_unique_items = current_unique_items - env.previous_unique_items

    if new_items > 0:
        env.item_collection_reward += new_items * 0.5
        logging.info(
            f"New items collected: {new_items}, New item reward: {env.item_collection_reward}"
        )

    if new_unique_items > 0:
        env.item_collection_reward += (
            new_unique_items * 2
        )  # Higher reward for unique items
        logging.info(
            f"New unique items collected: {new_unique_items}, New unique item reward: {env.item_collection_reward}"
        )

    env.previous_item_count = current_item_count
    env.previous_unique_items = current_unique_items


def get_item_collection_reward(env):
    return env.item_collection_reward
