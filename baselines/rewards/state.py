import logging
from math import floor
from einops import rearrange
import numpy as np
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
from constants.opponent_trainer_constants import ENEMY_PARTY_POKEMON_LEVEL

from rewards.badges import get_badge_reward, get_badges
from rewards.dead import get_dead_reward
from rewards.levels import get_level_reward
from rewards.exploration import get_explore_reward
from rewards.events import get_all_events_reward, get_event_reward
from rewards.health import get_heal_reward
from rewards.items import get_item_collection_reward
from rewards.money import get_money_reward
from rewards.pokemon import get_pokemon_caught_reward
from rewards.opponents import get_enemy_pokemon_defeated_reward, get_op_level_reward
from rewards.moves import (
    get_powerful_move_reward,
    get_type_effectiveness_reward,
)
from rewards.events import get_all_events_reward
from rewards.badges import get_badges


def get_game_state_reward(env):
    rewards = {
        "event": get_event_reward(env),
        "level": get_level_reward(env),
        "heal": get_heal_reward(env),
        "op_lvl": get_op_level_reward(env),
        "dead": get_dead_reward(env),
        "badge": get_badge_reward(env),
        "explore": get_explore_reward(env),
        "item": get_item_collection_reward(env),
        "pokemon_caught": get_pokemon_caught_reward(env),
        "money": get_money_reward(env),
        "enemy_defeated": get_enemy_pokemon_defeated_reward(env),
        "type_effectiveness": get_type_effectiveness_reward(env),
        "powerful_move": get_powerful_move_reward(env),
    }
    logging.info(f"Calculated rewards: {rewards}")
    return rewards


def group_rewards(env):
    prog = env.progress_reward
    # Default values for missing keys
    level = prog.get("level", 0)
    explore = prog.get("explore", 0)
    return (
        level * 100 / env.reward_scale,
        env.read_hp_fraction() * 100 / env.reward_scale,
        explore * 150 / (env.explore_weight * env.reward_scale),
    )


def update_reward(env):
    old_prog = group_rewards(env)
    env.progress_reward = get_game_state_reward(env)
    new_prog = group_rewards(env)
    new_total = sum(val for val in env.progress_reward.values())
    new_step = new_total - env.total_reward

    if new_step < 0 and env.read_hp_fraction() > 0:
        env.save_screenshot("negative_reward")

    env.total_reward = new_total
    return (
        new_step,
        (
            new_prog[0] - old_prog[0],
            new_prog[1] - old_prog[1],
            new_prog[2] - old_prog[2],
        ),
    )


def update_max_op_level(env):
    opponent_level = max(env.read_m(a) for a in ENEMY_PARTY_POKEMON_LEVEL) - 5
    env.max_opponent_level = max(env.max_opponent_level, opponent_level)
    return env.max_opponent_level * 0.2


def update_max_event_rew(env):
    cur_rew = get_all_events_reward(env)
    env.max_event_rew = max(cur_rew, env.max_event_rew)
    return env.max_event_rew
