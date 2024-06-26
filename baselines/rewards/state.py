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
from rewards.utils import read_hp_fraction, save_screenshot


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
    return (
        prog["level"] * 100 / env.reward_scale,
        read_hp_fraction(env) * 2000,
        prog["explore"] * 150 / (env.explore_weight * env.reward_scale),
    )


def update_reward(env):
    old_prog = group_rewards(env)
    env.progress_reward = get_game_state_reward(env)
    new_prog = group_rewards(env)
    new_total = sum(val for val in env.progress_reward.values())
    new_step = new_total - env.total_reward

    if new_step < 0 and read_hp_fraction(env) > 0:
        save_screenshot(env, "neg_reward")

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


def create_exploration_memory(env):
    w = env.output_shape[1]
    h = env.memory_height

    def make_reward_channel(r_val):
        col_steps = env.col_steps
        max_r_val = (w - 1) * h * col_steps
        r_val = min(r_val, max_r_val)
        row = floor(r_val / (h * col_steps))
        memory = np.zeros(shape=(h, w), dtype=np.uint8)
        memory[:, :row] = 255
        row_covered = row * h * col_steps
        col = floor((r_val - row_covered) / col_steps)
        memory[:col, row] = 255
        col_covered = col * col_steps
        last_pixel = floor(r_val - row_covered - col_covered)
        memory[col, row] = last_pixel * (255 // col_steps)
        return memory

    level, hp, explore = group_rewards(env)
    full_memory = np.stack(
        (
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore),
        ),
        axis=-1,
    )

    if get_badges(env) > 0:
        full_memory[:, -1, :] = 255

    return full_memory


def create_recent_memory(env):
    return rearrange(env.recent_memory, "(w h) c -> h w c", h=env.memory_height)
