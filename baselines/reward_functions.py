from math import floor
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

# Define all reward-related functions


def get_badges(env):
    return bit_count(env.read_m(BADGES))


def get_level_reward(env):
    explore_thresh = 22
    scale_factor = 4
    level_sum = env.get_levels_sum()
    if level_sum < explore_thresh:
        scaled = level_sum
    else:
        scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
    env.max_level_rew = max(env.max_level_rew, scaled)
    return env.max_level_rew


def get_knn_reward(env):
    pre_rew = env.explore_weight * 0.005
    post_rew = env.explore_weight * 0.01
    cur_size = (
        env.knn_index.get_current_count()
        if env.use_screen_explore
        else len(env.seen_coords)
    )
    base = (env.base_explore if env.levels_satisfied else cur_size) * pre_rew
    post = (cur_size if env.levels_satisfied else 0) * post_rew
    return base + post


def get_all_events_reward(env):
    event_flags_start = EVENT_000_FOLLOWED_OAK_INTO_LAB[0]
    event_flags_end = EVENT_9FF[0]
    base_event_flags = 13
    return max(
        sum(bit_count(env.read_m(i)) for i in range(event_flags_start, event_flags_end))
        - base_event_flags
        - 0,
    )


def update_heal_reward(env):
    cur_health = read_hp_fraction(env)
    if (
        cur_health > env.last_health
        and env.read_m(NUM_POKEMON_IN_PARTY_ADDRESS) == env.party_size
    ):
        if env.last_health > 0:
            heal_amount = cur_health - env.last_health
            if heal_amount > 0.5:
                print(f"healed: {heal_amount}")
                save_screenshot(env, "healing")
            env.total_healing_rew += heal_amount * 4
        else:
            env.died_count += 1


def get_died_reward(env):
    return -0.1 * env.died_count


def get_heal_reward(env):
    cur_health = read_hp_fraction(env)
    if cur_health > env.last_health:
        heal_amount = cur_health - env.last_health
        if heal_amount > 0.5:
            print(f"healed: {heal_amount}")
            save_screenshot(env, "healing")
        return heal_amount * 4
    return 0


def get_map_location(env, map_idx):
    return map_locations.get(map_idx, "Unknown Location")


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


def get_money(env):
    money_bytes = [env.read_m(addr) for addr in MONEY]
    money = money_bytes[0] + (money_bytes[1] << 8) + (money_bytes[2] << 16)
    return money


def get_money_reward(env):
    current_money = get_money(env)
    money_reward = (current_money - env.previous_money) / 1000  # Scale the reward
    env.previous_money = current_money
    return money_reward


# Opponent trainer's Pokémon rewards


def get_total_enemy_pokemon(env):
    total_enemy_pokemon = env.read_m(TOTAL_ENEMY_POKEMON)
    return total_enemy_pokemon


def get_enemy_pokemon_data(env, pokemon_index):
    base_address = env.read_m(ENEMY_PARTY_POKEMON[pokemon_index])
    return base_address


def get_enemy_pokemon_hp(env, pokemon_index):
    hp = env.read_m(ENEMY_PARTY_POKEMON_HP[pokemon_index])
    return hp


def get_enemy_pokemon_level(env, pokemon_index):
    level = env.read_m(ENEMY_PARTY_POKEMON_LEVEL[pokemon_index])
    return level


def get_enemy_pokemon_defeated_reward(env):
    total_defeated = 0
    for i in range(get_total_enemy_pokemon(env)):
        if get_enemy_pokemon_hp(env, i) == 0:
            total_defeated += 1
    reward_for_round = total_defeated * 2  # Reward for defeating enemy Pokémon
    env.total_enemy_defeated_reward += reward_for_round
    return env.total_enemy_defeated_reward


def get_status_effect_reward(env):
    status_effects = [
        ENEMY_PARALYZED_BIT,
        ENEMY_FROZEN_BIT,
        ENEMY_BURNED_BIT,
        ENEMY_POISONED_BIT,
    ]
    reward = 0
    for effect in status_effects:
        if env.read_bit(ENEMY_STATUS, effect):
            reward += 1  # Adjust the multiplier as needed
    sleep_counter = env.read_m(ENEMY_STATUS) & 0x07  # Bits 0-2
    if sleep_counter > 0:
        reward += 1  # Adjust the multiplier as needed
    return reward


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


def get_move_effectiveness_reward(env):
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
    return ENEMY_PARTY_POKEMON_MAX_HP[pokemon_index]


# Add the new reward functions to the state scores
def get_game_state_reward(env, print_stats=False):
    state_scores = {
        "event": env.reward_scale * update_max_event_rew(env),
        "level": env.reward_scale * get_level_reward(env),
        "heal": env.reward_scale * get_heal_reward(env),
        "op_lvl": env.reward_scale * update_max_op_level(env),
        "dead": env.reward_scale * get_died_reward(env),
        "badge": env.reward_scale * get_badges(env) * 5,
        "explore": env.reward_scale * get_knn_reward(env),
        "item": env.reward_scale * get_item_collection_reward(env),
        "pokemon_caught": env.reward_scale * get_pokemon_caught_reward(env),
        "money": env.reward_scale * get_money_reward(env),
        "enemy_defeated": env.reward_scale * get_enemy_pokemon_defeated_reward(env),
        "type_effectiveness": env.reward_scale * get_move_effectiveness_reward(env),
        "status_effect": env.reward_scale * get_status_effect_reward(env),
        "powerful_move": env.reward_scale * get_powerful_move_reward(env),
    }
    return state_scores


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


def read_hp_fraction(env):
    hp_sum = sum(env.read_hp(add) for add in PARTY_POKEMON_HP)
    max_hp_sum = sum(env.read_hp(add) for add in PARTY_POKEMON_MAX_HP)
    max_hp_sum = max(max_hp_sum, 1)
    return hp_sum / max_hp_sum


def bit_count(bits):
    return bin(bits).count("1")


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


def get_total_items(env):
    total_items = env.read_m(TOTAL_ITEMS)
    return total_items


def get_unique_items(env):
    # Read every other byte to see if the item is in the bag
    unique_items = set()
    for i in range(0, 256, 2):
        item = env.read_m(ITEMS + i)
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

    if new_unique_items > 0:
        env.item_collection_reward += (
            new_unique_items * 2
        )  # Higher reward for unique items

    env.previous_item_count = current_item_count
    env.previous_unique_items = current_unique_items


def get_item_collection_reward(env):
    return env.item_collection_reward


def get_total_pokemon_caught(env):
    total_pokemon = 0
    for addr in range(POKEDEX_OWNED_START, POKEDEX_OWNED_END + 1):
        owned_flags = env.read_m(addr)
        total_pokemon += bit_count(owned_flags)

    return total_pokemon


def get_unique_pokemon_caught(env):
    unique_pokemon = set()
    for addr in range(POKEDEX_OWNED_START, POKEDEX_OWNED_END + 1):
        owned_flags = env.read_m(addr)
        for i in range(8):
            if owned_flags & (1 << i):
                unique_pokemon.add((addr - POKEDEX_OWNED_START) * 8 + i)

    return len(unique_pokemon)


def update_pokemon_catch_reward(env):
    current_pokemon_count = get_total_pokemon_caught(env)
    current_unique_pokemon = get_unique_pokemon_caught(env)

    new_pokemon = current_pokemon_count - env.previous_pokemon_count
    new_unique_pokemon = current_unique_pokemon - env.previous_unique_pokemon

    if new_pokemon > 0:
        env.pokemon_catch_reward += new_pokemon

    if new_unique_pokemon > 0:
        env.pokemon_catch_reward += new_unique_pokemon * 4

    env.previous_pokemon_count = current_pokemon_count
    env.previous_unique_pokemon = current_unique_pokemon


def get_pokemon_catch_reward(env):
    return env.pokemon_catch_reward


def get_total_pokemon_seen(env):
    total_pokemon = 0
    for addr in range(POKEDEX_SEEN_START, POKEDEX_SEEN_END + 1):
        seen_flags = env.read_m(addr)
        total_pokemon += bit_count(seen_flags)

    return total_pokemon


def get_unique_pokemon_seen(env):
    unique_pokemon = set()
    for addr in range(POKEDEX_SEEN_START, POKEDEX_SEEN_END + 1):
        seen_flags = env.read_m(addr)
        for i in range(8):
            if seen_flags & (1 << i):
                unique_pokemon.add((addr - POKEDEX_SEEN_START) * 8 + i)

    return len(unique_pokemon)


def update_pokemon_seen_reward(env):
    current_pokemon_count = get_total_pokemon_seen(env)
    current_unique_pokemon = get_unique_pokemon_seen(env)

    new_pokemon = current_pokemon_count - env.previous_pokemon_seen_count
    new_unique_pokemon = current_unique_pokemon - env.previous_unique_pokemon_seen

    if new_pokemon > 0:
        env.pokemon_seen_reward += new_pokemon

    if new_unique_pokemon > 0:
        env.pokemon_seen_reward += new_unique_pokemon * 1

    env.previous_pokemon_seen_count = current_pokemon_count
    env.previous_unique_pokemon_seen = current_unique_pokemon


def get_pokemon_seen_reward(env):
    return env.pokemon_seen_reward


def save_screenshot(env, name):
    ss_dir = env.s_path / Path("screenshots")
    ss_dir.mkdir(exist_ok=True)
    plt.imsave(
        ss_dir
        / Path(
            f"frame{env.instance_id}_r{env.total_reward:.4f}_{env.reset_count}_{name}.jpeg"
        ),
        env.render(reduce_res=False),
    )
