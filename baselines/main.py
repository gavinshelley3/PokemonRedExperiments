import logging
import sys
import tensorflow as tf
from pathlib import Path
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from math import floor
from einops import rearrange
from rewards.badges import get_badges
from rewards.events import get_all_events_reward
from rewards.state import update_max_event_rew, update_max_op_level
from rewards.items import (
    get_item_collection_reward,
    update_item_collection_reward,
    get_total_items,
    get_unique_items,
)
from rewards.state import get_game_state_reward
from rewards.utils import read_hp_fraction, save_screenshot
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_FILE1 = "session/session_4da05e87_main_good/poke_439746560_steps"
MODEL_FILE2 = "baselines/checkpoints/ppo_model_512_steps"
HEADLESS = False


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


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def find_next_session_path(base_path="session/session"):
    i = 1
    while True:
        sess_path = Path(f"{base_path}_{i}")
        if not sess_path.exists():
            sess_path.mkdir(parents=True, exist_ok=True)
            return sess_path
        i += 1


def load_model(env):
    try:
        print("\nloading checkpoint")
        model = PPO.load(
            MODEL_FILE2, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
        )
        print("checkpoint loaded")
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh.")
        model = PPO("CnnPolicy", env, verbose=1)
    return model


def run_experiment(update_interval, env_config):
    sess_path = find_next_session_path()
    env_config["session_path"] = sess_path

    num_cpu = 1
    env = make_env(0, env_config)()
    model = load_model(env)

    log_dir = sess_path / "logs"
    writer = tf.summary.create_file_writer(str(log_dir))

    obs_buffer, action_buffer, reward_buffer, next_obs_buffer, done_buffer = (
        [],
        [],
        [],
        [],
        [],
    )
    obs, info = env.reset()
    step = 0
    total_rewards = 0

    while True:
        action, _states = model.predict(obs, deterministic=False)

        if action >= len(env.valid_actions):
            logging.warning(f"Invalid action: {action}. Defaulting to no-op action.")
            action = len(env.valid_actions) - 1

        try:
            next_obs, rewards, terminated, truncated, info = env.step(action)
        except IndexError:
            logging.error(f"Error: Action {action} is out of bounds.")
            break

        env.render()

        with writer.as_default():
            tf.summary.scalar("reward", rewards, step=step)
            for key, value in info.items():
                tf.summary.scalar(key, value, step=step)

        obs_buffer.append(obs)
        action_buffer.append(action)
        reward_buffer.append(rewards)
        next_obs_buffer.append(next_obs)
        done_buffer.append(terminated or truncated)

        if len(obs_buffer) >= update_interval:
            logging.info(f"Updating model at step {step}")
            model.learn(total_timesteps=update_interval, log_interval=1)
            obs_buffer, action_buffer, reward_buffer, next_obs_buffer, done_buffer = (
                [],
                [],
                [],
                [],
                [],
            )

        obs = next_obs
        total_rewards += rewards
        step += 1

        rewards_dict = {
            "step": step,
            "total_rewards": total_rewards,
            "event": info.get("event", 0),
            "level": info.get("level", 0),
            "heal": info.get("heal", 0),
            "op_lvl": info.get("op_lvl", 0),
            "dead": info.get("dead", 0),
            "badge": info.get("badge", 0),
            "explore": info.get("explore", 0),
            "item": info.get("item", 0),
            "pokemon_caught": info.get("pokemon_caught", 0),
            "money": info.get("money", 0),
            "enemy_defeated": info.get("enemy_defeated", 0),
            "type_effectiveness": info.get("type_effectiveness", 0),
        }

        print_rewards(rewards_dict)

        if terminated or truncated:
            break

    env.close()
    return total_rewards


def print_rewards(rewards):
    reward_str = " ".join([f"{key}: {value:.2f}" for key, value in rewards.items()])
    sys.stdout.write(f"\r{reward_str}")
    sys.stdout.flush()


def find_best_update_interval(env_config):
    intervals = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    results = {}

    for interval in intervals:
        logging.info(f"Running experiment with update interval: {interval}")
        total_rewards = run_experiment(interval, env_config)
        results[interval] = total_rewards
        logging.info(f"Total rewards for update interval {interval}: {total_rewards}")

    best_interval = max(results, key=results.get)
    logging.info(
        f"Best update interval: {best_interval} with total rewards: {results[best_interval]}"
    )
    return best_interval


if __name__ == "__main__":
    env_config = {
        "headless": HEADLESS,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../has_pokedex_nballs.state",
        "max_steps": 2**23,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "reward_scale": 4,
        "extra_buttons": True,
    }

    best_update_interval = find_best_update_interval(env_config)
    print(f"The best update interval is {best_update_interval}")
