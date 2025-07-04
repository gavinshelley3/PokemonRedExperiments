import logging
import sys
import re
import tensorflow as tf
from pathlib import Path
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from math import floor
from einops import rearrange
from rewards.badges import get_badges
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

MODEL_DIR = "checkpoints"
HEADLESS = False

# Placeholder for the model and step count
current_model = None
current_step = 0


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
    # Regular expression to extract step count from the filename
    step_pattern = re.compile(r"ppo_model_(\d+)_steps")

    # Find all model files and sort by the extracted step count
    model_files = sorted(
        Path(MODEL_DIR).glob("*.zip"),
        key=lambda x: (
            int(step_pattern.search(x.stem).group(1))
            if step_pattern.search(x.stem)
            else -1
        ),
    )

    if model_files and model_files[-1].exists():
        model_path = model_files[-1]
        print(f"\nLoading latest checkpoint from: {model_path}")
        model = PPO.load(
            model_path, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
        )
        print(f"Checkpoint loaded from file: {model_path.name}")
        # Extract the step count from the filename
        global current_step
        current_step = int(step_pattern.search(model_path.stem).group(1))
    else:
        print("No checkpoint found. Starting fresh.")
        model = PPO("CnnPolicy", env, verbose=1)
        current_step = 0
    return model


def save_model(model, step):
    model_path = Path(MODEL_DIR) / f"ppo_model_{step}_steps.zip"
    model.save(model_path)
    print(f"Model saved at {model_path}")


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

    print("Starting the experiment...")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)

            if action >= len(env.valid_actions):
                logging.warning(
                    f"Invalid action: {action}. Defaulting to no-op action."
                )
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
                    if isinstance(value, (int, float)):  # Only log scalar values
                        tf.summary.scalar(key, value, step=step)

            obs_buffer.append(obs)
            action_buffer.append(action)
            reward_buffer.append(rewards)
            next_obs_buffer.append(next_obs)
            done_buffer.append(terminated or truncated)

            if len(obs_buffer) >= update_interval:
                logging.info(f"Updating model at step {step}")
                model.learn(total_timesteps=update_interval, log_interval=1)
                (
                    obs_buffer,
                    action_buffer,
                    reward_buffer,
                    next_obs_buffer,
                    done_buffer,
                ) = (
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

            # Add logging for debugging
            logging.info(f"Step: {step}, Total Rewards: {total_rewards}")
            if terminated or truncated:
                logging.info(f"Terminated: {terminated}, Truncated: {truncated}")
                break
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Saving the model...")
        save_model(model, step)
        print("Model saved successfully. Exiting.")
    finally:
        env.close()
    return total_rewards


def print_rewards(rewards):
    reward_str = " ".join([f"{key}: {value:.2f}" for key, value in rewards.items()])
    sys.stdout.write(f"\r{reward_str}")
    sys.stdout.flush()


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

    interval = 512
    logging.info(f"Running experiment with {interval} steps.")
    total_rewards = run_experiment(interval, env_config)
    logging.info(f"Total rewards: {total_rewards}")
