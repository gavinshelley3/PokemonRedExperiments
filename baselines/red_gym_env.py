import sys
import uuid
import logging
import json
from pathlib import Path

import numpy as np
from math import floor
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy

import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from baselines.constants.opponent_trainer_constants import (
    ENEMY_PARTY_POKEMON_HP,
    ENEMY_PARTY_POKEMON_LEVEL,
    TOTAL_ENEMY_POKEMON,
)
from baselines.rewards.events import get_all_events_reward
from constants.player_constants import (
    NUM_POKEMON_IN_PARTY_ADDRESS,
    PARTY_POKEMON_ACTUAL_LEVEL,
    PARTY_POKEMON_ADDRESSES,
    PARTY_POKEMON_HP,
    PARTY_POKEMON_MAX_HP,
)
from rewards.state import (
    get_game_state_reward,
    group_rewards,
    update_reward,
)
from rewards.badges import get_badges
from rewards.health import update_heal_reward
from rewards.items import (
    get_total_items,
    get_unique_items,
    update_item_collection_reward,
)
from rewards.money import get_money
from rewards.opponents import get_total_enemy_pokemon, initialize_enemy_hp
from rewards.state import get_game_state_reward

from rewards.map import get_map_location
from constants.map_constants import *


class RedGymEnv(Env):
    def __init__(self, config=None):
        # Initialization
        gb_path = config["gb_path"]
        self.gb_path = config.get("gb_path", "../PokemonRed.gb")
        # Initialize PyBoy emulator
        self.pyboy = PyBoy(
            gb_path, window_type="SDL2" if config["headless"] == False else "headless"
        )
        self.debug = config["debug"]
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.vec_dim = 4320
        self.headless = config["headless"]
        self.num_elements = 20000
        self.init_state = config["init_state"]
        self.action_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.early_stop = config["early_stop"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.explore_weight = config.get("explore_weight", 1)
        self.similar_frame_dist = config["sim_frame_dist"]
        self.reward_scale = config.get("reward_scale", 4)
        self.extra_buttons = config.get("extra_buttons", True)

        # Initialize rewards-related attributes
        self.previous_money = 0
        self.item_collection_reward = 0
        self.previous_item_count = get_total_items(self)
        self.previous_unique_items = get_unique_items(self)

        # Initialize health-related attributes
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0

        # Initialize enemy-related attributes
        self.total_enemy_pokemon = self.get_total_enemy_pokemon()
        self.enemy_hp = [0] * self.total_enemy_pokemon
        self.enemy_levels = [0] * self.total_enemy_pokemon

        # Initialize recent frames and memory padding
        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks
            + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2],
        )
        self.recent_memory = np.zeros(
            (self.output_shape[1] * self.memory_height, 3), dtype=np.uint8
        )
        self.frame_stacks = 3
        self.col_steps = 5
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.recent_frames = np.zeros(
            (self.frame_stacks, *self.output_shape), dtype=np.uint8
        )
        self.mem_padding = 10
        self.output_full = (
            self.output_shape[0] * self.frame_stacks
            + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2],
        )
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])
        self.reset_count = 0

        # Define valid actions
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([WindowEvent.PASS, WindowEvent.PASS])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        # Load the initial state
        logging.info(f"Loading initial state from {self.init_state}")
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        # Game state
        self.current_state = self.pyboy.get_memory_value(
            0xFF26
        )  # Example initial state address
        self.current_reward = 0
        self.total_reward = 0
        self.progress_reward = {
            "level": 0,
            "explore": 0,
            "event": 0,
            "heal": 0,
            "op_lvl": 0,
            "dead": 0,
            "badge": 0,
            "item": 0,
            "pokemon_caught": 0,
            "money": 0,
            "enemy_defeated": 0,
            "type_effectiveness": 0,
            "powerful_move": 0,
        }

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 128, 40), dtype=np.uint8
        )

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        self.init_map_mem()

        self.recent_memory = np.zeros(
            (self.output_shape[1] * self.memory_height, 3), dtype=np.uint8
        )
        self.current_reward = 0
        self.total_reward = 0
        self.progress_reward = {
            "level": 0,
            "explore": 0,
            "event": 0,
            "heal": 0,
            "op_lvl": 0,
            "dead": 0,
            "badge": 0,
            "item": 0,
            "pokemon_caught": 0,
            "money": 0,
            "enemy_defeated": 0,
            "type_effectiveness": 0,
            "powerful_move": 0,
        }
        self.last_health = self.read_hp_fraction()  # Initialize last_health on reset
        self.previous_money = 0  # Initialize previous_money on reset
        self.total_enemy_pokemon = (
            self.get_total_enemy_pokemon()
        )  # Initialize total_enemy_pokemon on reset
        self.enemy_hp = [self.read_hp(addr) for addr in ENEMY_PARTY_POKEMON_HP]
        self.enemy_levels = [self.read_m(addr) for addr in ENEMY_PARTY_POKEMON_LEVEL]

        if self.save_video:
            base_dir = self.s_path / Path("rollouts")
            base_dir.mkdir(exist_ok=True)
            full_name = Path(
                f"full_reset_{self.reset_count}_id{self.instance_id}"
            ).with_suffix(".mp4")
            model_name = Path(
                f"model_reset_{self.reset_count}_id{self.instance_id}"
            ).with_suffix(".mp4")
            self.full_frame_writer = media.VideoWriter(
                base_dir / full_name, (144, 160), fps=60
            )
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(
                base_dir / model_name, self.output_full[:2], fps=60
            )
            self.model_frame_writer.__enter__()

        return np.zeros((3, 128, 40), dtype=np.uint8), {}

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        # Execute one time step within the environment
        logging.debug(f"Performing action: {self.valid_actions[action]}")
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick()
        for release_action in self.release_arrow + self.release_button:
            self.pyboy.send_input(release_action)

        next_state = np.zeros(
            (3, 128, 40), dtype=np.uint8
        )  # Update with actual observation
        reward, reward_components = update_reward(self)
        terminated = False  # Update with actual termination condition
        truncated = False  # Update with actual truncation condition
        info = {"reward_components": reward_components}

        self.last_health = self.read_hp_fraction()  # Update last_health after step

        return next_state, reward, terminated, truncated, info

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        # Get the screen image from PyBoy
        logging.debug("Rendering the current state of the emulator")
        game_pixels_render = self.pyboy.botsupport_manager().screen().screen_ndarray()

        def create_recent_memory():
            return rearrange(
                self.recent_memory, "(h w) c -> h w c", h=self.memory_height
            )

        def create_exploration_memory():
            w = self.output_shape[1]
            h = self.memory_height

            def make_reward_channel(r_val):
                col_steps = self.col_steps
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

            level, hp, explore = group_rewards(self)
            full_memory = np.stack(
                (
                    make_reward_channel(level),
                    make_reward_channel(hp),
                    make_reward_channel(explore),
                ),
                axis=-1,
            )

            if get_badges(self) > 0:
                full_memory[:, -1, :] = 255

            return full_memory

        if reduce_res:
            # Resize the game screen to the output shape
            game_pixels_render = (
                255
                * resize(game_pixels_render, self.output_shape[:2], anti_aliasing=True)
            ).astype(np.uint8)

            if update_mem:
                # Update the recent frames
                self.recent_frames[0] = game_pixels_render

            if add_memory:
                # Create padding for the memory display
                pad = np.zeros(
                    (self.mem_padding, self.output_shape[1], self.output_shape[2]),
                    dtype=np.uint8,
                )
                # Concatenate the game screen with the exploration and recent memories
                game_pixels_render = np.concatenate(
                    (
                        create_exploration_memory(),
                        pad,
                        create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, "f h w c -> (f h) w c"),
                    ),
                    axis=0,
                )

        return game_pixels_render

    def close(self):
        self.pyboy.stop()

    def read_hp(self, addr_tuple):
        if isinstance(addr_tuple, tuple):
            start = addr_tuple[0]
            end = addr_tuple[1]
            return 256 * self.read_m(start) + self.read_m(end)
        else:
            raise TypeError(f"Address {addr_tuple} is not a valid tuple")

    def read_m(self, addr):
        if isinstance(addr, int):
            return self.pyboy.get_memory_value(addr)
        elif isinstance(addr, tuple):
            return 256 * self.read_m(addr[0]) + self.read_m(addr[1])
        else:
            raise TypeError(f"Address {addr} is not an integer or tuple")

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_hp_fraction(self):
        hp = 0
        max_hp = 0
        for i in range(6):
            hp += (
                self.read_m(PARTY_POKEMON_HP[i][0])
                + self.read_m(PARTY_POKEMON_HP[i][1]) * 256
            )
            max_hp += (
                self.read_m(PARTY_POKEMON_MAX_HP[i][0])
                + self.read_m(PARTY_POKEMON_MAX_HP[i][1]) * 256
            )
        return hp / max_hp if max_hp > 0 else 1  # Avoid division by zero

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False),
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg"
                    ),
                    obs_memory,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False),
                )

            if self.save_video and done:
                self.full_frame_writer.close()
                self.model_frame_writer.close()

            if done:
                self.all_runs.append(self.progress_reward)
                with open(
                    self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
                ) as f:
                    json.dump(self.all_runs, f)
                pd.DataFrame(self.agent_stats).to_csv(
                    self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                    compression="gzip",
                    mode="a",
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path("screenshots")
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir
            / Path(
                f"frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg"
            ),
            self.render(reduce_res=False),
        )

    def calculate_rewards(self):
        # Calculate rewards for the current state
        rewards = get_game_state_reward(self)
        logging.info(f"Calculated rewards: {rewards}")
        return rewards

    def update_rewards(self):
        old_prog = group_rewards(self)
        self.progress_reward = self.calculate_rewards()
        new_prog = group_rewards(self)
        new_total = sum(val for val in self.progress_reward.values())
        new_step = new_total - self.total_reward

        if new_step < 0 and self.read_hp_fraction() > 0:
            self.save_screenshot("neg_reward")

        self.total_reward = new_total
        return (
            new_step,
            (
                new_prog[0] - old_prog[0],
                new_prog[1] - old_prog[1],
                new_prog[2] - old_prog[2],
            ),
        )

    def update_max_op_level(self):
        opponent_level = max(self.read_m(a) for a in ENEMY_PARTY_POKEMON_LEVEL) - 5
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2

    def update_max_event_rew(self):
        cur_rew = get_all_events_reward(self)
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_total_enemy_pokemon(self):
        return self.read_m(TOTAL_ENEMY_POKEMON)


def bit_count(value):
    return bin(value).count("1")
