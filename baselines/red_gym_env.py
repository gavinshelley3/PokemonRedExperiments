import uuid
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy

import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from constants.opponent_trainer_constants import ENEMY_PARTY_POKEMON_HP
from constants.player_constants import (
    NUM_POKEMON_IN_PARTY_ADDRESS,
    PARTY_POKEMON_ACTUAL_LEVEL,
    PARTY_POKEMON_ADDRESSES,
)
from rewards.state import (
    create_exploration_memory,
    create_recent_memory,
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
from rewards.utils import read_hp_fraction
from rewards.map import get_map_location
from constants.map_constants import *


class RedGymEnv(Env):
    def __init__(self, config=None):
        # Initialization
        rom_path = config["gb_path"]
        self.pyboy = PyBoy(
            rom_path, window_type="SDL2" if config["headless"] == False else "headless"
        )
        self.debug = config["debug"]
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.vec_dim = 4320
        self.headless = config["headless"]
        self.num_elements = 20000
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.early_stopping = config["early_stop"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = config.get("explore_weight", 1)
        self.use_screen_explore = config.get("use_screen_explore", True)
        self.similar_frame_dist = config["sim_frame_dist"]
        self.previous_money = get_money(self)
        self.item_collection_reward = 0
        self.previous_item_count = get_total_items(self)
        self.previous_unique_items = get_unique_items(self)
        self.total_enemy_pokemon = get_total_enemy_pokemon(self)
        self.total_enemy_defeated_reward = 0
        self.reward_scale = config.get("reward_scale", 1)
        self.extra_buttons = config.get("extra_buttons", False)
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []

        # Metadata and reward range
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

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

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.output_full, dtype=np.uint8
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()

        self.recent_memory = np.zeros(
            (self.output_shape[1] * self.memory_height, 3), dtype=np.uint8
        )
        self.recent_frames = np.zeros(
            (
                self.frame_stacks,
                self.output_shape[0],
                self.output_shape[1],
                self.output_shape[2],
            ),
            dtype=np.uint8,
        )
        self.agent_stats = []

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

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.progress_reward = get_game_state_reward(self)
        self.total_reward = sum(val for val in self.progress_reward.values())
        self.reset_count += 1

        initialize_enemy_hp(self)  # Initialize enemy HP states
        self.enemy_hp = [
            self.read_hp(addr) for addr in ENEMY_PARTY_POKEMON_HP
        ]  # Initialize enemy HP list

        return self.render(), {}

    def init_knn(self):
        self.knn_index = hnswlib.Index(space="l2", dim=self.vec_dim)
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16
        )

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()
        if reduce_res:
            game_pixels_render = (
                255 * resize(game_pixels_render, self.output_shape)
            ).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    (self.mem_padding, self.output_shape[1], 3), dtype=np.uint8
                )
                game_pixels_render = np.concatenate(
                    (
                        create_exploration_memory(self),
                        pad,
                        create_recent_memory(self),
                        pad,
                        rearrange(self.recent_frames, "f h w c -> (f h) w c"),
                    ),
                    axis=0,
                )
        return game_pixels_render

    def step(self, action):
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = (
            obs_memory[frame_start : frame_start + self.output_shape[0], ...]
            .flatten()
            .astype(np.float32)
        )

        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()

        update_heal_reward(self)
        self.party_size = self.read_m(NUM_POKEMON_IN_PARTY_ADDRESS)

        new_reward, new_prog = update_reward(self)
        self.last_health = read_hp_fraction(self)

        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()
        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward * 0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            if i == 8:
                if action < 4:
                    self.pyboy.send_input(self.release_arrow[action])
                if 3 < action < 6:
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False, update_mem=False)
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True, update_mem=False)
        )

    # Update agent stats
    def append_agent_stats(self, action):
        x_pos = self.read_m(CURRENT_PLAYER_X_POSITION)
        y_pos = self.read_m(CURRENT_PLAYER_Y_POSITION)
        map_n = self.read_m(CURRENT_MAP_NUMBER)
        levels = [self.read_m(a) for a in PARTY_POKEMON_ACTUAL_LEVEL]
        expl = (
            ("frames", self.knn_index.get_current_count())
            if self.use_screen_explore
            else ("coord_count", len(self.seen_coords))
        )
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": get_map_location(self, map_n),
                "last_action": action,
                "pcount": self.read_m(NUM_POKEMON_IN_PARTY_ADDRESS),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": read_hp_fraction(self),
                expl[0]: expl[1],
                "deaths": self.died_count,
                "badge": get_badges(self),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "powerful_move": self.progress_reward["powerful_move"],
            }
        )

    def update_frame_knn_index(self, frame_vec):
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            labels, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0][0] > self.similar_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_seen_coords(self):
        x_pos = self.read_m(CURRENT_PLAYER_X_POSITION)
        y_pos = self.read_m(CURRENT_PLAYER_Y_POSITION)
        map_n = self.read_m(CURRENT_MAP_NUMBER)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}

        self.seen_coords[coord_string] = self.step_count

    def get_total_items(self):
        return get_total_items(self)

    def get_unique_items(self):
        return get_unique_items(self)

    def update_item_collection_reward(self):
        update_item_collection_reward(self)

    def check_if_done(self):
        if self.early_stopping:
            done = self.step_count > 128 and self.recent_memory.sum() < (255 * 1)
        else:
            done = self.step_count >= self.max_steps
        return done

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

    def read_m(self, addr):
        if isinstance(addr, int):
            return self.pyboy.get_memory_value(addr)
        elif isinstance(addr, tuple):
            return 256 * self.read_m(addr[0]) + self.read_m(addr[1])
        else:
            raise TypeError(f"Address {addr} is not an integer or tuple")

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in PARTY_POKEMON_ACTUAL_LEVEL]
        return max(sum(poke_levels) - 4, 0)

    def read_party(self):
        return [self.read_m(addr) for addr in PARTY_POKEMON_ADDRESSES]

    def read_hp(self, addr_tuple):
        if isinstance(addr_tuple, tuple):
            start = addr_tuple[0]
            end = addr_tuple[1]
            return 256 * self.read_m(start) + self.read_m(end)
        else:
            raise TypeError(f"Address {addr_tuple} is not a valid tuple")
