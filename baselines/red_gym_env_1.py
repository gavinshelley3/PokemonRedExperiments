import sys
import uuid
import os
from math import floor, sqrt
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
from baselines.constants.event_constants import *
from baselines.constants.item_constants import *
from baselines.constants.player_constants import *
from baselines.constants.opponent_trainer_constants import *
from baselines.constants.map_constants import *


class RedGymEnv(Env):
    def __init__(self, config=None):
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
        self.previous_item_count = 0
        self.previous_unique_items = 0
        self.reward_scale = config.get("reward_scale", 1)
        self.extra_buttons = config.get("extra_buttons", False)
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        self.item_collection_reward = 0  # Initialize item collection reward

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

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

        head = "headless" if config["headless"] else "SDL2"
        self.pyboy = PyBoy(
            config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window="--quiet" in sys.argv,
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
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum(val for val in self.progress_reward.values())
        self.previous_item_count = 0  # Reset previous item count
        self.previous_unique_items = 0  # Reset previous unique items
        self.item_collection_reward = 0  # Reset item collection reward
        self.reset_count += 1
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
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
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

        self.update_heal_reward()
        self.party_size = self.read_m(NUM_POKEMON_IN_PARTY_ADDRESS)

        self.update_item_collection_reward()  # Update item collection reward

        new_reward, new_prog = self.update_reward()
        self.last_health = self.read_hp_fraction()

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
                "map_location": self.get_map_location(map_n),
                "last_action": action,
                "pcount": self.read_m(NUM_POKEMON_IN_PARTY_ADDRESS),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                expl[0]: expl[1],
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
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
        ITEM_ENTRY_SIZE = 2
        MAX_ITEMS = 20

        total_items = 0
        for i in range(MAX_ITEMS):
            addr = START_OF_ITEM_LIST + (i * ITEM_ENTRY_SIZE)
            if addr >= END_OF_ITEM_LIST:
                break

            item_id = self.read_m(addr)
            if item_id != 0:  # 0 means empty slot
                item_quantity = self.read_m(addr + 1)
                total_items += item_quantity

        return total_items

    def get_unique_items(self):
        ITEM_ENTRY_SIZE = 2
        MAX_ITEMS = 20

        unique_items = set()
        for i in range(MAX_ITEMS):
            addr = START_OF_ITEM_LIST + (i * ITEM_ENTRY_SIZE)
            if addr >= END_OF_ITEM_LIST:
                break

            item_id = self.read_m(addr)
            if item_id != 0:
                unique_items.add(item_id)

        return len(unique_items)

    def update_item_collection_reward(self):
        current_item_count = self.get_total_items()
        current_unique_items = self.get_unique_items()

        new_items = current_item_count - self.previous_item_count
        new_unique_items = current_unique_items - self.previous_unique_items

        if new_items > 0:
            self.item_collection_reward += new_items * 0.25

        if new_unique_items > 0:
            self.item_collection_reward += (
                new_unique_items * 0.5
            )  # Higher reward for unique items

        self.previous_item_count = current_item_count
        self.previous_unique_items = current_unique_items

    def update_reward(self):
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
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

    def group_rewards(self):
        prog = self.progress_reward
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def create_exploration_memory(self):
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

        level, hp, explore = self.group_rewards()
        full_memory = np.stack(
            (
                make_reward_channel(level),
                make_reward_channel(hp),
                make_reward_channel(explore),
            ),
            axis=-1,
        )

        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(self.recent_memory, "(w h) c -> h w c", h=self.memory_height)

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
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in PARTY_POKEMON_ACTUAL_LEVEL]
        return max(sum(poke_levels) - 4, 0)

    def get_level_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_knn_reward(self):
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = (
            self.knn_index.get_current_count()
            if self.use_screen_explore
            else len(self.seen_coords)
        )
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def get_badges(self):
        return self.bit_count(self.read_m(BADGES))

    def read_party(self):
        return [self.read_m(addr) for addr in PARTY_POKEMON_ADDRESSES]

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if (
            cur_health > self.last_health
            and self.read_m(NUM_POKEMON_IN_PARTY_ADDRESS) == self.party_size
        ):
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f"healed: {heal_amount}")
                    self.save_screenshot("healing")
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def get_all_events_reward(self):
        event_flags_start = EVENT_000_FOLLOWED_OAK_INTO_LAB
        event_flags_end = EVENT_9FF
        base_event_flags = 13
        return max(
            sum(
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            )
            - base_event_flags
            - 0,
        )

    def get_game_state_reward(self, print_stats=False):
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew(),
            "level": self.reward_scale * self.get_level_reward(),
            "heal": self.reward_scale * self.total_healing_rew,
            "op_lvl": self.reward_scale * self.update_max_op_level(),
            "dead": self.reward_scale * -0.1 * self.died_count,
            "badge": self.reward_scale * self.get_badges() * 5,
            "explore": self.reward_scale * self.get_knn_reward(),
            "item_collection": self.reward_scale
            * self.item_collection_reward,  # Add item collection reward
        }
        return state_scores

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

    def update_max_op_level(self):
        opponent_level = max(self.read_m(a) for a in ENEMY_PARTY_POKEMON_LEVEL) - 5
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = sum(self.read_hp(add) for add in PARTY_POKEMON_HP)
        max_hp_sum = sum(self.read_hp(add) for add in PARTY_POKEMON_MAX_HP)
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def bit_count(self, bits):
        return bin(bits).count("1")

    def read_triple(self, start_add):
        return (
            256 * 256 * self.read_m(start_add)
            + 256 * self.read_m(start_add + 1)
            + self.read_m(start_add + 2)
        )

    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

    def read_money(self):
        return (
            100 * 100 * self.read_bcd(self.read_m(MONEY_1))
            + 100 * self.read_bcd(self.read_m(MONEY_2))
            + self.read_bcd(self.read_m(MONEY_3))
        )

    def get_map_location(self, map_idx):
        map_locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)",
        }
        return map_locations.get(map_idx, "Unknown Location")
