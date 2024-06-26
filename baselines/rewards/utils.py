from math import floor
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


def read_hp_fraction(env):
    hp_sum = sum(env.read_hp(addr) for addr in PARTY_POKEMON_HP)
    max_hp_sum = sum(env.read_hp(addr) for addr in PARTY_POKEMON_MAX_HP)
    max_hp_sum = max(max_hp_sum, 1)
    return hp_sum / max_hp_sum


def bit_count(value):
    return bin(value).count("1")


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
