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
from rewards.utils import read_hp_fraction, save_screenshot


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
