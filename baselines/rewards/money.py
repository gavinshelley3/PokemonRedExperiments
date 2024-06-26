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


def get_money(env):
    money_bytes = [env.read_m(addr) for addr in MONEY]
    money = money_bytes[0] + (money_bytes[1] << 8) + (money_bytes[2] << 16)
    return money


def get_money_reward(env):
    current_money = get_money(env)
    money_reward = (current_money - env.previous_money) / 1000  # Scale the reward
    env.previous_money = current_money
    return money_reward
