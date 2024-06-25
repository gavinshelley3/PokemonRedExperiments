# opponent_trainer_constants.py

# Total enemy Pokémon
TOTAL_ENEMY_POKEMON = 0xD89C

# Enemy Pokémon in party
ENEMY_POKEMON_PARTY = [
    0xD89D,  # Pokémon 1
    0xD89E,  # Pokémon 2
    0xD89F,  # Pokémon 3
    0xD8A0,  # Pokémon 4
    0xD8A1,  # Pokémon 5
    0xD8A2,  # Pokémon 6
]

# Enemy Pokémon data start addresses
ENEMY_POKEMON_DATA = [
    0xD8A4,  # Pokémon 1 data start
    0xD8D0,  # Pokémon 2 data start
    0xD8FC,  # Pokémon 3 data start
    0xD928,  # Pokémon 4 data start
    0xD954,  # Pokémon 5 data start
    0xD980,  # Pokémon 6 data start
]

# Offsets for individual Pokémon data
CURRENT_HP_OFFSET = 0x01
STATUS_OFFSET = 0x04
TYPE_1_OFFSET = 0x05
TYPE_2_OFFSET = 0x06
MOVE_1_OFFSET = 0x08
MOVE_2_OFFSET = 0x09
MOVE_3_OFFSET = 0x0A
MOVE_4_OFFSET = 0x0B
LEVEL_OFFSET = 0x21
MAX_HP_OFFSET = 0x22
ATTACK_OFFSET = 0x24
DEFENSE_OFFSET = 0x26
SPEED_OFFSET = 0x28
SPECIAL_OFFSET = 0x2A
