# player_constants.py

# Player's name
PLAYER_NAME_ADDRESS = (0xD158, 0xD162)  # INT: 0xD158-0xD162

# Number of Pokémon in the party
NUM_POKEMON_IN_PARTY_ADDRESS = 0xD163  # INT: 0xD163

# Pokémon in the party
PARTY_POKEMON_ADDRESSES = [
    0xD164,  # Pokémon 1
    0xD165,  # Pokémon 2
    0xD166,  # Pokémon 3
    0xD167,  # Pokémon 4
    0xD168,  # Pokémon 5
    0xD169,  # Pokémon 6
    0xD16A   # End of list
]

# Pokémon details (in the party)
PARTY_POKEMON_DETAILS = [
    {
        "pokemon": 0xD16B,
        "current_hp": (0xD16C, 0xD16D),
        "level": 0xD16E,
        "status": 0xD16F,
        "type1": 0xD170,
        "type2": 0xD171,
        "catch_rate_or_held_item": 0xD172,
        "moves": [0xD173, 0xD174, 0xD175, 0xD176],
        "trainer_id": (0xD177, 0xD178),
        "experience": (0xD179, 0xD17B),
        "hp_ev": (0xD17C, 0xD17D),
        "attack_ev": (0xD17E, 0xD17F),
        "defense_ev": (0xD180, 0xD181),
        "speed_ev": (0xD182, 0xD183),
        "special_ev": (0xD184, 0xD185),
        "attack_defense_iv": 0xD186,
        "speed_special_iv": 0xD187,
        "pp_moves": [0xD188, 0xD189, 0xD18A, 0xD18B],
        "actual_level": 0xD18C,
        "max_hp": (0xD18D, 0xD18E),
        "attack": (0xD18F, 0xD190),
        "defense": (0xD191, 0xD192),
        "speed": (0xD193, 0xD194),
        "special": (0xD195, 0xD196)
    },
    {
        "pokemon": 0xD197,
        "current_hp": (0xD198, 0xD199),
        "level": 0xD19A,
        "status": 0xD19B,
        "type1": 0xD19C,
        "type2": 0xD19D,
        "catch_rate_or_held_item": 0xD19E,
        "moves": [0xD19F, 0xD1A0, 0xD1A1, 0xD1A2],
        "trainer_id": (0xD1A3, 0xD1A4),
        "experience": (0xD1A5, 0xD1A7),
        "hp_ev": (0xD1A8, 0xD1A9),
        "attack_ev": (0xD1AA, 0xD1AB),
        "defense_ev": (0xD1AC, 0xD1AD),
        "speed_ev": (0xD1AE, 0xD1AF),
        "special_ev": (0xD1B0, 0xD1B1),
        "attack_defense_iv": 0xD1B2,
        "speed_special_iv": 0xD1B3,
        "pp_moves": [0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7],
        "actual_level": 0xD1B8,
        "max_hp": (0xD1B9, 0xD1BA),
        "attack": (0xD1BB, 0xD1BC),
        "defense": (0xD1BD, 0xD1BE),
        "speed": (0xD1BF, 0xD1C0),
        "special": (0xD1C1, 0xD1C2)
    },
    {
        "pokemon": 0xD1C3,
        "current_hp": (0xD1C4, 0xD1C5),
        "level": 0xD1C6,
        "status": 0xD1C7,
        "type1": 0xD1C8,
        "type2": 0xD1C9,
        "catch_rate_or_held_item": 0xD1CA,
        "moves": [0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE],
        "trainer_id": (0xD1CF, 0xD1D0),
        "experience": (0xD1D1, 0xD1D3),
        "hp_ev": (0xD1D4, 0xD1D5),
        "attack_ev": (0xD1D6, 0xD1D7),
        "defense_ev": (0xD1D8, 0xD1D9),
        "speed_ev": (0xD1DA, 0xD1DB),
        "special_ev": (0xD1DC, 0xD1DD),
        "attack_defense_iv": 0xD1DE,
        "speed_special_iv": 0xD1DF,
        "pp_moves": [0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3],
        "actual_level": 0xD1E4,
        "max_hp": (0xD1E5, 0xD1E6),
        "attack": (0xD1E7, 0xD1E8),
        "defense": (0xD1E9, 0xD1EA),
        "speed": (0xD1EB, 0xD1EC),
        "special": (0xD1ED, 0xD1EE)
    },
    {
        "pokemon": 0xD1EF,
        "current_hp": (0xD1F0, 0xD1F1),
        "level": 0xD1F2,
        "status": 0xD1F3,
        "type1": 0xD1F4,
        "type2": 0xD1F5,
        "catch_rate_or_held_item": 0xD1F6,
        "moves": [0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA],
        "trainer_id": (0xD1FB, 0xD1FC),
        "experience": (0xD1FD, 0xD1FF),
        "hp_ev": (0xD200, 0xD201),
        "attack_ev": (0xD202, 0xD203),
        "defense_ev": (0xD204, 0xD205),
        "speed_ev": (0xD206, 0xD207),
        "special_ev": (0xD208, 0xD209),
        "attack_defense_iv": 0xD20A,
        "speed_special_iv": 0xD20B,
        "pp_moves": [0xD20C, 0xD20D, 0xD20E, 0xD20F],
        "actual_level": 0xD210,
        "max_hp": (0xD211, 0xD212),
        "attack": (0xD213, 0xD214),
        "defense": (0xD215, 0xD216),
        "speed": (0xD217, 0xD218),
        "special": (0xD219, 0xD21A)
    },
    {
        "pokemon": 0xD21B,
        "current_hp": (0xD21C, 0xD21D),
        "level": 0xD21E,
        "status": 0xD21F,
        "type1": 0xD220,
        "type2": 0xD221,
        "catch_rate_or_held_item": 0xD222,
        "moves": [0xD223, 0xD224, 0xD225, 0xD226],
        "trainer_id": (0xD227, 0xD228),
        "experience": (0xD229, 0xD22B),
        "hp_ev": (0xD22C, 0xD22D),
        "attack_ev": (0xD22E, 0xD22F),
        "defense_ev": (0xD230, 0xD231),
        "speed_ev": (0xD232, 0xD233),
        "special_ev": (0xD234, 0xD235),
        "attack_defense_iv": 0xD236,
        "speed_special_iv": 0xD237,
        "pp_moves": [0xD238, 0xD239, 0xD23A, 0xD23B],
        "actual_level": 0xD23C,
        "max_hp": (0xD23D, 0xD23E),
        "attack": (0xD23F, 0xD240),
        "defense": (0xD241, 0xD242),
        "speed": (0xD243, 0xD244),
        "special": (0xD245, 0xD246)
    },
    {
        "pokemon": 0xD247,
        "current_hp": (0xD248, 0xD249),
        "level": 0xD24A,
        "status": 0xD24B,
        "type1": 0xD24C,
        "type2": 0xD24D,
        "catch_rate_or_held_item": 0xD24E,
        "moves": [0xD24F, 0xD250, 0xD251, 0xD252],
        "trainer_id": (0xD253, 0xD254),
        "experience": (0xD255, 0xD257),
        "hp_ev": (0xD258, 0xD259),
        "attack_ev": (0xD25A, 0xD25B),
        "defense_ev": (0xD25C, 0xD25D),
        "speed_ev": (0xD25E, 0xD25F),
        "special_ev": (0xD260, 0xD261),
        "attack_defense_iv": 0xD262,
        "speed_special_iv": 0xD263,
        "pp_moves": [0xD264, 0xD265, 0xD266, 0xD267],
        "actual_level": 0xD268,
        "max_hp": (0xD269, 0xD26A),
        "attack": (0xD26B, 0xD26C),
        "defense": (0xD26D, 0xD26E),
        "speed": (0xD26F, 0xD270),
        "special": (0xD271, 0xD272)
    }
]
