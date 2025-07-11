# move_constants.py

POUND = 0x01
KARATE_CHOP = 0x02
DOUBLESLAP = 0x03
COMET_PUNCH = 0x04
MEGA_PUNCH = 0x05
PAY_DAY = 0x06
FIRE_PUNCH = 0x07
ICE_PUNCH = 0x08
THUNDERPUNCH = 0x09
SCRATCH = 0x0A
VICEGRIP = 0x0B
GUILLOTINE = 0x0C
RAZOR_WIND = 0x0D
SWORDS_DANCE = 0x0E
CUT = 0x0F
GUST = 0x10
WING_ATTACK = 0x11
WHIRLWIND = 0x12
FLY = 0x13
BIND = 0x14
SLAM = 0x15
VINE_WHIP = 0x16
STOMP = 0x17
DOUBLE_KICK = 0x18
MEGA_KICK = 0x19
JUMP_KICK = 0x1A
ROLLING_KICK = 0x1B
SAND_ATTACK = 0x1C
HEADBUTT = 0x1D
HORN_ATTACK = 0x1E
FURY_ATTACK = 0x1F
HORN_DRILL = 0x20
TACKLE = 0x21
BODY_SLAM = 0x22
WRAP = 0x23
TAKE_DOWN = 0x24
THRASH = 0x25
DOUBLE_EDGE = 0x26
TAIL_WHIP = 0x27
POISON_STING = 0x28
TWINEEDLE = 0x29
PIN_MISSILE = 0x2A
LEER = 0x2B
BITE = 0x2C
GROWL = 0x2D
ROAR = 0x2E
SING = 0x2F
SUPERSONIC = 0x30
SONICBOOM = 0x31
DISABLE = 0x32
ACID = 0x33
EMBER = 0x34
FLAMETHROWER = 0x35
MIST = 0x36
WATER_GUN = 0x37
HYDRO_PUMP = 0x38
SURF = 0x39
ICE_BEAM = 0x3A
BLIZZARD = 0x3B
PSYBEAM = 0x3C
BUBBLEBEAM = 0x3D
AURORA_BEAM = 0x3E
HYPER_BEAM = 0x3F
PECK = 0x40
DRILL_PECK = 0x41
SUBMISSION = 0x42
LOW_KICK = 0x43
COUNTER = 0x44
SEISMIC_TOSS = 0x45
STRENGTH = 0x46
ABSORB = 0x47
MEGA_DRAIN = 0x48
LEECH_SEED = 0x49
GROWTH = 0x4A
RAZOR_LEAF = 0x4B
SOLARBEAM = 0x4C
POISONPOWDER = 0x4D
STUN_SPORE = 0x4E
SLEEP_POWDER = 0x4F
PETAL_DANCE = 0x50
STRING_SHOT = 0x51
DRAGON_RAGE = 0x52
FIRE_SPIN = 0x53
THUNDERSHOCK = 0x54
THUNDERBOLT = 0x55
THUNDER_WAVE = 0x56
THUNDER = 0x57
ROCK_THROW = 0x58
EARTHQUAKE = 0x59
FISSURE = 0x5A
DIG = 0x5B
TOXIC = 0x5C
CONFUSION = 0x5D
PSYCHIC_M = 0x5E
HYPNOSIS = 0x5F
MEDITATE = 0x60
AGILITY = 0x61
QUICK_ATTACK = 0x62
RAGE = 0x63
TELEPORT = 0x64
NIGHT_SHADE = 0x65
MIMIC = 0x66
SCREECH = 0x67
DOUBLE_TEAM = 0x68
RECOVER = 0x69
HARDEN = 0x6A
MINIMIZE = 0x6B
SMOKESCREEN = 0x6C
CONFUSE_RAY = 0x6D
WITHDRAW = 0x6E
DEFENSE_CURL = 0x6F
BARRIER = 0x70
LIGHT_SCREEN = 0x71
HAZE = 0x72
REFLECT = 0x73
FOCUS_ENERGY = 0x74
BIDE = 0x75
METRONOME = 0x76
MIRROR_MOVE = 0x77
SELFDESTRUCT = 0x78
EGG_BOMB = 0x79
LICK = 0x7A
SMOG = 0x7B
SLUDGE = 0x7C
BONE_CLUB = 0x7D
FIRE_BLAST = 0x7E
WATERFALL = 0x7F
CLAMP = 0x80
SWIFT = 0x81
SKULL_BASH = 0x82
SPIKE_CANNON = 0x83
CONSTRICT = 0x84
AMNESIA = 0x85
KINESIS = 0x86
SOFTBOILED = 0x87
HI_JUMP_KICK = 0x88
GLARE = 0x89
DREAM_EATER = 0x8A
POISON_GAS = 0x8B
BARRAGE = 0x8C
LEECH_LIFE = 0x8D
LOVELY_KISS = 0x8E
SKY_ATTACK = 0x8F
TRANSFORM = 0x90
BUBBLE = 0x91
DIZZY_PUNCH = 0x92
SPORE = 0x93
FLASH = 0x94
PSYWAVE = 0x95
SPLASH = 0x96
ACID_ARMOR = 0x97
CRABHAMMER = 0x98
EXPLOSION = 0x99
FURY_SWIPES = 0x9A
BONEMERANG = 0x9B
REST = 0x9C
ROCK_SLIDE = 0x9D
HYPER_FANG = 0x9E
SHARPEN = 0x9F
CONVERSION = 0xA0
TRI_ATTACK = 0xA1
SUPER_FANG = 0xA2
SLASH = 0xA3
SUBSTITUTE = 0xA4


# Characteristics of each move.

MOVES = [
    # (animation, effect, power, type, accuracy, pp)
    ("POUND", "NO_ADDITIONAL_EFFECT", 40, "NORMAL", 100, 35),
    ("KARATE_CHOP", "NO_ADDITIONAL_EFFECT", 50, "NORMAL", 100, 25),
    ("DOUBLESLAP", "TWO_TO_FIVE_ATTACKS_EFFECT", 15, "NORMAL", 85, 10),
    ("COMET_PUNCH", "TWO_TO_FIVE_ATTACKS_EFFECT", 18, "NORMAL", 85, 15),
    ("MEGA_PUNCH", "NO_ADDITIONAL_EFFECT", 80, "NORMAL", 85, 20),
    ("PAY_DAY", "PAY_DAY_EFFECT", 40, "NORMAL", 100, 20),
    ("FIRE_PUNCH", "BURN_SIDE_EFFECT1", 75, "FIRE", 100, 15),
    ("ICE_PUNCH", "FREEZE_SIDE_EFFECT", 75, "ICE", 100, 15),
    ("THUNDERPUNCH", "PARALYZE_SIDE_EFFECT1", 75, "ELECTRIC", 100, 15),
    ("SCRATCH", "NO_ADDITIONAL_EFFECT", 40, "NORMAL", 100, 35),
    ("VICEGRIP", "NO_ADDITIONAL_EFFECT", 55, "NORMAL", 100, 30),
    ("GUILLOTINE", "OHKO_EFFECT", 1, "NORMAL", 30, 5),
    ("RAZOR_WIND", "CHARGE_EFFECT", 80, "NORMAL", 75, 10),
    ("SWORDS_DANCE", "ATTACK_UP2_EFFECT", 0, "NORMAL", 100, 30),
    ("CUT", "NO_ADDITIONAL_EFFECT", 50, "NORMAL", 95, 30),
    ("GUST", "NO_ADDITIONAL_EFFECT", 40, "NORMAL", 100, 35),
    ("WING_ATTACK", "NO_ADDITIONAL_EFFECT", 35, "FLYING", 100, 35),
    ("WHIRLWIND", "SWITCH_AND_TELEPORT_EFFECT", 0, "NORMAL", 85, 20),
    ("FLY", "FLY_EFFECT", 70, "FLYING", 95, 15),
    ("BIND", "TRAPPING_EFFECT", 15, "NORMAL", 75, 20),
    ("SLAM", "NO_ADDITIONAL_EFFECT", 80, "NORMAL", 75, 20),
    ("VINE_WHIP", "NO_ADDITIONAL_EFFECT", 35, "GRASS", 100, 10),
    ("STOMP", "FLINCH_SIDE_EFFECT2", 65, "NORMAL", 100, 20),
    ("DOUBLE_KICK", "ATTACK_TWICE_EFFECT", 30, "FIGHTING", 100, 30),
    ("MEGA_KICK", "NO_ADDITIONAL_EFFECT", 120, "NORMAL", 75, 5),
    ("JUMP_KICK", "JUMP_KICK_EFFECT", 70, "FIGHTING", 95, 25),
    ("ROLLING_KICK", "FLINCH_SIDE_EFFECT2", 60, "FIGHTING", 85, 15),
    ("SAND_ATTACK", "ACCURACY_DOWN1_EFFECT", 0, "NORMAL", 100, 15),
    ("HEADBUTT", "FLINCH_SIDE_EFFECT2", 70, "NORMAL", 100, 15),
    ("HORN_ATTACK", "NO_ADDITIONAL_EFFECT", 65, "NORMAL", 100, 25),
    ("FURY_ATTACK", "TWO_TO_FIVE_ATTACKS_EFFECT", 15, "NORMAL", 85, 20),
    ("HORN_DRILL", "OHKO_EFFECT", 1, "NORMAL", 30, 5),
    ("TACKLE", "NO_ADDITIONAL_EFFECT", 35, "NORMAL", 95, 35),
    ("BODY_SLAM", "PARALYZE_SIDE_EFFECT2", 85, "NORMAL", 100, 15),
    ("WRAP", "TRAPPING_EFFECT", 15, "NORMAL", 85, 20),
    ("TAKE_DOWN", "RECOIL_EFFECT", 90, "NORMAL", 85, 20),
    ("THRASH", "THRASH_PETAL_DANCE_EFFECT", 90, "NORMAL", 100, 20),
    ("DOUBLE_EDGE", "RECOIL_EFFECT", 100, "NORMAL", 100, 15),
    ("TAIL_WHIP", "DEFENSE_DOWN1_EFFECT", 0, "NORMAL", 100, 30),
    ("POISON_STING", "POISON_SIDE_EFFECT1", 15, "POISON", 100, 35),
    ("TWINEEDLE", "TWINEEDLE_EFFECT", 25, "BUG", 100, 20),
    ("PIN_MISSILE", "TWO_TO_FIVE_ATTACKS_EFFECT", 14, "BUG", 85, 20),
    ("LEER", "DEFENSE_DOWN1_EFFECT", 0, "NORMAL", 100, 30),
    ("BITE", "FLINCH_SIDE_EFFECT1", 60, "NORMAL", 100, 25),
    ("GROWL", "ATTACK_DOWN1_EFFECT", 0, "NORMAL", 100, 40),
    ("ROAR", "SWITCH_AND_TELEPORT_EFFECT", 0, "NORMAL", 100, 20),
    ("SING", "SLEEP_EFFECT", 0, "NORMAL", 55, 15),
    ("SUPERSONIC", "CONFUSION_EFFECT", 0, "NORMAL", 55, 20),
    ("SONICBOOM", "SPECIAL_DAMAGE_EFFECT", 1, "NORMAL", 90, 20),
    ("DISABLE", "DISABLE_EFFECT", 0, "NORMAL", 55, 20),
    ("ACID", "DEFENSE_DOWN_SIDE_EFFECT", 40, "POISON", 100, 30),
    ("EMBER", "BURN_SIDE_EFFECT1", 40, "FIRE", 100, 25),
    ("FLAMETHROWER", "BURN_SIDE_EFFECT1", 95, "FIRE", 100, 15),
    ("MIST", "MIST_EFFECT", 0, "ICE", 100, 30),
    ("WATER_GUN", "NO_ADDITIONAL_EFFECT", 40, "WATER", 100, 25),
    ("HYDRO_PUMP", "NO_ADDITIONAL_EFFECT", 120, "WATER", 80, 5),
    ("SURF", "NO_ADDITIONAL_EFFECT", 95, "WATER", 100, 15),
    ("ICE_BEAM", "FREEZE_SIDE_EFFECT", 95, "ICE", 100, 10),
    ("BLIZZARD", "FREEZE_SIDE_EFFECT", 120, "ICE", 90, 5),
    ("PSYBEAM", "CONFUSION_SIDE_EFFECT", 65, "PSYCHIC", 100, 20),
    ("BUBBLEBEAM", "SPEED_DOWN_SIDE_EFFECT", 65, "WATER", 100, 20),
    ("AURORA_BEAM", "ATTACK_DOWN_SIDE_EFFECT", 65, "ICE", 100, 20),
    ("HYPER_BEAM", "HYPER_BEAM_EFFECT", 150, "NORMAL", 90, 5),
    ("PECK", "NO_ADDITIONAL_EFFECT", 35, "FLYING", 100, 35),
    ("DRILL_PECK", "NO_ADDITIONAL_EFFECT", 80, "FLYING", 100, 20),
    ("SUBMISSION", "RECOIL_EFFECT", 80, "FIGHTING", 80, 25),
    ("LOW_KICK", "FLINCH_SIDE_EFFECT2", 50, "FIGHTING", 90, 20),
    ("COUNTER", "NO_ADDITIONAL_EFFECT", 1, "FIGHTING", 100, 20),
    ("SEISMIC_TOSS", "SPECIAL_DAMAGE_EFFECT", 1, "FIGHTING", 100, 20),
    ("STRENGTH", "NO_ADDITIONAL_EFFECT", 80, "NORMAL", 100, 15),
    ("ABSORB", "DRAIN_HP_EFFECT", 20, "GRASS", 100, 20),
    ("MEGA_DRAIN", "DRAIN_HP_EFFECT", 40, "GRASS", 100, 10),
    ("LEECH_SEED", "LEECH_SEED_EFFECT", 0, "GRASS", 90, 10),
    ("GROWTH", "SPECIAL_UP1_EFFECT", 0, "NORMAL", 100, 40),
    ("RAZOR_LEAF", "NO_ADDITIONAL_EFFECT", 55, "GRASS", 95, 25),
    ("SOLARBEAM", "CHARGE_EFFECT", 120, "GRASS", 100, 10),
    ("POISONPOWDER", "POISON_EFFECT", 0, "POISON", 75, 35),
    ("STUN_SPORE", "PARALYZE_EFFECT", 0, "GRASS", 75, 30),
    ("SLEEP_POWDER", "SLEEP_EFFECT", 0, "GRASS", 75, 15),
    ("PETAL_DANCE", "THRASH_PETAL_DANCE_EFFECT", 70, "GRASS", 100, 20),
    ("STRING_SHOT", "SPEED_DOWN1_EFFECT", 0, "BUG", 95, 40),
    ("DRAGON_RAGE", "SPECIAL_DAMAGE_EFFECT", 1, "DRAGON", 100, 10),
    ("FIRE_SPIN", "TRAPPING_EFFECT", 15, "FIRE", 70, 15),
    ("THUNDERSHOCK", "PARALYZE_SIDE_EFFECT1", 40, "ELECTRIC", 100, 30),
    ("THUNDERBOLT", "PARALYZE_SIDE_EFFECT1", 95, "ELECTRIC", 100, 15),
    ("THUNDER_WAVE", "PARALYZE_EFFECT", 0, "ELECTRIC", 100, 20),
    ("THUNDER", "PARALYZE_SIDE_EFFECT1", 120, "ELECTRIC", 70, 10),
    ("ROCK_THROW", "NO_ADDITIONAL_EFFECT", 50, "ROCK", 65, 15),
    ("EARTHQUAKE", "NO_ADDITIONAL_EFFECT", 100, "GROUND", 100, 10),
    ("FISSURE", "OHKO_EFFECT", 1, "GROUND", 30, 5),
    ("DIG", "CHARGE_EFFECT", 100, "GROUND", 100, 10),
    ("TOXIC", "POISON_EFFECT", 0, "POISON", 85, 10),
    ("CONFUSION", "CONFUSION_SIDE_EFFECT", 50, "PSYCHIC", 100, 25),
    ("PSYCHIC_M", "SPECIAL_DOWN_SIDE_EFFECT", 90, "PSYCHIC", 100, 10),
    ("HYPNOSIS", "SLEEP_EFFECT", 0, "PSYCHIC", 60, 20),
    ("MEDITATE", "ATTACK_UP1_EFFECT", 0, "PSYCHIC", 100, 40),
    ("AGILITY", "SPEED_UP2_EFFECT", 0, "PSYCHIC", 100, 30),
    ("QUICK_ATTACK", "NO_ADDITIONAL_EFFECT", 40, "NORMAL", 100, 30),
    ("RAGE", "RAGE_EFFECT", 20, "NORMAL", 100, 20),
    ("TELEPORT", "SWITCH_AND_TELEPORT_EFFECT", 0, "PSYCHIC", 100, 20),
    ("NIGHT_SHADE", "SPECIAL_DAMAGE_EFFECT", 0, "GHOST", 100, 15),
    ("MIMIC", "MIMIC_EFFECT", 0, "NORMAL", 100, 10),
    ("SCREECH", "DEFENSE_DOWN2_EFFECT", 0, "NORMAL", 85, 40),
    ("DOUBLE_TEAM", "EVASION_UP1_EFFECT", 0, "NORMAL", 100, 15),
    ("RECOVER", "HEAL_EFFECT", 0, "NORMAL", 100, 20),
    ("HARDEN", "DEFENSE_UP1_EFFECT", 0, "NORMAL", 100, 30),
    ("MINIMIZE", "EVASION_UP1_EFFECT", 0, "NORMAL", 100, 20),
    ("SMOKESCREEN", "ACCURACY_DOWN1_EFFECT", 0, "NORMAL", 100, 20),
    ("CONFUSE_RAY", "CONFUSION_EFFECT", 0, "GHOST", 100, 10),
    ("WITHDRAW", "DEFENSE_UP1_EFFECT", 0, "WATER", 100, 40),
    ("DEFENSE_CURL", "DEFENSE_UP1_EFFECT", 0, "NORMAL", 100, 40),
    ("BARRIER", "DEFENSE_UP2_EFFECT", 0, "PSYCHIC", 100, 30),
    ("LIGHT_SCREEN", "LIGHT_SCREEN_EFFECT", 0, "PSYCHIC", 100, 30),
    ("HAZE", "HAZE_EFFECT", 0, "ICE", 100, 30),
    ("REFLECT", "REFLECT_EFFECT", 0, "PSYCHIC", 100, 20),
    ("FOCUS_ENERGY", "FOCUS_ENERGY_EFFECT", 0, "NORMAL", 100, 30),
    ("BIDE", "BIDE_EFFECT", 0, "NORMAL", 100, 10),
    ("METRONOME", "METRONOME_EFFECT", 0, "NORMAL", 100, 10),
    ("MIRROR_MOVE", "MIRROR_MOVE_EFFECT", 0, "FLYING", 100, 20),
    ("SELFDESTRUCT", "EXPLODE_EFFECT", 130, "NORMAL", 100, 5),
    ("EGG_BOMB", "NO_ADDITIONAL_EFFECT", 100, "NORMAL", 75, 10),
    ("LICK", "PARALYZE_SIDE_EFFECT2", 20, "GHOST", 100, 30),
    ("SMOG", "POISON_SIDE_EFFECT2", 20, "POISON", 70, 20),
    ("SLUDGE", "POISON_SIDE_EFFECT2", 65, "POISON", 100, 20),
    ("BONE_CLUB", "FLINCH_SIDE_EFFECT1", 65, "GROUND", 85, 20),
    ("FIRE_BLAST", "BURN_SIDE_EFFECT2", 120, "FIRE", 85, 5),
    ("WATERFALL", "NO_ADDITIONAL_EFFECT", 80, "WATER", 100, 15),
    ("CLAMP", "TRAPPING_EFFECT", 35, "WATER", 75, 10),
    ("SWIFT", "SWIFT_EFFECT", 60, "NORMAL", 100, 20),
    ("SKULL_BASH", "CHARGE_EFFECT", 100, "NORMAL", 100, 15),
    ("SPIKE_CANNON", "TWO_TO_FIVE_ATTACKS_EFFECT", 20, "NORMAL", 100, 15),
    ("CONSTRICT", "SPEED_DOWN_SIDE_EFFECT", 10, "NORMAL", 100, 35),
    ("AMNESIA", "SPECIAL_UP2_EFFECT", 0, "PSYCHIC", 100, 20),
    ("KINESIS", "ACCURACY_DOWN1_EFFECT", 0, "PSYCHIC", 80, 15),
    ("SOFTBOILED", "HEAL_EFFECT", 0, "NORMAL", 100, 10),
    ("HI_JUMP_KICK", "JUMP_KICK_EFFECT", 85, "FIGHTING", 90, 20),
    ("GLARE", "PARALYZE_EFFECT", 0, "NORMAL", 75, 30),
    ("DREAM_EATER", "DREAM_EATER_EFFECT", 100, "PSYCHIC", 100, 15),
    ("POISON_GAS", "POISON_EFFECT", 0, "POISON", 55, 40),
    ("BARRAGE", "TWO_TO_FIVE_ATTACKS_EFFECT", 15, "NORMAL", 85, 20),
    ("LEECH_LIFE", "DRAIN_HP_EFFECT", 20, "BUG", 100, 15),
    ("LOVELY_KISS", "SLEEP_EFFECT", 0, "NORMAL", 75, 10),
    ("SKY_ATTACK", "CHARGE_EFFECT", 140, "FLYING", 90, 5),
    ("TRANSFORM", "TRANSFORM_EFFECT", 0, "NORMAL", 100, 10),
    ("BUBBLE", "SPEED_DOWN_SIDE_EFFECT", 20, "WATER", 100, 30),
    ("DIZZY_PUNCH", "NO_ADDITIONAL_EFFECT", 70, "NORMAL", 100, 10),
    ("SPORE", "SLEEP_EFFECT", 0, "GRASS", 100, 15),
    ("FLASH", "ACCURACY_DOWN1_EFFECT", 0, "NORMAL", 70, 20),
    ("PSYWAVE", "SPECIAL_DAMAGE_EFFECT", 1, "PSYCHIC", 80, 15),
    ("SPLASH", "SPLASH_EFFECT", 0, "NORMAL", 100, 40),
    ("ACID_ARMOR", "DEFENSE_UP2_EFFECT", 0, "POISON", 100, 40),
    ("CRABHAMMER", "NO_ADDITIONAL_EFFECT", 90, "WATER", 85, 10),
    ("EXPLOSION", "EXPLODE_EFFECT", 170, "NORMAL", 100, 5),
    ("FURY_SWIPES", "TWO_TO_FIVE_ATTACKS_EFFECT", 18, "NORMAL", 80, 15),
    ("BONEMERANG", "ATTACK_TWICE_EFFECT", 50, "GROUND", 90, 10),
    ("REST", "HEAL_EFFECT", 0, "PSYCHIC", 100, 10),
    ("ROCK_SLIDE", "NO_ADDITIONAL_EFFECT", 75, "ROCK", 90, 10),
    ("HYPER_FANG", "FLINCH_SIDE_EFFECT1", 80, "NORMAL", 90, 15),
    ("SHARPEN", "ATTACK_UP1_EFFECT", 0, "NORMAL", 100, 30),
    ("CONVERSION", "CONVERSION_EFFECT", 0, "NORMAL", 100, 30),
    ("TRI_ATTACK", "NO_ADDITIONAL_EFFECT", 80, "NORMAL", 100, 10),
    ("SUPER_FANG", "SUPER_FANG_EFFECT", 1, "NORMAL", 90, 10),
    ("SLASH", "NO_ADDITIONAL_EFFECT", 70, "NORMAL", 100, 20),
    ("SUBSTITUTE", "SUBSTITUTE_EFFECT", 0, "NORMAL", 100, 10),
    ("STRUGGLE", "RECOIL_EFFECT", 50, "NORMAL", 100, 10),
]
