from os.path import exists
from pathlib import Path
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
import wandb
from wandb.integration.sb3 import WandbCallback


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed environment.

    Args:
    - rank (int): Index of the subprocess.
    - env_conf (dict): Configuration dictionary for the environment.
    - seed (int): Initial seed for RNG.

    Returns:
    - function: Initialized environment with custom reward function.
    """

    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def find_next_session_path(base_path="session"):
    """
    Find the next available session path by incrementing a number.

    Args:
    - base_path (str): Base path for the session folders.

    Returns:
    - Path: Path to the next available session folder.
    """
    i = 1
    while True:
        sess_path = Path(f"{base_path}_{i}")
        if not sess_path.exists():
            sess_path.mkdir(parents=True, exist_ok=True)
            return sess_path
        i += 1


def main():
    use_wandb_logging = True  # Set to True to use Weights & Biases for logging
    ep_length = 1024 * 10  # Reduced Episode length
    sess_path = (
        find_next_session_path()
    )  # Create a session path with the next available number

    # Configuration section for hyperparameters
    hyperparameters = {
        "learning_rate": 0.0005,
        "batch_size": 128,
        "n_steps": 1024,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.015,
        "vf_coef": 0.4,
        "max_grad_norm": 0.55,
        "n_epochs": 10,
    }

    # Environment configuration
    env_config = {
        "headless": True,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../has_pokedex_nballs.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "reward_scale": 5,
        "extra_buttons": True,
    }

    num_cpu = 6  # Reduced number of parallel environments
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    print("\nLoading initial checkpoint")
    initial_checkpoint_path = "session_4da05e87_main_good/poke_439746560_steps"

    # Load the model and initialize it with hyperparameters
    model = PPO("CnnPolicy", env, verbose=1, **hyperparameters)
    model = PPO.load(
        initial_checkpoint_path,
        env=env,
        custom_objects={
            "learning_rate": hyperparameters["learning_rate"],
            "clip_range": hyperparameters["clip_range"],
        },
    )

    # Set up callbacks for saving checkpoints and logging with TensorBoard
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=sess_path,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    tensorboard_callback = TensorboardCallback()
    callbacks = [checkpoint_callback, tensorboard_callback]

    # Add wandb logging if enabled
    if use_wandb_logging:
        wandb.init(project="pokemon_red_rl", sync_tensorboard=True)
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    callback = CallbackList(callbacks)

    # Train the model with the specified hyperparameters
    model.learn(total_timesteps=8_000_000, callback=callback)  # Train the model
    model.save(sess_path / "final_model")  # Save the final model


if __name__ == "__main__":
    main()
