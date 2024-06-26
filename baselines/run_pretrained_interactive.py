import tensorflow as tf
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def find_next_session_path(base_path="session/session"):
    i = 1
    while True:
        sess_path = Path(f"{base_path}_{i}")
        if not sess_path.exists():
            sess_path.mkdir(parents=True, exist_ok=True)
            return sess_path
        i += 1


def run_experiment(update_interval):
    sess_path = find_next_session_path()
    ep_length = 2**23

    env_config = {
        "headless": False,
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
        "reward_scale": 4,
        "extra_buttons": True,
    }

    num_cpu = 1
    env = make_env(0, env_config)()

    file_name = "session/session_4da05e87_main_good/poke_439746560_steps"
    print("\nloading checkpoint")
    model = PPO.load(
        file_name, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
    )

    log_dir = sess_path / "logs"
    writer = tf.summary.create_file_writer(str(log_dir))

    obs_buffer, action_buffer, reward_buffer, next_obs_buffer, done_buffer = (
        [],
        [],
        [],
        [],
        [],
    )

    obs, info = env.reset()
    step = 0
    total_rewards = 0

    while True:
        action, _states = model.predict(obs, deterministic=False)

        if action >= len(env.valid_actions):
            print(f"Invalid action: {action}. Defaulting to no-op action.")
            action = len(env.valid_actions) - 1

        try:
            next_obs, rewards, terminated, truncated, info = env.step(action)
        except IndexError:
            print(f"Error: Action {action} is out of bounds.")
            break

        env.render()

        with writer.as_default():
            tf.summary.scalar("reward", rewards, step=step)
            for key, value in info.items():
                tf.summary.scalar(key, value, step=step)

        obs_buffer.append(obs)
        action_buffer.append(action)
        reward_buffer.append(rewards)
        next_obs_buffer.append(next_obs)
        done_buffer.append(terminated or truncated)

        if len(obs_buffer) >= update_interval:
            print(f"Updating model at step {step}")
            model.learn(total_timesteps=update_interval, log_interval=1)
            obs_buffer, action_buffer, reward_buffer, next_obs_buffer, done_buffer = (
                [],
                [],
                [],
                [],
                [],
            )

        obs = next_obs
        total_rewards += rewards
        step += 1

        if terminated or truncated:
            break

    env.close()
    return total_rewards


def find_best_update_interval():
    intervals = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    results = {}

    for interval in intervals:
        print(f"Running experiment with update interval: {interval}")
        total_rewards = run_experiment(interval)
        results[interval] = total_rewards
        print(f"Total rewards for update interval {interval}: {total_rewards}")

    best_interval = max(results, key=results.get)
    print(
        f"Best update interval: {best_interval} with total rewards: {results[best_interval]}"
    )
    return best_interval


if __name__ == "__main__":
    best_update_interval = find_best_update_interval()
    print(f"The best update interval is {best_update_interval}")
