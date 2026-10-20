"""
Training entry point with residual-policy support.

Place this file in SurgicAI/RL/ and run it instead of RL_training_online.py when
training insert_residual. It remains compatible with non-residual tasks.
"""

import argparse
import copy
import gc
import importlib
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from algorithm_configs_online import get_algorithm_config


gc.collect()
torch.cuda.empty_cache()
Base_directory = os.path.dirname(os.path.abspath(__file__))


def load_expert_data(task_name):
    expert_data_path = os.path.join(Base_directory, "Expert_traj", task_name, "all_episodes_merged.pkl")
    try:
        with open(expert_data_path, "rb") as file:
            print(f"Loaded expert data from {expert_data_path}")
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Expert data file not found: {expert_data_path}")
        return None


def _task_to_module_and_class(task_name):
    module_name = f"{task_name.capitalize()}_env"
    class_name = f"SRC_{task_name.lower()}"
    return module_name, class_name


def setup_environment(args):
    max_episode_steps = 300
    trans_step = 1.0e-3
    angle_step = np.deg2rad(2)
    jaw_step = 0.05
    step_size = np.array(
        [trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step],
        dtype=np.float32,
    )
    threshold = np.array([args.trans_error, np.deg2rad(args.angle_error)], dtype=np.float32)

    module_name, class_name = _task_to_module_and_class(args.task_name)
    module = importlib.import_module(module_name)
    SRC_class = getattr(module, class_name)

    env_kwargs = dict(
        render_mode="human",
        reward_type=args.reward_type,
        max_episode_step=max_episode_steps,
        seed=args.seed,
        step_size=step_size,
        threshold=threshold,
    )

    if args.task_name.lower() == "insert_residual":
        env_kwargs.update(
            base_model_path=args.base_insert_model_path,
            base_algorithm=args.base_insert_algo,
            residual_scale=args.residual_scale,
            enable_residual_jaw=args.enable_residual_jaw,
        )

    env_id = f"{args.task_name}_{args.algorithm}_{args.reward_type}_v0"
    if env_id not in gym.envs.registry:
        gym.envs.register(id=env_id, entry_point=SRC_class, max_episode_steps=max_episode_steps)

    env = gym.make(env_id, **env_kwargs)
    return env, step_size, threshold, max_episode_steps


def _is_bc_algorithm(algorithm_name):
    return "BC" in algorithm_name.upper()


def _iter_transition_containers(data):
    """Yield all transition dicts while preserving the original data structure."""
    if data is None:
        return
    if isinstance(data, dict):
        yield data
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from _iter_transition_containers(item)


def transform_expert_actions_to_residual(expert_data, env):
    """
    Convert full-action expert demonstrations into residual-action demonstrations.

    TD3_BC's BC loss trains actor(obs) directly toward trans["action"]. For a
    residual environment, trans["action"] must therefore be the residual action,
    not the original full insert action.
    """
    if expert_data is None:
        return None

    residual_env = env.unwrapped
    if not hasattr(residual_env, "residual_from_expert_action"):
        return expert_data

    transformed = copy.deepcopy(expert_data)
    converted = 0
    for transition in _iter_transition_containers(transformed):
        if "obs" not in transition or "action" not in transition:
            continue
        transition["action"] = residual_env.residual_from_expert_action(
            transition["action"], obs=transition["obs"]
        )
        converted += 1

    print(f"Converted {converted} expert transitions from full actions to residual actions.")
    return transformed


def create_model(args, env, expert_data):
    algorithm_config = get_algorithm_config(
        args.algorithm,
        env,
        args.task_name,
        args.reward_type,
        args.seed,
        expert_data,
    )
    model_class = algorithm_config["class"]
    model_params = algorithm_config["params"]
    return model_class(**model_params)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a SurgicAI reinforcement-learning agent.")
    parser.add_argument("--algorithm", type=str, required=True, help="Name of the RL algorithm to use")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task/environment")
    parser.add_argument("--reward_type", type=str, choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--trans_error", type=float, required=True)
    parser.add_argument("--angle_error", type=float, required=True)

    # Residual insert options. They are ignored for non-residual tasks.
    parser.add_argument("--base_insert_model_path", type=str, default=os.environ.get("BASE_INSERT_MODEL_PATH"))
    parser.add_argument("--base_insert_algo", type=str, default=os.environ.get("BASE_INSERT_ALGO", "TD3_BC"))
    parser.add_argument("--residual_scale", type=float, default=float(os.environ.get("RESIDUAL_SCALE", 0.1)))
    parser.add_argument("--enable_residual_jaw", action="store_true")

    # Expert data options. For insert_residual, the default expert source is the
    # original insert demonstrations, not Expert_traj/insert_residual/.
    parser.add_argument("--expert_task_name", type=str, default=None)
    parser.add_argument(
        "--no_residual_bc_transform",
        action="store_true",
        help="Do not convert full expert actions into residual expert actions.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    set_random_seed(args.seed)

    env, step_size, threshold, max_episode_steps = setup_environment(args)

    expert_task_name = args.expert_task_name
    if expert_task_name is None:
        expert_task_name = "insert" if args.task_name.lower() == "insert_residual" else args.task_name

    expert_data = load_expert_data(expert_task_name)

    should_transform = (
        args.task_name.lower() == "insert_residual"
        and _is_bc_algorithm(args.algorithm)
        and not args.no_residual_bc_transform
    )
    if should_transform:
        expert_data = transform_expert_actions_to_residual(expert_data, env)

    model = create_model(args, env, expert_data)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(
            Base_directory,
            args.task_name,
            args.algorithm,
            args.reward_type,
            f"seed_{args.seed}",
            "checkpoints",
        ),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    save_path = os.path.join(
        Base_directory,
        args.task_name,
        args.algorithm,
        args.reward_type,
        f"seed_{args.seed}",
        "final_model",
    )
    model.save(save_path)
    print(f"Final model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
