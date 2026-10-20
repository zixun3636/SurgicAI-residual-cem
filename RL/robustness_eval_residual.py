import argparse
from pathlib import Path
import numpy as np

from RL_algo.td3_BC import TD3_BC

try:
    from RL_algo.td3 import TD3
except Exception:
    from stable_baselines3 import TD3

from Model_evaluation import setup_environment


def load_model(algorithm, model_path, env):
    algorithm = algorithm.upper()
    model_path = Path(model_path)

    print(f"Loading model: {model_path}")
    print(f"Exists: {model_path.exists()}")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if algorithm in ["TD3_BC", "TD3_HER_BC"]:
        return TD3_BC.load(str(model_path), env=env)

    if algorithm in ["TD3", "TD3_HER"]:
        return TD3.load(str(model_path), env=env)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_mode", type=str, choices=["base", "residual"], default="base")

    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")

    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)

    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)

    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--angle_bias_deg", type=float, default=0.0)

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--base_insert_model_path", type=str, default=None)
    parser.add_argument("--base_insert_algo", type=str, default="TD3_HER_BC")
    parser.add_argument("--residual_scale", type=float, default=0.1)

    args = parser.parse_args()

    if args.eval_mode == "residual" and args.base_insert_model_path is None:
        raise ValueError("--base_insert_model_path is required when --eval_mode residual")

    # 注意：
    # residual 評估也使用原本 Insert env。
    # 這樣可以公平比較：
    # base_action
    # vs
    # base_action + residual_scale * residual_action
    env, step_size, threshold, max_episode_steps = setup_environment(args)

    rng = np.random.default_rng(args.eval_seed)

    model = load_model(args.algorithm, args.model_path, env)

    base_model = None
    if args.eval_mode == "residual":
        base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    successes = []
    traj_lengths = []
    time_costs = []

    angle_bias_rad = np.deg2rad(args.angle_bias_deg)
    angle_step = float(env.unwrapped.step_size[3])
    bias_in_action_space = angle_bias_rad / angle_step if angle_step != 0 else 0.0

    print("\n=== Robustness Eval Config ===")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Task name: {args.task_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Action noise std: {args.action_noise_std}")
    print(f"Angle bias deg: {args.angle_bias_deg}")
    print(f"Bias in action space: {bias_in_action_space}")
    print(f"Residual scale: {args.residual_scale}")
    print(f"Episodes: {args.episodes}")
    print("==============================\n")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.eval_seed + ep)

        done = False
        truncated = False
        steps = 0
        traj_mm = 0.0

        while not (done or truncated):
            if args.eval_mode == "base":
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).copy()

            else:
                base_action, _ = base_model.predict(obs, deterministic=True)
                residual_action, _ = model.predict(obs, deterministic=True)

                base_action = np.asarray(base_action, dtype=np.float32)
                residual_action = np.asarray(residual_action, dtype=np.float32)

                action = base_action + args.residual_scale * residual_action
                action = np.asarray(action, dtype=np.float32).copy()

            if args.angle_bias_deg != 0:
                action[3:6] += bias_in_action_space

            if args.action_noise_std > 0:
                action += rng.normal(
                    loc=0.0,
                    scale=args.action_noise_std,
                    size=action.shape
                ).astype(np.float32)

            action = np.clip(action, -1.0, 1.0)

            obs, reward, done, truncated, info = env.step(action)

            steps += 1
            traj_mm += np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000)

        success = bool(info.get("is_success", False))
        successes.append(success)

        if success:
            traj_lengths.append(traj_mm)
            time_costs.append(steps)

        print(
            f"Episode {ep+1:03d} | "
            f"success={success} | "
            f"steps={steps} | "
            f"traj_mm={traj_mm:.2f}"
        )

    success_rate = 100.0 * np.mean(successes)
    avg_traj = float(np.mean(traj_lengths)) if len(traj_lengths) > 0 else float("nan")
    avg_steps = float(np.mean(time_costs)) if len(time_costs) > 0 else float("nan")

    print("\n=== Robustness Result ===")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Action noise std: {args.action_noise_std}")
    print(f"Angle bias deg: {args.angle_bias_deg}")
    print(f"Episodes: {args.episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Trajectory Length: {avg_traj:.2f} mm")
    print(f"Average Time Cost: {avg_steps:.2f} steps")

    env.close()


if __name__ == "__main__":
    main()
