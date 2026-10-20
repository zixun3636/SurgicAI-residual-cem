import argparse
from pathlib import Path
import numpy as np

from stable_baselines3 import TD3
from RL_algo.td3_BC import TD3_BC
from Model_evaluation import setup_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")
    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--angle_bias_deg", type=float, default=0.0)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    env, step_size, threshold, max_episode_steps = setup_environment(args)

    model_path = Path(args.model_path)
    print(f"Loading model: {model_path}")
    print(f"Exists: {model_path.exists()}")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    try:
        model = TD3_BC.load(str(model_path), env=env)
    except Exception as e:
        print(f"TD3_BC.load failed: {e}")
        print("Falling back to TD3.load...")
        model = TD3.load(str(model_path), env=env)

    successes = []
    traj_lengths = []
    time_costs = []

    # 將角度偏置轉成 action space 裡的單位比例
    # action[3:6] 乘上 env.unwrapped.step_size[3:6] 才是實際角度步長
    # 所以要塞一個固定角度偏置，需先除回去
    angle_bias_rad = np.deg2rad(args.angle_bias_deg)
    angle_step = float(env.unwrapped.step_size[3])  # 3/4/5 都一樣
    bias_in_action_space = angle_bias_rad / angle_step if angle_step != 0 else 0.0

    print(f"Angle bias (deg): {args.angle_bias_deg}")
    print(f"Angle step (rad): {angle_step}")
    print(f"Bias in action space: {bias_in_action_space}")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        traj_mm = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).copy()

            if args.angle_bias_deg != 0:
                # 對 roll/pitch/yaw 都加同樣偏置
                action[3:6] += bias_in_action_space
                action = np.clip(action, -1.0, 1.0)

            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            traj_mm += np.linalg.norm(np.asarray(action)[0:3] * env.unwrapped.step_size[0:3] * 1000)

        success = bool(info.get("is_success", False))
        successes.append(success)

        if success:
            traj_lengths.append(traj_mm)
            time_costs.append(steps)

        print(f"Episode {ep+1:02d} | success={success} | steps={steps} | traj_mm={traj_mm:.2f}")

    success_rate = 100.0 * np.mean(successes)
    avg_traj = float(np.mean(traj_lengths)) if len(traj_lengths) > 0 else float('nan')
    avg_steps = float(np.mean(time_costs)) if len(time_costs) > 0 else float('nan')

    print("\n=== Angle Bias Robustness Result ===")
    print(f"Angle bias (deg): {args.angle_bias_deg}")
    print(f"Episodes: {args.episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Trajectory Length: {avg_traj:.2f} mm")
    print(f"Average Time Cost: {avg_steps:.2f} steps")

    env.close()


if __name__ == "__main__":
    main()
