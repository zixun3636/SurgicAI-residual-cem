import argparse
import json
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import (
    load_model,
    clip_theta,
    theta_to_scale_bias,
)


def parse_theta(theta_str):
    values = [float(x.strip()) for x in theta_str.split(",")]
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")

    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)

    parser.add_argument("--eval_seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)

    parser.add_argument("--angle_bias_deg", type=float, required=True)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_insert_model_path", type=str, required=True)
    parser.add_argument("--base_insert_algo", type=str, default="TD3_HER_BC")

    parser.add_argument("--residual_scale", type=float, default=1.0)

    parser.add_argument(
        "--postprocess_mode",
        type=str,
        choices=["global_scale", "rot_scale_bias", "scale6", "scale_bias6"],
        default="rot_scale_bias",
    )

    parser.add_argument(
        "--theta",
        type=str,
        required=True,
        help="Comma-separated theta values",
    )

    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--bias_bound", type=float, default=1.0)

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/suzixun/SurgicAI/RL/evolution_results/residual_trajectories",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load theta
    # ------------------------------------------------------------
    theta = parse_theta(args.theta)

    theta = clip_theta(
        theta,
        args.postprocess_mode,
        args.scale_min,
        args.scale_max,
        args.bias_bound,
    )

    scale_vec, bias_vec = theta_to_scale_bias(
        theta,
        args.postprocess_mode,
    )

    print("\n========== Residual Trajectory Analysis ==========")
    print(f"angle_bias_deg : {args.angle_bias_deg}")
    print(f"theta          : {theta}")
    print(f"scale_vec      : {scale_vec}")
    print(f"bias_vec       : {bias_vec}")
    print(f"episodes       : {args.episodes}")
    print("===================================================\n")

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------
    env, step_size, threshold, max_episode_steps = setup_environment(args)

    residual_model = load_model(
        args.algorithm,
        args.model_path,
        env,
    )

    base_model = load_model(
        args.base_insert_algo,
        args.base_insert_model_path,
        env,
    )

    angle_bias_rad = np.deg2rad(args.angle_bias_deg)
    angle_step = float(env.unwrapped.step_size[3])

    bias_in_action_space = (
        angle_bias_rad / angle_step
        if angle_step != 0
        else 0.0
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------
    for ep in range(args.episodes):

        obs, info = env.reset(
            seed=args.eval_seed + ep
        )

        done = False
        truncated = False
        step = 0
        ep_return = 0.0
        traj_mm = 0.0

        episode_data = {
            "episode": ep,
            "angle_bias_deg": args.angle_bias_deg,
            "steps": [],
            "observations": [],
            "base_actions": [],
            "raw_residual_actions": [],
            "scaled_residual_actions": [],
            "calibrated_residual_actions": [],
            "injected_bias_action": [],
            "final_actions": [],
            "rewards": [],
        }

        while not (done or truncated):

            # Observation may be a Dict observation.
            # Keep the original structure for the policy,
            # and make a safe copy only for logging.
            if isinstance(obs, dict):
                obs_before = {
                    k: np.asarray(v).copy()
                    for k, v in obs.items()
                }
            else:
                obs_before = np.asarray(
                    obs,
                    dtype=np.float32
                ).copy()

            # ----------------------------------------------------
            # Base policy
            # ----------------------------------------------------
            base_action, _ = base_model.predict(
                obs,
                deterministic=True,
            )

            # ----------------------------------------------------
            # Residual policy
            # ----------------------------------------------------
            residual_action, _ = residual_model.predict(
                obs,
                deterministic=True,
            )

            base_action = np.asarray(
                base_action,
                dtype=np.float32,
            ).copy()

            residual_action = np.asarray(
                residual_action,
                dtype=np.float32,
            ).copy()

            # ----------------------------------------------------
            # Residual scale
            # ----------------------------------------------------
            scaled_residual = (
                args.residual_scale
                * residual_action
            )

            # ----------------------------------------------------
            # CEM post-processing
            # ----------------------------------------------------
            calibrated_residual = (
                args.residual_scale
                * scale_vec
                * residual_action
                + bias_vec
            )

            # ----------------------------------------------------
            # Base + calibrated residual
            # ----------------------------------------------------
            action_before_bias = (
                base_action
                + calibrated_residual
            )

            # ----------------------------------------------------
            # Artificial angle bias
            # ----------------------------------------------------
            injected_bias = np.zeros_like(
                action_before_bias,
                dtype=np.float32,
            )

            if args.angle_bias_deg != 0:
                injected_bias[3:6] = bias_in_action_space

            # ----------------------------------------------------
            # Final action
            # ----------------------------------------------------
            final_action = (
                action_before_bias
                + injected_bias
            )

            final_action = np.clip(
                final_action,
                -1.0,
                1.0,
            )

            # ----------------------------------------------------
            # Environment step
            # ----------------------------------------------------
            next_obs, reward, done, truncated, info = env.step(
                final_action
            )

            # ----------------------------------------------------
            # Log
            # ----------------------------------------------------
            episode_data["steps"].append(step)

            # Observation is a Dict observation.
            # We do not need to save the full observation for this
            # mechanism analysis; the policy already receives it.

            episode_data["base_actions"].append(
                base_action
            )

            episode_data["raw_residual_actions"].append(
                residual_action
            )

            episode_data["scaled_residual_actions"].append(
                scaled_residual
            )

            episode_data["calibrated_residual_actions"].append(
                calibrated_residual
            )

            episode_data["injected_bias_action"].append(
                injected_bias
            )

            episode_data["final_actions"].append(
                final_action
            )

            episode_data["rewards"].append(
                float(reward)
            )

            traj_mm += np.linalg.norm(
                final_action[0:3]
                * env.unwrapped.step_size[0:3]
                * 1000
            )

            ep_return += float(reward)

            obs = next_obs
            step += 1

        success = bool(
            info.get("is_success", False)
        )

        episode_data["success"] = success
        episode_data["total_steps"] = step
        episode_data["return"] = ep_return
        episode_data["trajectory_mm"] = traj_mm

        # Convert lists to numpy arrays
        for key in [
            "steps",
            "base_actions",
            "raw_residual_actions",
            "scaled_residual_actions",
            "calibrated_residual_actions",
            "injected_bias_action",
            "final_actions",
            "rewards",
        ]:
            episode_data[key] = np.asarray(
                episode_data[key],
                dtype=np.float32,
            )

        all_episodes.append(episode_data)

        print(
            f"Episode {ep+1:03d} | "
            f"bias={args.angle_bias_deg:+.2f}° | "
            f"success={success} | "
            f"steps={step} | "
            f"traj={traj_mm:.2f} mm"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output_path = (
        save_dir
        / f"bias_{args.angle_bias_deg:+.2f}.npz"
    )

    np.savez_compressed(
        output_path,
        episodes=np.array(
            all_episodes,
            dtype=object,
        ),
    )

    summary = {
        "angle_bias_deg": args.angle_bias_deg,
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "theta": theta.tolist(),
        "scale_vec": scale_vec.tolist(),
        "bias_vec": bias_vec.tolist(),
        "residual_scale": args.residual_scale,
        "output_path": str(output_path),
    }

    summary_path = (
        save_dir
        / f"bias_{args.angle_bias_deg:+.2f}_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\n========== Saved ==========")
    print(output_path)
    print(summary_path)
    print("===========================\n")

    env.close()


if __name__ == "__main__":
    main()
