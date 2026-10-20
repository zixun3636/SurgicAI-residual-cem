import argparse
import json
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import (
    load_model,
    theta_to_scale_bias,
)


def parse_vec3(text):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError(
            "--angle_bias_vec must contain exactly 3 values: roll,pitch,yaw"
        )
    return np.asarray(values, dtype=np.float32)


def parse_theta(text):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 6:
        raise ValueError(
            "--theta must contain 6 values: "
            "s_roll,s_pitch,s_yaw,b_roll,b_pitch,b_yaw"
        )
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")

    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)

    parser.add_argument("--eval_seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)

    parser.add_argument("--action_noise_std", type=float, default=0.0)

    parser.add_argument(
        "--angle_bias_vec",
        type=str,
        required=True,
        help='Roll,Pitch,Yaw bias in degrees, e.g. "2,-2,2"',
    )

    parser.add_argument(
        "--theta",
        type=str,
        required=True,
        help=(
            "s_roll,s_pitch,s_yaw,"
            "b_roll,b_pitch,b_yaw"
        ),
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_insert_model_path", type=str, required=True)
    parser.add_argument(
        "--base_insert_algo",
        type=str,
        default="TD3_HER_BC",
    )

    parser.add_argument("--residual_scale", type=float, default=1.0)

    parser.add_argument(
        "--postprocess_mode",
        type=str,
        default="rot_scale_bias",
        choices=["rot_scale_bias"],
    )

    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------
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

    # --------------------------------------------------
    # CEM theta
    # --------------------------------------------------
    theta = parse_theta(args.theta)

    scale_vec, bias_vec = theta_to_scale_bias(
        theta,
        args.postprocess_mode,
    )

    # --------------------------------------------------
    # Mixed-axis rotational bias
    # [roll, pitch, yaw] in degree-equivalent commands
    # --------------------------------------------------
    angle_bias_vec_deg = parse_vec3(args.angle_bias_vec)

    rotation_step = np.asarray(
        env.unwrapped.step_size[3:6],
        dtype=np.float32,
    )

    bias_in_action_space = (
        np.deg2rad(angle_bias_vec_deg) / rotation_step
    ).astype(np.float32)

    print()
    print("========== Mixed Bias Evaluation ==========")
    print(f"Bias [roll,pitch,yaw] deg : {angle_bias_vec_deg}")
    print(f"Bias in action space      : {bias_in_action_space}")
    print(f"Theta                     : {theta}")
    print(f"Scale vec                 : {scale_vec}")
    print(f"Calibration bias vec      : {bias_vec}")
    print(f"Residual scale            : {args.residual_scale}")
    print(f"Eval seed                 : {args.eval_seed}")
    print(f"Episodes                  : {args.episodes}")
    print("===========================================")
    print()

    rng = np.random.default_rng(args.eval_seed)

    successes = []
    steps_list = []
    traj_list = []
    returns = []

    successful_steps = []
    successful_traj = []

    rot_sat_counts = np.zeros(3, dtype=np.int64)
    total_control_steps = 0

    for ep in range(args.episodes):

        obs, info = env.reset(seed=args.eval_seed + ep)

        done = False
        truncated = False

        steps = 0
        traj_mm = 0.0
        ep_return = 0.0

        while not (done or truncated):

            base_action, _ = base_model.predict(
                obs,
                deterministic=True,
            )

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

            # Base + calibrated residual
            action = (
                base_action
                + args.residual_scale
                * scale_vec
                * residual_action
                + bias_vec
            )

            # Mixed-axis disturbance:
            # roll +2°, pitch -2°, yaw +2° for "2,-2,2"
            action[3:6] += bias_in_action_space

            if args.action_noise_std > 0:
                action += rng.normal(
                    loc=0.0,
                    scale=args.action_noise_std,
                    size=action.shape,
                ).astype(np.float32)

            # Final executed normalized action
            action = np.clip(action, -1.0, 1.0)

            # Saturation statistics
            rot_sat_counts += (
                np.abs(action[3:6]) >= 0.999999
            ).astype(np.int64)

            total_control_steps += 1

            obs, reward, done, truncated, info = env.step(action)

            steps += 1
            ep_return += float(reward)

            traj_mm += np.linalg.norm(
                action[0:3]
                * env.unwrapped.step_size[0:3]
                * 1000.0
            )

        success = bool(info.get("is_success", False))

        successes.append(success)
        steps_list.append(steps)
        traj_list.append(traj_mm)
        returns.append(ep_return)

        if success:
            successful_steps.append(steps)
            successful_traj.append(traj_mm)

        print(
            f"Episode {ep+1:03d} | "
            f"success={success} | "
            f"steps={steps} | "
            f"traj={traj_mm:.2f} mm"
        )

    success_rate = float(np.mean(successes))

    mean_steps_all = float(np.mean(steps_list))
    mean_traj_all = float(np.mean(traj_list))
    mean_return = float(np.mean(returns))

    if successful_steps:
        mean_steps_success = float(np.mean(successful_steps))
        mean_traj_success = float(np.mean(successful_traj))
    else:
        mean_steps_success = None
        mean_traj_success = None

    if total_control_steps > 0:
        saturation_rate = (
            rot_sat_counts / total_control_steps * 100.0
        )
    else:
        saturation_rate = np.zeros(3)

    print()
    print("=============== RESULT =================")
    print(
        f"Mixed bias [deg]       : "
        f"{angle_bias_vec_deg.tolist()}"
    )
    print(
        f"Success Rate           : "
        f"{success_rate * 100:.2f}%"
    )
    print(
        f"Success Episodes       : "
        f"{sum(successes)}/{args.episodes}"
    )
    print(
        f"Mean Steps (all)       : "
        f"{mean_steps_all:.2f}"
    )

    if mean_steps_success is not None:
        print(
            f"Mean Steps (success)   : "
            f"{mean_steps_success:.2f}"
        )
        print(
            f"Mean Traj (success)    : "
            f"{mean_traj_success:.2f} mm"
        )
    else:
        print("Mean Steps (success)   : N/A")
        print("Mean Traj (success)    : N/A")

    print(
        "Saturation R/P/Y [%]  : "
        f"{saturation_rate[0]:.2f}, "
        f"{saturation_rate[1]:.2f}, "
        f"{saturation_rate[2]:.2f}"
    )

    print(f"Mean Return            : {mean_return:.2f}")
    print(f"Theta                  : {theta.tolist()}")
    print("========================================")
    print()

    result = {
        "method": "mixed_axis_bias_evaluation",
        "angle_bias_vec_deg": angle_bias_vec_deg.tolist(),
        "bias_in_action_space": bias_in_action_space.tolist(),
        "theta": theta.tolist(),
        "scale_vec": scale_vec.tolist(),
        "calibration_bias_vec": bias_vec.tolist(),
        "residual_scale": float(args.residual_scale),
        "eval_seed": int(args.eval_seed),
        "episodes": int(args.episodes),
        "success_count": int(sum(successes)),
        "success_rate": success_rate,
        "mean_steps_all": mean_steps_all,
        "mean_steps_success": mean_steps_success,
        "mean_traj_all_mm": mean_traj_all,
        "mean_traj_success_mm": mean_traj_success,
        "mean_return": mean_return,
        "rot_saturation_percent": {
            "roll": float(saturation_rate[0]),
            "pitch": float(saturation_rate[1]),
            "yaw": float(saturation_rate[2]),
        },
        "model_path": args.model_path,
        "base_insert_model_path": args.base_insert_model_path,
    }

    if args.save_json is not None:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved result to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
