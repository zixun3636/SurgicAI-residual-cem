import argparse
import copy
import json
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import load_model, rollout_score


def parse_bias_list(bias_list_str):
    return [float(x.strip()) for x in bias_list_str.split(",") if x.strip() != ""]


def load_theta(params_path):
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    data = json.loads(params_path.read_text())

    if "best" not in data or "theta" not in data["best"]:
        raise KeyError(f"Cannot find data['best']['theta'] in {params_path}")

    return np.asarray(data["best"]["theta"], dtype=np.float32), data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")

    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)

    parser.add_argument("--eval_seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)

    parser.add_argument(
        "--bias_list",
        type=str,
        default="-2,-1.5,-1,-0.5,0.5,1,1.5,2",
    )

    parser.add_argument("--action_noise_std", type=float, default=0.0)

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

    parser.add_argument("--positive_params_path", type=str, required=True)
    parser.add_argument("--negative_params_path", type=str, required=True)

    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--bias_bound", type=float, default=1.0)

    parser.add_argument(
        "--save_json",
        type=str,
        default="/home/suzixun/SurgicAI/RL/evolution_results/direction_conditioned_eval.json",
    )

    args = parser.parse_args()

    theta_pos, data_pos = load_theta(args.positive_params_path)
    theta_neg, data_neg = load_theta(args.negative_params_path)

    print("\n========== Direction-Conditioned CEM Params ==========")
    print(f"positive_params_path : {args.positive_params_path}")
    print(f"negative_params_path : {args.negative_params_path}")
    print(f"theta_pos            : {theta_pos}")
    print(f"theta_neg            : {theta_neg}")
    print(f"bias_list            : {args.bias_list}")
    print("=====================================================\n")

    # setup_environment may expect angle_bias_deg to exist.
    args.angle_bias_deg = 0.0

    env, step_size, threshold, max_episode_steps = setup_environment(args)

    residual_model = load_model(args.algorithm, args.model_path, env)
    base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    bias_values = parse_bias_list(args.bias_list)

    per_bias_results = {}
    success_rates = []
    mean_steps_list = []
    mean_traj_list = []
    mean_return_list = []

    for idx, bias_deg in enumerate(bias_values):
        local_args = copy.copy(args)
        local_args.angle_bias_deg = float(bias_deg)

        if bias_deg < 0:
            theta = theta_neg
            theta_name = "theta_neg"
        else:
            theta = theta_pos
            theta_name = "theta_pos"

        # Different bias values use different deterministic seed offsets.
        bias_eval_seed = args.eval_seed + idx * 1000

        result = rollout_score(
            env=env,
            base_model=base_model,
            residual_model=residual_model,
            theta=theta,
            args=local_args,
            eval_seed=bias_eval_seed,
            episodes=args.episodes,
            verbose=False,
        )

        per_bias_results[str(bias_deg)] = {
            "theta_used": theta_name,
            "success_rate": result["success_rate"],
            "mean_steps": result["mean_steps"],
            "mean_traj_mm": result["mean_traj_mm"],
            "mean_return": result["mean_return"],
            "theta": result["theta"],
            "scale_vec": result["scale_vec"],
            "bias_vec": result["bias_vec"],
        }

        success_rates.append(result["success_rate"])
        mean_steps_list.append(result["mean_steps"])
        mean_traj_list.append(result["mean_traj_mm"])
        mean_return_list.append(result["mean_return"])

        print(
            f"bias={bias_deg:>5} | "
            f"theta={theta_name:>9} | "
            f"success={result['success_rate'] * 100:.2f}% | "
            f"steps={result['mean_steps']:.2f} | "
            f"traj={result['mean_traj_mm']:.2f} mm | "
            f"return={result['mean_return']:.2f}"
        )

    mean_success = float(np.mean(success_rates))
    min_success = float(np.min(success_rates))
    mean_steps = float(np.mean(mean_steps_list))
    mean_traj = float(np.mean(mean_traj_list))
    mean_return = float(np.mean(mean_return_list))

    output = {
        "method": "direction_conditioned_cem",
        "rule": "bias < 0 uses theta_neg; bias >= 0 uses theta_pos",
        "bias_values": bias_values,
        "episodes_per_bias": args.episodes,
        "eval_seed": args.eval_seed,
        "mean_success_rate": mean_success,
        "min_success_rate": min_success,
        "mean_steps": mean_steps,
        "mean_traj_mm": mean_traj,
        "mean_return": mean_return,
        "theta_pos": theta_pos.tolist(),
        "theta_neg": theta_neg.tolist(),
        "positive_params_path": args.positive_params_path,
        "negative_params_path": args.negative_params_path,
        "per_bias_results": per_bias_results,
    }

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(output, indent=2))

    print("\n========== Direction-Conditioned Evaluation Summary ==========")
    print(f"Bias list         : {args.bias_list}")
    print(f"Episodes per bias : {args.episodes}")
    print(f"Mean Success Rate : {mean_success * 100:.2f}%")
    print(f"Min Success Rate  : {min_success * 100:.2f}%")
    print(f"Mean Steps        : {mean_steps:.2f}")
    print(f"Mean Trajectory   : {mean_traj:.2f} mm")
    print(f"Mean Return       : {mean_return:.2f}")
    print(f"Saved to          : {save_path}")
    print("==============================================================\n")

    env.close()


if __name__ == "__main__":
    main()
