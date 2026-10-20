import argparse
import copy
import json
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment

# Reuse functions from your existing CEM script
from evolve_residual_postprocess import (
    load_model,
    init_theta,
    clip_theta,
    theta_to_scale_bias,
    rollout_score,
)


def parse_bias_list(bias_list_str):
    """
    Example:
        "0.5,1,1.5,2" -> [0.5, 1.0, 1.5, 2.0]
    """
    return [float(x.strip()) for x in bias_list_str.split(",") if x.strip() != ""]


def make_std(args, dim):
    std = np.ones(dim, dtype=np.float32) * args.init_std

    if args.postprocess_mode == "rot_scale_bias":
        std[0:3] = args.scale_init_std
        std[3:6] = args.bias_init_std

    elif args.postprocess_mode == "scale_bias6":
        std[0:6] = args.scale_init_std
        std[6:12] = args.bias_init_std

    elif args.postprocess_mode in ["global_scale", "scale6"]:
        std[:] = args.scale_init_std

    return std.astype(np.float32)


def evaluate_theta_multibias(
    env,
    base_model,
    residual_model,
    theta,
    args,
    eval_seed,
    episodes_per_bias,
    verbose=False,
):
    """
    Evaluate one candidate theta on multiple angle biases.
    Score is designed for bias-range robustness.

    Main objectives:
        1. High average success rate over all bias values.
        2. High minimum success rate, to avoid only performing well on one bias.
        3. Lower steps and trajectory length as tie-breakers.
    """
    bias_values = parse_bias_list(args.bias_list)

    per_bias_results = {}
    success_rates = []
    mean_steps_list = []
    mean_traj_list = []
    mean_return_list = []

    for idx, bias_deg in enumerate(bias_values):
        local_args = copy.copy(args)
        local_args.angle_bias_deg = float(bias_deg)

        # Use a deterministic but different seed offset for each bias.
        # All candidates see the same seed set for fairness.
        bias_eval_seed = eval_seed + idx * 1000

        result = rollout_score(
            env=env,
            base_model=base_model,
            residual_model=residual_model,
            theta=theta,
            args=local_args,
            eval_seed=bias_eval_seed,
            episodes=episodes_per_bias,
            verbose=False,
        )

        per_bias_results[str(bias_deg)] = result

        success_rates.append(result["success_rate"])
        mean_steps_list.append(result["mean_steps"])
        mean_traj_list.append(result["mean_traj_mm"])
        mean_return_list.append(result["mean_return"])

        if verbose:
            print(
                f"  bias={bias_deg:>4} | "
                f"success={result['success_rate'] * 100:.2f}% | "
                f"steps={result['mean_steps']:.2f} | "
                f"traj={result['mean_traj_mm']:.2f}"
            )

    mean_success = float(np.mean(success_rates))
    min_success = float(np.min(success_rates))
    mean_steps = float(np.mean(mean_steps_list))
    mean_traj = float(np.mean(mean_traj_list))
    mean_return = float(np.mean(mean_return_list))

    # Multi-bias objective:
    # - mean_success rewards overall performance across the bias range.
    # - min_success prevents one bias condition from collapsing.
    # - steps/traj are only secondary tie-breakers.
    score = (
        mean_success * 1000.0
        + min_success * 500.0
        - mean_steps * 0.5
        - mean_traj * 0.01
    )

    theta = clip_theta(
        theta,
        args.postprocess_mode,
        args.scale_min,
        args.scale_max,
        args.bias_bound,
    )

    scale_vec, bias_vec = theta_to_scale_bias(theta, args.postprocess_mode)

    summary = {
        "score": float(score),
        "mean_success_rate": mean_success,
        "min_success_rate": min_success,
        "mean_steps": mean_steps,
        "mean_traj_mm": mean_traj,
        "mean_return": mean_return,
        "theta": theta.tolist(),
        "scale_vec": scale_vec.tolist(),
        "bias_vec": bias_vec.tolist(),
        "bias_values": bias_values,
        "per_bias_results": per_bias_results,
    }

    return summary


def optimize_multibias(env, base_model, residual_model, args):
    rng = np.random.default_rng(args.evo_seed)

    mean = init_theta(args.postprocess_mode)
    dim = len(mean)

    std = make_std(args, dim)

    best = None
    history = []

    print("\n========== Multi-Bias CEM Config ==========")
    print(f"postprocess_mode       : {args.postprocess_mode}")
    print(f"bias_list              : {args.bias_list}")
    print(f"generations            : {args.generations}")
    print(f"population             : {args.population}")
    print(f"elite_frac             : {args.elite_frac}")
    print(f"episodes_per_bias      : {args.episodes_per_bias}")
    print(f"residual_scale         : {args.residual_scale}")
    print(f"action_noise_std       : {args.action_noise_std}")
    print(f"initial mean           : {mean}")
    print(f"initial std            : {std}")
    print("==========================================\n")

    for gen in range(args.generations):
        print(f"\n========== Generation {gen + 1}/{args.generations} ==========")

        candidates = []

        # Keep current mean as a candidate, so search does not forget the current best center.
        candidates.append(mean.copy())

        for _ in range(args.population - 1):
            theta = mean + std * rng.standard_normal(dim).astype(np.float32)
            theta = clip_theta(
                theta,
                args.postprocess_mode,
                args.scale_min,
                args.scale_max,
                args.bias_bound,
            )
            candidates.append(theta)

        results = []

        for i, theta in enumerate(candidates):
            result = evaluate_theta_multibias(
                env=env,
                base_model=base_model,
                residual_model=residual_model,
                theta=theta,
                args=args,
                eval_seed=args.eval_seed,
                episodes_per_bias=args.episodes_per_bias,
                verbose=False,
            )

            results.append(result)

            per_bias_success = {
                k: round(v["success_rate"] * 100.0, 2)
                for k, v in result["per_bias_results"].items()
            }

            print(
                f"cand={i:02d} | "
                f"score={result['score']:.3f} | "
                f"mean_success={result['mean_success_rate'] * 100:.2f}% | "
                f"min_success={result['min_success_rate'] * 100:.2f}% | "
                f"steps={result['mean_steps']:.2f} | "
                f"traj={result['mean_traj_mm']:.2f} | "
                f"per_bias={per_bias_success} | "
                f"theta={np.array(result['theta'])}"
            )

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        if best is None or results[0]["score"] > best["score"]:
            best = results[0]

        elite_count = max(1, int(args.population * args.elite_frac))
        elites = results[:elite_count]

        elite_thetas = np.asarray([e["theta"] for e in elites], dtype=np.float32)

        mean = elite_thetas.mean(axis=0)
        std = elite_thetas.std(axis=0) + args.min_std

        mean = clip_theta(
            mean,
            args.postprocess_mode,
            args.scale_min,
            args.scale_max,
            args.bias_bound,
        )

        history.append(
            {
                "generation": gen + 1,
                "best_generation": results[0],
                "best_overall": best,
                "new_mean": mean.tolist(),
                "new_std": std.tolist(),
            }
        )

        print("\n--- Generation Summary ---")
        print(f"Best gen mean success : {results[0]['mean_success_rate'] * 100:.2f}%")
        print(f"Best gen min success  : {results[0]['min_success_rate'] * 100:.2f}%")
        print(f"Best gen score        : {results[0]['score']:.3f}")
        print(f"Best overall mean succ: {best['mean_success_rate'] * 100:.2f}%")
        print(f"Best overall min succ : {best['min_success_rate'] * 100:.2f}%")
        print(f"Best overall theta    : {np.array(best['theta'])}")
        print(f"New mean              : {mean}")
        print(f"New std               : {std}")

    output = {
        "method": "multi_bias_cem_style_postprocess",
        "postprocess_mode": args.postprocess_mode,
        "bias_list": parse_bias_list(args.bias_list),
        "residual_scale": args.residual_scale,
        "action_noise_std": args.action_noise_std,
        "algorithm": args.algorithm,
        "residual_model_path": args.model_path,
        "base_insert_model_path": args.base_insert_model_path,
        "base_insert_algo": args.base_insert_algo,
        "best": best,
        "history": history,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("\n========== Multi-Bias CEM Finished ==========")
    print(f"Best mean success rate: {best['mean_success_rate'] * 100:.2f}%")
    print(f"Best min success rate : {best['min_success_rate'] * 100:.2f}%")
    print(f"Best score            : {best['score']:.3f}")
    print(f"Best theta            : {np.array(best['theta'])}")
    print(f"Scale vec             : {np.array(best['scale_vec'])}")
    print(f"Bias vec              : {np.array(best['bias_vec'])}")
    print(f"Saved to              : {output_path}")
    print("============================================\n")

    return output


def eval_multibias(env, base_model, residual_model, args):
    params_path = Path(args.params_path)
    data = json.loads(params_path.read_text())

    theta = np.asarray(data["best"]["theta"], dtype=np.float32)

    print("\n========== Loaded Multi-Bias CEM Params ==========")
    print(f"params_path      : {params_path}")
    print(f"postprocess_mode : {args.postprocess_mode}")
    print(f"bias_list        : {args.bias_list}")
    print(f"theta            : {theta}")
    print("=================================================\n")

    result = evaluate_theta_multibias(
        env=env,
        base_model=base_model,
        residual_model=residual_model,
        theta=theta,
        args=args,
        eval_seed=args.eval_seed,
        episodes_per_bias=args.episodes,
        verbose=True,
    )

    print("\n========== Multi-Bias Evaluation Summary ==========")
    print(f"Bias list             : {args.bias_list}")
    print(f"Episodes per bias     : {args.episodes}")
    print(f"Action noise std      : {args.action_noise_std}")
    print(f"Mean Success Rate     : {result['mean_success_rate'] * 100:.2f}%")
    print(f"Min Success Rate      : {result['min_success_rate'] * 100:.2f}%")
    print(f"Mean Steps            : {result['mean_steps']:.2f}")
    print(f"Mean Trajectory       : {result['mean_traj_mm']:.2f} mm")
    print(f"Mean Return           : {result['mean_return']:.2f}")
    print(f"Theta                 : {np.array(result['theta'])}")
    print(f"Scale vec             : {np.array(result['scale_vec'])}")
    print(f"Bias vec              : {np.array(result['bias_vec'])}")
    print("\nPer-bias results:")

    for bias_deg, r in result["per_bias_results"].items():
        print(
            f"  bias={bias_deg:>4} | "
            f"success={r['success_rate'] * 100:.2f}% | "
            f"steps={r['mean_steps']:.2f} | "
            f"traj={r['mean_traj_mm']:.2f} mm | "
            f"return={r['mean_return']:.2f}"
        )

    print("==================================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["optimize", "eval"], default="optimize")

    parser.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    parser.add_argument("--task_name", type=str, default="Insert")
    parser.add_argument("--reward_type", type=str, default="sparse")

    parser.add_argument("--trans_error", type=float, default=0.5)
    parser.add_argument("--angle_error", type=float, default=10)

    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--evo_seed", type=int, default=0)

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episodes_per_bias", type=int, default=20)

    parser.add_argument("--bias_list", type=str, default="0.5,1,1.5,2")
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

    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite_frac", type=float, default=0.3)

    parser.add_argument("--init_std", type=float, default=0.3)
    parser.add_argument("--scale_init_std", type=float, default=0.25)
    parser.add_argument("--bias_init_std", type=float, default=0.25)
    parser.add_argument("--min_std", type=float, default=0.02)

    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--bias_bound", type=float, default=1.0)

    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/suzixun/SurgicAI/RL/evolution_results/multibias_rot_scale_bias.json",
    )

    parser.add_argument("--params_path", type=str, default=None)

    args = parser.parse_args()

    # setup_environment expects angle_bias_deg to exist in some evaluation paths.
    # Multi-bias evaluation overwrites it internally.
    args.angle_bias_deg = 0.0

    env, step_size, threshold, max_episode_steps = setup_environment(args)

    residual_model = load_model(args.algorithm, args.model_path, env)
    base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    if args.mode == "optimize":
        optimize_multibias(env, base_model, residual_model, args)

    elif args.mode == "eval":
        if args.params_path is None:
            raise ValueError("--params_path is required when --mode eval")
        eval_multibias(env, base_model, residual_model, args)

    env.close()


if __name__ == "__main__":
    main()
