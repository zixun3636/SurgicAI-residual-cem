import argparse
import json
from pathlib import Path
import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import load_model, rollout_score


def parse_bias_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def init_theta(mode):
    if mode == "global_scale":
        return np.array([1.0], dtype=np.float32)

    if mode == "rot_scale_bias":
        # [roll_scale, pitch_scale, yaw_scale,
        #  roll_bias,  pitch_bias,  yaw_bias]
        return np.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    if mode == "scale6":
        return np.ones(6, dtype=np.float32)

    if mode == "scale_bias6":
        return np.concatenate(
            [
                np.ones(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
            ]
        )

    raise ValueError(f"Unknown postprocess_mode: {mode}")


def clip_theta(theta, mode, scale_min, scale_max, bias_bound):
    theta = np.asarray(theta, dtype=np.float32).copy()

    if mode == "global_scale":
        theta[:] = np.clip(theta, scale_min, scale_max)

    elif mode == "rot_scale_bias":
        theta[0:3] = np.clip(theta[0:3], scale_min, scale_max)
        theta[3:6] = np.clip(theta[3:6], -bias_bound, bias_bound)

    elif mode == "scale6":
        theta[:] = np.clip(theta, scale_min, scale_max)

    elif mode == "scale_bias6":
        theta[0:6] = np.clip(theta[0:6], scale_min, scale_max)
        theta[6:12] = np.clip(theta[6:12], -bias_bound, bias_bound)

    return theta


def optimize_one_bias(
    env,
    base_model,
    residual_model,
    args,
    bias_deg,
    seed,
):
    rng = np.random.default_rng(seed)

    mean = init_theta(args.postprocess_mode)
    dim = len(mean)

    std = np.ones(dim, dtype=np.float32) * args.init_std

    if args.postprocess_mode == "rot_scale_bias":
        std[0:3] = args.scale_init_std
        std[3:6] = args.bias_init_std

    elif args.postprocess_mode == "scale_bias6":
        std[0:6] = args.scale_init_std
        std[6:12] = args.bias_init_std

    else:
        std[:] = args.scale_init_std

    local_args = args
    local_args.angle_bias_deg = float(bias_deg)

    best = None
    history = []

    print()
    print("=" * 70)
    print(f"Magnitude-conditioned CEM")
    print(f"Bias = {bias_deg:+.2f} deg")
    print("=" * 70)

    for gen in range(args.generations):

        candidates = [mean.copy()]

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

        print()
        print(
            f"Generation {gen + 1}/{args.generations} "
            f"| bias={bias_deg:+.2f}°"
        )

        for i, theta in enumerate(candidates):

            candidate_seed = (
                seed
                + gen * 10000
                + i * 100
            )

            result = rollout_score(
                env=env,
                base_model=base_model,
                residual_model=residual_model,
                theta=theta,
                args=local_args,
                eval_seed=candidate_seed,
                episodes=args.episodes_per_candidate,
                verbose=False,
            )

            results.append(result)

            print(
                f"candidate={i:02d} | "
                f"success={result['success_rate'] * 100:.2f}% | "
                f"steps={result['mean_steps']:.2f} | "
                f"traj={result['mean_traj_mm']:.2f} mm | "
                f"return={result['mean_return']:.2f}"
            )

        # 以 success rate 為主要目標
        results.sort(
            key=lambda x: (
                x["success_rate"],
                -x["mean_steps"],
            ),
            reverse=True,
        )

        generation_best = results[0]

        if (
            best is None
            or generation_best["success_rate"] > best["success_rate"]
            or (
                generation_best["success_rate"]
                == best["success_rate"]
                and generation_best["mean_steps"]
                < best["mean_steps"]
            )
        ):
            best = generation_best

        elite_count = max(
            1,
            int(args.population * args.elite_frac),
        )

        elite_thetas = np.asarray(
            [
                x["theta"]
                for x in results[:elite_count]
            ],
            dtype=np.float32,
        )

        mean = elite_thetas.mean(axis=0)

        std = (
            elite_thetas.std(axis=0)
            + args.min_std
        )

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
                "best_generation": generation_best,
                "best_overall": best,
                "new_mean": mean.tolist(),
                "new_std": std.tolist(),
            }
        )

        print()
        print(
            f"Best generation success : "
            f"{generation_best['success_rate'] * 100:.2f}%"
        )

        print(
            f"Best overall success    : "
            f"{best['success_rate'] * 100:.2f}%"
        )

        print(f"New mean : {mean}")
        print(f"New std  : {std}")

    return {
        "bias_deg": float(bias_deg),
        "best": best,
        "history": history,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        default="TD3_HER_BC",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="Insert",
    )

    parser.add_argument(
        "--reward_type",
        type=str,
        default="sparse",
    )

    parser.add_argument(
        "--trans_error",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--angle_error",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--evo_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--episodes_per_candidate",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--bias_list",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base_insert_model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base_insert_algo",
        type=str,
        default="TD3_HER_BC",
    )

    parser.add_argument(
        "--postprocess_mode",
        type=str,
        choices=[
            "global_scale",
            "rot_scale_bias",
            "scale6",
            "scale_bias6",
        ],
        default="rot_scale_bias",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--population",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--elite_frac",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--init_std",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--scale_init_std",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--bias_init_std",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--min_std",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--scale_max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--bias_bound",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=(
            "/home/suzixun/SurgicAI/RL/"
            "evolution_results/"
            "magnitude_conditioned_cem.json"
        ),
    )

    args = parser.parse_args()

    args.angle_bias_deg = 0.0

    env, _, _, _ = setup_environment(args)

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

    bias_values = parse_bias_list(args.bias_list)

    results = {}

    print()
    print("=" * 70)
    print("MAGNITUDE-CONDITIONED CEM")
    print("=" * 70)
    print(f"Bias values         : {bias_values}")
    print(
        f"Episodes/candidate : "
        f"{args.episodes_per_candidate}"
    )
    print(f"Generations        : {args.generations}")
    print(f"Population         : {args.population}")
    print(f"Residual scale     : {args.residual_scale}")
    print()
    print(
        "Each bias value gets its OWN theta."
    )
    print("=" * 70)

    for idx, bias in enumerate(bias_values):

        bias_seed = (
            args.evo_seed
            + idx * 100000
        )

        result = optimize_one_bias(
            env=env,
            base_model=base_model,
            residual_model=residual_model,
            args=args,
            bias_deg=bias,
            seed=bias_seed,
        )

        results[str(bias)] = result

        best = result["best"]

        print()
        print(
            f"FINAL | bias={bias:+.2f}° | "
            f"success={best['success_rate'] * 100:.2f}% | "
            f"steps={best['mean_steps']:.2f}"
        )

        print(
            f"theta={np.asarray(best['theta'])}"
        )

    output = {
        "method": "magnitude_conditioned_cem",
        "condition": (
            "theta is independently optimized "
            "for each angle bias"
        ),
        "bias_values": bias_values,
        "algorithm": args.algorithm,
        "residual_scale": args.residual_scale,
        "model_path": args.model_path,
        "base_insert_model_path": (
            args.base_insert_model_path
        ),
        "base_insert_algo": args.base_insert_algo,
        "postprocess_mode": args.postprocess_mode,
        "generations": args.generations,
        "population": args.population,
        "episodes_per_candidate": (
            args.episodes_per_candidate
        ),
        "results": results,
    }

    output_path = Path(args.output_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("MAGNITUDE CEM FINISHED")
    print("=" * 70)

    for bias in bias_values:

        best = results[str(bias)]["best"]

        print(
            f"bias={bias:+.2f}° | "
            f"success={best['success_rate'] * 100:.2f}% | "
            f"steps={best['mean_steps']:.2f} | "
            f"traj={best['mean_traj_mm']:.2f} mm"
        )

    print()
    print(f"Saved to: {output_path}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
