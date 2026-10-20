import argparse
import json
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import (
    load_model,
    init_theta,
    clip_theta,
    theta_to_scale_bias,
)


def generate_bias_bank(seed, episodes, max_bias_deg):
    """
    Each episode gets an independent:
        [roll_bias, pitch_bias, yaw_bias]
    where every axis ~ Uniform(-max_bias_deg, +max_bias_deg)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(
        low=-max_bias_deg,
        high=max_bias_deg,
        size=(episodes, 3),
    ).astype(np.float32)


def evaluate_theta(
    env,
    base_model,
    residual_model,
    theta,
    args,
    bias_bank_deg,
    env_seed_base,
    verbose=False,
):
    theta = clip_theta(
        theta,
        args.postprocess_mode,
        args.scale_min,
        args.scale_max,
        args.bias_bound,
    )

    scale_vec, calibration_bias_vec = theta_to_scale_bias(
        theta,
        args.postprocess_mode,
    )

    rotation_step = np.asarray(
        env.unwrapped.step_size[3:6],
        dtype=np.float32,
    )

    successes = []
    steps_list = []
    traj_list = []
    returns = []

    # Save per-episode result for later analysis
    episode_records = []

    # octant statistics, e.g. (+,-,+)
    octant_stats = {}

    for ep, bias_deg in enumerate(bias_bank_deg):

        obs, info = env.reset(seed=env_seed_base + ep)

        # Convert degree-equivalent disturbance into normalized action space
        bias_action = (
            np.deg2rad(bias_deg) / rotation_step
        ).astype(np.float32)

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

            # Base + calibrated Residual
            action = (
                base_action
                + args.residual_scale
                * scale_vec
                * residual_action
                + calibration_bias_vec
            )

            # Random 3-DOF disturbance
            action[3:6] += bias_action

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

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

        sign_key = tuple(
            "+" if x >= 0 else "-"
            for x in bias_deg
        )

        sign_name = "".join(sign_key)

        if sign_name not in octant_stats:
            octant_stats[sign_name] = {
                "successes": [],
                "steps": [],
            }

        octant_stats[sign_name]["successes"].append(success)
        octant_stats[sign_name]["steps"].append(steps)

        episode_records.append({
            "episode": ep,
            "bias_deg": bias_deg.tolist(),
            "bias_action": bias_action.tolist(),
            "success": success,
            "steps": steps,
            "trajectory_mm": float(traj_mm),
            "return": float(ep_return),
        })

        if verbose:
            print(
                f"Ep {ep+1:03d} | "
                f"bias=[{bias_deg[0]:+.2f}, "
                f"{bias_deg[1]:+.2f}, "
                f"{bias_deg[2]:+.2f}] deg | "
                f"success={success} | "
                f"steps={steps}"
            )

    success_rate = float(np.mean(successes))
    mean_steps = float(np.mean(steps_list))
    mean_traj = float(np.mean(traj_list))
    mean_return = float(np.mean(returns))

    # Same scoring idea as previous CEM
    score = (
        success_rate * 1000.0
        - mean_steps * 0.5
        - mean_traj * 0.01
    )

    octant_summary = {}

    for name, d in sorted(octant_stats.items()):
        octant_summary[name] = {
            "episodes": len(d["successes"]),
            "success_rate": float(np.mean(d["successes"])),
            "mean_steps": float(np.mean(d["steps"])),
        }

    return {
        "score": score,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_traj_mm": mean_traj,
        "mean_return": mean_return,
        "theta": theta.tolist(),
        "scale_vec": scale_vec.tolist(),
        "calibration_bias_vec": calibration_bias_vec.tolist(),
        "octants": octant_summary,
        "episodes": episode_records,
    }


def optimize(
    env,
    base_model,
    residual_model,
    args,
):
    evo_rng = np.random.default_rng(args.evo_seed)

    mean = init_theta(args.postprocess_mode)
    dim = len(mean)

    std = np.ones(
        dim,
        dtype=np.float32,
    ) * args.init_std

    if args.postprocess_mode == "rot_scale_bias":
        std[0:3] = args.scale_init_std
        std[3:6] = args.bias_init_std

    best = None
    history = []

    print()
    print("========== Random 3D CEM ==========")
    print(f"bias range per axis   : ±{args.max_bias_deg} deg")
    print(f"generations           : {args.generations}")
    print(f"population            : {args.population}")
    print(f"elite fraction        : {args.elite_frac}")
    print(f"episodes/candidate    : {args.episodes_per_candidate}")
    print(f"evo seed              : {args.evo_seed}")
    print(f"bias seed             : {args.bias_seed}")
    print(f"env seed              : {args.eval_seed}")
    print(f"residual scale        : {args.residual_scale}")
    print(f"initial mean          : {mean}")
    print(f"initial std           : {std}")
    print("===================================")
    print()

    for gen in range(args.generations):

        # Important:
        # Every candidate within the SAME generation gets the
        # exact same random 3D bias bank and environment seeds.
        # This makes candidate comparison much fairer.
        bias_bank = generate_bias_bank(
            seed=args.bias_seed + gen,
            episodes=args.episodes_per_candidate,
            max_bias_deg=args.max_bias_deg,
        )

        candidates = [mean.copy()]

        for _ in range(args.population - 1):

            theta = (
                mean
                + std
                * evo_rng.standard_normal(dim).astype(np.float32)
            )

            theta = clip_theta(
                theta,
                args.postprocess_mode,
                args.scale_min,
                args.scale_max,
                args.bias_bound,
            )

            candidates.append(theta)

        results = []

        print(
            f"\n========== Generation "
            f"{gen+1}/{args.generations} =========="
        )

        for i, theta in enumerate(candidates):

            result = evaluate_theta(
                env,
                base_model,
                residual_model,
                theta,
                args,
                bias_bank_deg=bias_bank,
                env_seed_base=(
                    args.eval_seed
                    + gen * 10000
                ),
                verbose=False,
            )

            results.append(result)

            print(
                f"cand={i:02d} | "
                f"score={result['score']:.2f} | "
                f"success={result['success_rate']*100:.1f}% | "
                f"steps={result['mean_steps']:.2f} | "
                f"traj={result['mean_traj_mm']:.2f} | "
                f"theta={np.asarray(result['theta'])}"
            )

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True,
        )

        if (
            best is None
            or results[0]["score"] > best["score"]
        ):
            best = {
                **results[0],
                "generation": gen + 1,
            }

        elite_count = max(
            1,
            int(args.population * args.elite_frac),
        )

        elites = results[:elite_count]

        elite_thetas = np.asarray(
            [x["theta"] for x in elites],
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

        history.append({
            "generation": gen + 1,
            "training_bias_bank_deg": bias_bank.tolist(),
            "best_generation": results[0],
            "new_mean": mean.tolist(),
            "new_std": std.tolist(),
        })

        print()
        print("--- Generation Summary ---")
        print(
            f"Best success : "
            f"{results[0]['success_rate']*100:.2f}%"
        )
        print(
            f"Best score   : "
            f"{results[0]['score']:.2f}"
        )
        print(
            f"Best theta   : "
            f"{np.asarray(results[0]['theta'])}"
        )
        print(f"New mean     : {mean}")
        print(f"New std      : {std}")

    return best, mean, std, history


def run_validation(
    env,
    base_model,
    residual_model,
    best_theta,
    args,
):
    print()
    print("========== HELD-OUT VALIDATION ==========")
    print(
        f"Generating {args.validation_episodes} NEW "
        f"random 3D biases from ±{args.max_bias_deg} deg"
    )
    print(
        f"validation bias seed = "
        f"{args.validation_bias_seed}"
    )
    print("=========================================")
    print()

    validation_bank = generate_bias_bank(
        seed=args.validation_bias_seed,
        episodes=args.validation_episodes,
        max_bias_deg=args.max_bias_deg,
    )

    # Same random biases + same env initial seeds
    # for Residual-only and CEM theta
    baseline_theta = np.array(
        [1, 1, 1, 0, 0, 0],
        dtype=np.float32,
    )

    baseline = evaluate_theta(
        env,
        base_model,
        residual_model,
        baseline_theta,
        args,
        bias_bank_deg=validation_bank,
        env_seed_base=args.validation_env_seed,
        verbose=False,
    )

    evolved = evaluate_theta(
        env,
        base_model,
        residual_model,
        np.asarray(best_theta, dtype=np.float32),
        args,
        bias_bank_deg=validation_bank,
        env_seed_base=args.validation_env_seed,
        verbose=False,
    )

    print("Residual Only")
    print(
        f"  Success : "
        f"{baseline['success_rate']*100:.2f}%"
    )
    print(
        f"  Steps   : "
        f"{baseline['mean_steps']:.2f}"
    )
    print(
        f"  Traj    : "
        f"{baseline['mean_traj_mm']:.2f} mm"
    )

    print()
    print("Residual + Random3D CEM")
    print(
        f"  Success : "
        f"{evolved['success_rate']*100:.2f}%"
    )
    print(
        f"  Steps   : "
        f"{evolved['mean_steps']:.2f}"
    )
    print(
        f"  Traj    : "
        f"{evolved['mean_traj_mm']:.2f} mm"
    )

    print()
    print("Per-octant success rate:")

    all_octants = sorted(
        set(baseline["octants"].keys())
        | set(evolved["octants"].keys())
    )

    for octant in all_octants:

        b = baseline["octants"].get(octant)
        e = evolved["octants"].get(octant)

        b_rate = (
            b["success_rate"] * 100
            if b is not None
            else float("nan")
        )

        e_rate = (
            e["success_rate"] * 100
            if e is not None
            else float("nan")
        )

        n = (
            e["episodes"]
            if e is not None
            else b["episodes"]
        )

        print(
            f"  {octant} | "
            f"N={n:3d} | "
            f"Residual={b_rate:6.2f}% | "
            f"CEM={e_rate:6.2f}%"
        )

    return validation_bank, baseline, evolved


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
        "--residual_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--postprocess_mode",
        type=str,
        default="rot_scale_bias",
        choices=["rot_scale_bias"],
    )

    # Random 3D disturbance range
    parser.add_argument(
        "--max_bias_deg",
        type=float,
        default=2.0,
    )

    # CEM
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
        "--episodes_per_candidate",
        type=int,
        default=20,
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

    # Random seeds
    parser.add_argument(
        "--evo_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--bias_seed",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval_seed",
        type=int,
        default=2000,
    )

    # Independent validation
    parser.add_argument(
        "--validation_episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--validation_bias_seed",
        type=int,
        default=9000,
    )

    parser.add_argument(
        "--validation_env_seed",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=(
            "/home/suzixun/SurgicAI/RL/"
            "evolution_results/random3d_cem.json"
        ),
    )

    args = parser.parse_args()

    env, step_size, threshold, max_episode_steps = (
        setup_environment(args)
    )

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

    best, final_mean, final_std, history = optimize(
        env,
        base_model,
        residual_model,
        args,
    )

    print()
    print("========== BEST SEARCH RESULT ==========")
    print(f"Generation   : {best['generation']}")
    print(f"Success      : {best['success_rate']*100:.2f}%")
    print(f"Mean steps   : {best['mean_steps']:.2f}")
    print(f"Mean traj    : {best['mean_traj_mm']:.2f}")
    print(f"Theta        : {np.asarray(best['theta'])}")
    print("========================================")

    (
        validation_bank,
        validation_baseline,
        validation_evolved,
    ) = run_validation(
        env,
        base_model,
        residual_model,
        best["theta"],
        args,
    )

    output = {
        "method": "random_3d_rotational_bias_cem",
        "bias_distribution": {
            "type": "independent_uniform_per_axis",
            "roll_deg": [
                -args.max_bias_deg,
                args.max_bias_deg,
            ],
            "pitch_deg": [
                -args.max_bias_deg,
                args.max_bias_deg,
            ],
            "yaw_deg": [
                -args.max_bias_deg,
                args.max_bias_deg,
            ],
        },
        "model_path": args.model_path,
        "base_insert_model_path": (
            args.base_insert_model_path
        ),
        "residual_scale": args.residual_scale,
        "generations": args.generations,
        "population": args.population,
        "elite_frac": args.elite_frac,
        "episodes_per_candidate": (
            args.episodes_per_candidate
        ),
        "evo_seed": args.evo_seed,
        "bias_seed": args.bias_seed,
        "eval_seed": args.eval_seed,
        "best_search_result": best,
        "final_mean": final_mean.tolist(),
        "final_std": final_std.tolist(),
        "history": history,
        "validation": {
            "episodes": args.validation_episodes,
            "bias_seed": args.validation_bias_seed,
            "env_seed": args.validation_env_seed,
            "bias_bank_deg": validation_bank.tolist(),
            "residual_only": validation_baseline,
            "residual_plus_cem": validation_evolved,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(f"Saved to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
