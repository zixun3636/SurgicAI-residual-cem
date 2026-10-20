import argparse
import json
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


def init_theta(postprocess_mode):
    if postprocess_mode == "global_scale":
        # theta = [alpha]
        return np.array([1.0], dtype=np.float32)

    if postprocess_mode == "rot_scale_bias":
        # theta = [s_roll, s_pitch, s_yaw, b_roll, b_pitch, b_yaw]
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if postprocess_mode == "scale6":
        # theta = [sx, sy, sz, sroll, spitch, syaw]
        return np.ones(6, dtype=np.float32)

    if postprocess_mode == "scale_bias6":
        # theta = [sx, sy, sz, sroll, spitch, syaw, bx, by, bz, broll, bpitch, byaw]
        return np.concatenate(
            [np.ones(6, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        )

    raise ValueError(f"Unknown postprocess_mode: {postprocess_mode}")


def clip_theta(theta, postprocess_mode, scale_min, scale_max, bias_bound):
    theta = np.asarray(theta, dtype=np.float32).copy()

    if postprocess_mode == "global_scale":
        theta[0] = np.clip(theta[0], scale_min, scale_max)

    elif postprocess_mode == "rot_scale_bias":
        theta[0:3] = np.clip(theta[0:3], scale_min, scale_max)
        theta[3:6] = np.clip(theta[3:6], -bias_bound, bias_bound)

    elif postprocess_mode == "scale6":
        theta[0:6] = np.clip(theta[0:6], scale_min, scale_max)

    elif postprocess_mode == "scale_bias6":
        theta[0:6] = np.clip(theta[0:6], scale_min, scale_max)
        theta[6:12] = np.clip(theta[6:12], -bias_bound, bias_bound)

    else:
        raise ValueError(f"Unknown postprocess_mode: {postprocess_mode}")

    return theta


def theta_to_scale_bias(theta, postprocess_mode):
    scale_vec = np.ones(7, dtype=np.float32)
    bias_vec = np.zeros(7, dtype=np.float32)

    if postprocess_mode == "global_scale":
        scale_vec[0:6] = float(theta[0])

    elif postprocess_mode == "rot_scale_bias":
        scale_vec[3:6] = theta[0:3]
        bias_vec[3:6] = theta[3:6]

    elif postprocess_mode == "scale6":
        scale_vec[0:6] = theta[0:6]

    elif postprocess_mode == "scale_bias6":
        scale_vec[0:6] = theta[0:6]
        bias_vec[0:6] = theta[6:12]

    else:
        raise ValueError(f"Unknown postprocess_mode: {postprocess_mode}")

    scale_vec[6] = 0.0
    bias_vec[6] = 0.0

    return scale_vec, bias_vec


def rollout_score(
    env,
    base_model,
    residual_model,
    theta,
    args,
    eval_seed,
    episodes,
    verbose=False,
):
    theta = clip_theta(
        theta,
        args.postprocess_mode,
        args.scale_min,
        args.scale_max,
        args.bias_bound,
    )

    scale_vec, bias_vec = theta_to_scale_bias(theta, args.postprocess_mode)

    angle_bias_rad = np.deg2rad(args.angle_bias_deg)
    angle_step = float(env.unwrapped.step_size[3])
    bias_in_action_space = angle_bias_rad / angle_step if angle_step != 0 else 0.0

    rng = np.random.default_rng(eval_seed)

    successes = []
    steps_list = []
    traj_list = []
    returns = []

    for ep in range(episodes):
        obs, info = env.reset(seed=eval_seed + ep)

        done = False
        truncated = False
        steps = 0
        traj_mm = 0.0
        ep_return = 0.0

        while not (done or truncated):
            base_action, _ = base_model.predict(obs, deterministic=True)
            residual_action, _ = residual_model.predict(obs, deterministic=True)

            base_action = np.asarray(base_action, dtype=np.float32)
            residual_action = np.asarray(residual_action, dtype=np.float32)

            action = base_action + args.residual_scale * scale_vec * residual_action + bias_vec

            if args.angle_bias_deg != 0:
                action[3:6] += bias_in_action_space

            if args.action_noise_std > 0:
                action += rng.normal(
                    loc=0.0,
                    scale=args.action_noise_std,
                    size=action.shape,
                ).astype(np.float32)

            action = np.clip(action, -1.0, 1.0)

            obs, reward, done, truncated, info = env.step(action)

            steps += 1
            ep_return += float(reward)
            traj_mm += np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000)

        success = bool(info.get("is_success", False))
        successes.append(success)
        steps_list.append(steps)
        traj_list.append(traj_mm)
        returns.append(ep_return)

        if verbose:
            print(
                f"Episode {ep+1:03d} | "
                f"success={success} | "
                f"steps={steps} | "
                f"traj_mm={traj_mm:.2f} | "
                f"return={ep_return:.2f}"
            )

    success_rate = float(np.mean(successes))
    mean_steps = float(np.mean(steps_list))
    mean_traj = float(np.mean(traj_list))
    mean_return = float(np.mean(returns))

    # Success rate is the main target.
    # Shorter time and shorter trajectory are secondary tie-breakers.
    score = success_rate * 1000.0 - mean_steps * 0.5 - mean_traj * 0.01

    result = {
        "score": score,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_traj_mm": mean_traj,
        "mean_return": mean_return,
        "theta": theta.tolist(),
        "scale_vec": scale_vec.tolist(),
        "bias_vec": bias_vec.tolist(),
    }

    return result


def optimize(env, base_model, residual_model, args):
    rng = np.random.default_rng(args.evo_seed)

    mean = init_theta(args.postprocess_mode)
    dim = len(mean)

    std = np.ones(dim, dtype=np.float32) * args.init_std

    # Bias terms usually need a wider search range when the evaluation disturbance is angle bias.
    if args.postprocess_mode == "rot_scale_bias":
        std[0:3] = args.scale_init_std
        std[3:6] = args.bias_init_std
    elif args.postprocess_mode == "scale_bias6":
        std[0:6] = args.scale_init_std
        std[6:12] = args.bias_init_std
    elif args.postprocess_mode in ["global_scale", "scale6"]:
        std[:] = args.scale_init_std

    best = None
    history = []

    print("\n========== Evolution Config ==========")
    print(f"postprocess_mode      : {args.postprocess_mode}")
    print(f"generations           : {args.generations}")
    print(f"population            : {args.population}")
    print(f"elite_frac            : {args.elite_frac}")
    print(f"episodes_per_candidate: {args.episodes_per_candidate}")
    print(f"angle_bias_deg        : {args.angle_bias_deg}")
    print(f"action_noise_std      : {args.action_noise_std}")
    print(f"residual_scale        : {args.residual_scale}")
    print(f"initial mean          : {mean}")
    print(f"initial std           : {std}")
    print("=====================================\n")

    for gen in range(args.generations):
        candidates = []

        # Always include current mean as one candidate.
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

        print(f"\n========== Generation {gen+1}/{args.generations} ==========")

        for i, theta in enumerate(candidates):
            candidate_seed = args.eval_seed + gen * 10000 + i * 100
            result = rollout_score(
                env,
                base_model,
                residual_model,
                theta,
                args,
                eval_seed=candidate_seed,
                episodes=args.episodes_per_candidate,
                verbose=False,
            )
            results.append(result)

            print(
                f"cand={i:02d} | "
                f"score={result['score']:.3f} | "
                f"success={result['success_rate']*100:.2f}% | "
                f"steps={result['mean_steps']:.2f} | "
                f"traj={result['mean_traj_mm']:.2f} | "
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
        print(f"Best gen success : {results[0]['success_rate']*100:.2f}%")
        print(f"Best gen score   : {results[0]['score']:.3f}")
        print(f"Best overall succ: {best['success_rate']*100:.2f}%")
        print(f"Best overall     : {np.array(best['theta'])}")
        print(f"New mean         : {mean}")
        print(f"New std          : {std}")

    output = {
        "postprocess_mode": args.postprocess_mode,
        "residual_scale": args.residual_scale,
        "angle_bias_deg": args.angle_bias_deg,
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

    print("\n========== Evolution Finished ==========")
    print(f"Best success rate: {best['success_rate']*100:.2f}%")
    print(f"Best score       : {best['score']:.3f}")
    print(f"Best theta       : {np.array(best['theta'])}")
    print(f"Scale vec        : {np.array(best['scale_vec'])}")
    print(f"Bias vec         : {np.array(best['bias_vec'])}")
    print(f"Saved to         : {output_path}")
    print("=======================================\n")

    return output


def eval_with_params(env, base_model, residual_model, args):
    params_path = Path(args.params_path)
    data = json.loads(params_path.read_text())

    theta = np.asarray(data["best"]["theta"], dtype=np.float32)

    print("\n========== Loaded Evolved Params ==========")
    print(f"params_path      : {params_path}")
    print(f"postprocess_mode : {args.postprocess_mode}")
    print(f"theta            : {theta}")
    print("==========================================\n")

    result = rollout_score(
        env,
        base_model,
        residual_model,
        theta,
        args,
        eval_seed=args.eval_seed,
        episodes=args.episodes,
        verbose=True,
    )

    print("\n========== Evolved Evaluation Result ==========")
    print(f"Angle bias deg       : {args.angle_bias_deg}")
    print(f"Action noise std     : {args.action_noise_std}")
    print(f"Episodes             : {args.episodes}")
    print(f"Success Rate         : {result['success_rate']*100:.2f}%")
    print(f"Mean Steps           : {result['mean_steps']:.2f}")
    print(f"Mean Trajectory      : {result['mean_traj_mm']:.2f} mm")
    print(f"Mean Return          : {result['mean_return']:.2f}")
    print(f"Theta                : {np.array(result['theta'])}")
    print(f"Scale vec            : {np.array(result['scale_vec'])}")
    print(f"Bias vec             : {np.array(result['bias_vec'])}")
    print("==============================================\n")


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
    parser.add_argument("--episodes_per_candidate", type=int, default=20)

    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--angle_bias_deg", type=float, default=2.0)

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

    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite_frac", type=float, default=0.25)

    parser.add_argument("--init_std", type=float, default=0.3)
    parser.add_argument("--scale_init_std", type=float, default=0.3)
    parser.add_argument("--bias_init_std", type=float, default=0.4)
    parser.add_argument("--min_std", type=float, default=0.03)

    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--bias_bound", type=float, default=1.0)

    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/suzixun/SurgicAI/RL/evolution_results/best_residual_postprocess.json",
    )
    parser.add_argument("--params_path", type=str, default=None)

    args = parser.parse_args()

    env, step_size, threshold, max_episode_steps = setup_environment(args)

    residual_model = load_model(args.algorithm, args.model_path, env)
    base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    if args.mode == "optimize":
        optimize(env, base_model, residual_model, args)

    elif args.mode == "eval":
        if args.params_path is None:
            raise ValueError("--params_path is required when --mode eval")
        eval_with_params(env, base_model, residual_model, args)

    env.close()


if __name__ == "__main__":
    main()
