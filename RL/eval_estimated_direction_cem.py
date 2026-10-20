"""
Estimated-direction CEM calibration (擾動方向估計 + 方向條件式 CEM).

Problem
-------
Direction-conditioned CEM (eval_direction_conditioned_cem.py) chooses
theta_neg when the injected rotational action bias d < 0 and theta_pos otherwise.
That rule needs the sign of d to be known in advance, which is not available
at deployment.

Idea
----
The rotational action bias is added to every action before the environment
executes it. Insert_env integrates the executed action into the PSM goal pose,
and observation[3:6] is that pose (rad). So at every step

    delta_t = (rot_{t+1} - rot_t) / angle_step = clip(a_pre_t + d, -1, 1)

where a_pre_t is the action our controller *intended* (base + residual + CEM,
before the bias). When the executed action on an axis is not saturated,

    d_hat_t = delta_t - a_pre_t            (per axis, in action units)

is an unbiased per-step estimate of d. The controller averages the unsaturated
samples during a short warm-up (theta = neutral = residual only), then picks
theta_pos / theta_neg from the sign of the estimate.

Methods compared under the same episode seeds
---------------------------------------------
  oracle         : sign of the TRUE bias selects theta (= previous experiment)
  estimated      : sign of d_hat after warm-up selects theta (this experiment)
  residual_only  : theta = [1,1,1,0,0,0] for the whole episode
  pos_only       : theta_pos for the whole episode (single CEM parameter set)

Robustness knobs
----------------
  --est_noise_deg   Gaussian noise (deg) added to the rotation reading used by
                    the estimator only (simulates pose-sensing noise; the policy
                    still sees the clean observation, so only the estimator is
                    stressed).
  --action_noise_std  same per-step action noise as the earlier scripts.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import load_model, clip_theta, theta_to_scale_bias


NEUTRAL_THETA = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
ALL_METHODS = ["oracle", "estimated", "residual_only", "pos_only"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def load_theta(params_path):
    data = json.loads(Path(params_path).read_text())
    if "best" not in data or "theta" not in data["best"]:
        raise KeyError(f"Cannot find data['best']['theta'] in {params_path}")
    return np.asarray(data["best"]["theta"], dtype=np.float32)


def get_rot(obs):
    """Rotation part of the current PSM pose (rad). observation[3:6] is not rescaled."""
    if isinstance(obs, dict):
        return np.asarray(obs["observation"][3:6], dtype=np.float64).copy()
    return np.asarray(obs[3:6], dtype=np.float64).copy()


class DirectionEstimator:
    """Running mean of per-axis bias samples taken from unsaturated steps."""

    def __init__(self, sat_margin=0.98, kind="mean"):
        self.sat_margin = sat_margin
        self.kind = kind
        self.sum = np.zeros(3)
        self.count = np.zeros(3)
        self.samples = [[], [], []]

    def update(self, rot_before, rot_after, a_pre_rot, angle_step):
        delta = (rot_after - rot_before) / angle_step          # executed action (rot)
        valid = np.abs(delta) < self.sat_margin                # not clipped at +-1
        sample = delta - a_pre_rot
        self.sum[valid] += sample[valid]
        self.count[valid] += 1
        for i in range(3):
            if valid[i]:
                self.samples[i].append(float(sample[i]))
        return sample, valid

    @property
    def estimate(self):
        """Per-axis estimate in action units; NaN where no valid sample yet."""
        if self.kind == "median":
            return np.array([np.median(x) if x else np.nan for x in self.samples])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.sum / np.maximum(self.count, 1), np.nan)

    def scalar_estimate(self):
        """Pool all valid samples of the three axes (scalar bias [d,d,d] assumption)."""
        pooled = self.samples[0] + self.samples[1] + self.samples[2]
        if not pooled:
            return np.nan
        return float(np.median(pooled) if self.kind == "median" else np.mean(pooled))


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(
    env,
    base_model,
    residual_model,
    method,
    bias_deg_vec,
    theta_pos,
    theta_neg,
    args,
    ep_seed,
    rng,
):
    angle_step = float(env.unwrapped.step_size[3])
    bias_action = np.deg2rad(np.asarray(bias_deg_vec, dtype=np.float64)) / angle_step
    # 0 = no bias: any choice is acceptable, excluded from direction accuracy
    true_dir = float(np.sign(np.sum(bias_action)))

    def theta_to_vecs(theta):
        theta = clip_theta(theta, args.postprocess_mode, args.scale_min, args.scale_max, args.bias_bound)
        return theta_to_scale_bias(theta, args.postprocess_mode)

    vecs = {
        "neutral": theta_to_vecs(NEUTRAL_THETA),
        "theta_pos": theta_to_vecs(theta_pos),
        "theta_neg": theta_to_vecs(theta_neg),
    }

    # Initial theta choice
    if method == "oracle":
        theta_name = "theta_pos" if true_dir >= 0 else "theta_neg"  # same rule as before: d>=0 -> pos
    elif method == "pos_only":
        theta_name = "theta_pos"
    else:  # residual_only, estimated (warm-up)
        theta_name = "neutral"

    estimator = DirectionEstimator(sat_margin=args.sat_margin, kind=args.estimator)
    decided = method != "estimated"
    decision_step = None
    decision_estimate = None
    est_trace = []
    switches = 0

    obs, info = env.reset(seed=ep_seed)
    done = truncated = False
    steps = 0
    traj_mm = 0.0
    ep_return = 0.0
    sat_count = np.zeros(3)

    while not (done or truncated):
        scale_vec, bias_vec = vecs[theta_name]

        base_action, _ = base_model.predict(obs, deterministic=True)
        residual_action, _ = residual_model.predict(obs, deterministic=True)
        base_action = np.asarray(base_action, dtype=np.float32)
        residual_action = np.asarray(residual_action, dtype=np.float32)

        a_pre = base_action + args.residual_scale * scale_vec * residual_action + bias_vec
        action = a_pre.copy()
        action[3:6] += bias_action.astype(np.float32)
        if args.action_noise_std > 0:
            action += rng.normal(0.0, args.action_noise_std, size=action.shape).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        sat_count += (np.abs(action[3:6]) >= 0.999).astype(float)

        rot_before = get_rot(obs)
        obs, reward, done, truncated, info = env.step(action.copy())
        rot_after = get_rot(obs)

        steps += 1
        ep_return += float(reward)
        traj_mm += float(np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000))

        if method == "estimated":
            # The estimator only sees a (possibly noisy) pose reading.
            if args.est_noise_deg > 0:
                noise = np.deg2rad(rng.normal(0.0, args.est_noise_deg, size=3))
                rot_b_meas = rot_before + noise
                rot_a_meas = rot_after + np.deg2rad(rng.normal(0.0, args.est_noise_deg, size=3))
            else:
                rot_b_meas, rot_a_meas = rot_before, rot_after
            estimator.update(rot_b_meas, rot_a_meas, a_pre[3:6].astype(np.float64), angle_step)
            d_hat = estimator.scalar_estimate()
            est_trace.append(None if np.isnan(d_hat) else d_hat)

            ready = steps >= args.warmup_steps and not np.isnan(d_hat)
            if (not decided and ready) or (decided and args.keep_updating and ready):
                if abs(d_hat) < args.deadzone:
                    new_name = "neutral"
                else:
                    new_name = "theta_pos" if d_hat > 0 else "theta_neg"
                if not decided:
                    decided = True
                    decision_step = steps
                    decision_estimate = d_hat
                elif new_name != theta_name:
                    switches += 1
                theta_name = new_name

    success = bool(info.get("is_success", False))

    record = {
        "seed": int(ep_seed),
        "bias_deg": [float(x) for x in bias_deg_vec],
        "success": success,
        "steps": int(steps),
        "traj_mm": float(traj_mm),
        "return": float(ep_return),
        "final_theta": theta_name,
        "rot_saturation_rate": (sat_count / max(steps, 1)).tolist(),
    }
    if method == "estimated":
        est_vec = estimator.estimate
        record.update(
            {
                "decision_step": decision_step,
                "decision_estimate_action": decision_estimate,
                "decision_estimate_deg": None
                if decision_estimate is None
                else float(np.rad2deg(decision_estimate * angle_step)),
                "final_estimate_axis_deg": [
                    None if np.isnan(v) else float(np.rad2deg(v * angle_step)) for v in est_vec
                ],
                "true_dir": true_dir,
                "chosen_dir": None
                if decision_estimate is None
                else (0.0 if abs(decision_estimate) < args.deadzone else float(np.sign(decision_estimate))),
                "switches": switches,
                "est_trace_action": est_trace[: args.trace_len],
            }
        )
    return record


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def summarize(records):
    succ = np.array([r["success"] for r in records], dtype=float)
    steps = np.array([r["steps"] for r in records], dtype=float)
    s = {
        "episodes": len(records),
        "success_rate": float(succ.mean()),
        "mean_steps": float(steps.mean()),
        "mean_steps_success": float(steps[succ > 0].mean()) if succ.any() else None,
        "mean_traj_mm": float(np.mean([r["traj_mm"] for r in records])),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "rot_saturation_rate": np.mean([r["rot_saturation_rate"] for r in records], axis=0).tolist(),
    }
    if "chosen_dir" in records[0]:
        chosen = [r["chosen_dir"] for r in records]
        true = [r["true_dir"] for r in records]
        decided = [c is not None for c in chosen]
        correct = [c == t for c, t in zip(chosen, true) if c is not None and t != 0]
        dsteps = [r["decision_step"] for r in records if r["decision_step"] is not None]
        est_deg = [r["decision_estimate_deg"] for r in records if r["decision_estimate_deg"] is not None]
        s.update(
            {
                "decided_rate": float(np.mean(decided)),
                "direction_accuracy": float(np.mean(correct)) if correct else None,
                "mean_decision_step": float(np.mean(dsteps)) if dsteps else None,
                "decision_estimate_deg_mean": float(np.mean(est_deg)) if est_deg else None,
                "decision_estimate_deg_std": float(np.std(est_deg)) if est_deg else None,
                "mean_switches": float(np.mean([r["switches"] for r in records])),
            }
        )
    return s


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", type=str, default="TD3_HER_BC")
    p.add_argument("--task_name", type=str, default="Insert")
    p.add_argument("--reward_type", type=str, default="sparse")
    p.add_argument("--trans_error", type=float, default=0.5)
    p.add_argument("--angle_error", type=float, default=10)

    p.add_argument("--eval_seed", type=int, default=100)
    p.add_argument("--episodes", type=int, default=100)

    p.add_argument("--bias_mode", choices=["scalar", "random3d"], default="scalar")
    p.add_argument("--bias_list", type=str, default="-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5",
                   help="scalar mode: d (deg) applied to roll/pitch/yaw")
    p.add_argument("--random3d_low", type=float, default=-2.0)
    p.add_argument("--random3d_high", type=float, default=2.0)

    p.add_argument("--methods", type=str, default="oracle,estimated,residual_only,pos_only")

    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--deadzone", type=float, default=0.0,
                   help="|d_hat| (action units) below this keeps the neutral theta")
    p.add_argument("--sat_margin", type=float, default=0.98)
    p.add_argument("--estimator", choices=["mean", "median"], default="mean")
    p.add_argument("--keep_updating", action="store_true",
                   help="re-select theta every step after warm-up instead of locking once")
    p.add_argument("--est_noise_deg", type=float, default=0.0)
    p.add_argument("--action_noise_std", type=float, default=0.0)
    p.add_argument("--trace_len", type=int, default=40)

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--base_insert_model_path", type=str, required=True)
    p.add_argument("--base_insert_algo", type=str, default="TD3_HER_BC")
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--postprocess_mode", type=str, default="rot_scale_bias",
                   choices=["rot_scale_bias"])
    p.add_argument("--positive_params_path", type=str, required=True)
    p.add_argument("--negative_params_path", type=str, required=True)
    p.add_argument("--scale_min", type=float, default=0.0)
    p.add_argument("--scale_max", type=float, default=2.0)
    p.add_argument("--bias_bound", type=float, default=1.0)

    p.add_argument("--save_json", type=str,
                   default="/home/suzixun/SurgicAI/RL/evolution_results/estimated_direction/result.json")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"unknown method {m}; choose from {ALL_METHODS}")

    theta_pos = load_theta(args.positive_params_path)
    theta_neg = load_theta(args.negative_params_path)

    # Build the list of bias conditions. Each condition gets its own seed block,
    # identical across methods (matched comparison). For scalar mode the seed
    # offset (eval_seed + idx*1000) follows eval_direction_conditioned_cem.py,
    # so "oracle" reproduces the earlier direction-conditioned numbers when the
    # same bias_list order is used.
    if args.bias_mode == "scalar":
        bias_values = parse_float_list(args.bias_list)
        conditions = [
            {"key": f"{b:g}", "bias_vec_fn": (lambda ep, b=b: [b, b, b]), "seed0": args.eval_seed + i * 1000}
            for i, b in enumerate(bias_values)
        ]
    else:
        bias_rng = np.random.default_rng(args.eval_seed + 777)
        vecs = bias_rng.uniform(args.random3d_low, args.random3d_high, size=(args.episodes, 3))
        conditions = [{"key": "random3d", "bias_vec_fn": (lambda ep: vecs[ep].tolist()), "seed0": args.eval_seed}]

    args.angle_bias_deg = 0.0
    env, step_size, threshold, max_episode_steps = setup_environment(args)
    residual_model = load_model(args.algorithm, args.model_path, env)
    base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    print("\n========== Estimated-Direction CEM ==========")
    print(f"theta_pos      : {theta_pos}")
    print(f"theta_neg      : {theta_neg}")
    print(f"bias_mode      : {args.bias_mode}")
    print(f"conditions     : {[c['key'] for c in conditions]}")
    print(f"methods        : {methods}")
    print(f"warmup_steps   : {args.warmup_steps}   deadzone: {args.deadzone}   keep_updating: {args.keep_updating}")
    print(f"est_noise_deg  : {args.est_noise_deg}   action_noise_std: {args.action_noise_std}")
    print("=============================================\n")

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "method": "estimated_direction_cem",
        "config": {k: v for k, v in vars(args).items()},
        "theta_pos": theta_pos.tolist(),
        "theta_neg": theta_neg.tolist(),
        "results": {},  # results[condition][method] = {"summary":..., "episodes":[...]}
    }

    t0 = time.time()
    for cond in conditions:
        output["results"][cond["key"]] = {}
        for method in methods:
            # Separate RNG per (condition, method) so noise draws are reproducible
            # and identical across methods for the same condition.
            rng = np.random.default_rng(cond["seed0"] + 12345)
            records = []
            for ep in range(args.episodes):
                rec = run_episode(
                    env, base_model, residual_model, method,
                    cond["bias_vec_fn"](ep), theta_pos, theta_neg,
                    args, cond["seed0"] + ep, rng,
                )
                records.append(rec)
            summ = summarize(records)
            output["results"][cond["key"]][method] = {"summary": summ, "episodes": records}

            line = (
                f"bias={cond['key']:>8} | {method:>13} | "
                f"success={summ['success_rate']*100:6.2f}% | steps={summ['mean_steps']:6.2f}"
            )
            if method == "estimated":
                acc = summ["direction_accuracy"]
                line += (
                    f" | dir_acc={'n/a' if acc is None else f'{acc*100:.1f}%'}"
                    f" | decide@{summ['mean_decision_step']}"
                    f" | d_hat={summ['decision_estimate_deg_mean']}"
                )
            print(line, flush=True)

            # checkpoint after every (condition, method) so a long run is never lost
            save_path.write_text(json.dumps(output, indent=1))

    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
    print("\n========== Summary (success %) ==========")
    header = f"{'bias':>8} | " + " | ".join(f"{m:>13}" for m in methods)
    print(header)
    for key, res in output["results"].items():
        print(f"{key:>8} | " + " | ".join(f"{res[m]['summary']['success_rate']*100:13.1f}" for m in methods))
    print(f"\nSaved to: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
