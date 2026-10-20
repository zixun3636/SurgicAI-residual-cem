"""
Estimated-direction CEM, version 2 (擾動方向估計 v2).

What is new compared with eval_estimated_direction_cem.py (v1, kept unchanged
so the 10/15 main experiment stays reproducible):

1. Estimator variants (exp. 1)
   - which axes feed the estimator: all three, or pitch only (pitch saturates
     far less often than roll / yaw, so its samples are less censored)
   - decision rule: fixed warm-up (v1) or sequential: decide as soon as the
     pooled samples give |mean| / (sd/sqrt(n)) >= z, otherwise keep waiting,
     forced decision at --max_wait
2. Signed saturation logging (exp. 1)
   - for every axis: fraction of steps the executed action is pinned at +1 / -1,
     the same for the pre-bias action a_pre, and the fraction of estimator
     samples that were kept
3. Random 3-DOF bias with the new estimators (exp. 2)  --bias_mode random3d
4. Bias that flips sign mid-episode (exp. 3)  --flip_step K
   - "adaptive" keeps updating a windowed estimate after the first decision
     and switches theta when the opposite sign becomes significant
5. Residual model can be a BC-only model (exp. 4)  --algorithm BC

Methods
-------
  oracle         theta chosen from the sign of the TRUE current bias (switches
                 exactly at the flip step when --flip_step is used)
  residual_only  theta = [1,1,1,0,0,0]
  pos_only       theta_pos for the whole episode
  est_v1         fixed warm-up, all axes                      (= v1 estimated)
  est_pitch      fixed warm-up, pitch axis only
  est_seq        sequential decision, all axes
  est_seq_pitch  sequential decision, pitch axis only
  adaptive       est_seq + keep updating (windowed) after the first decision
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import clip_theta, theta_to_scale_bias


NEUTRAL_THETA = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

ESTIMATOR_METHODS = {
    #                decision      axes          keep_updating
    "est_v1":        ("fixed",      (0, 1, 2),    False),
    "est_pitch":     ("fixed",      (1,),         False),
    "est_seq":       ("sequential", (0, 1, 2),    False),
    "est_seq_pitch": ("sequential", (1,),         False),
    "adaptive":      ("sequential", None,         True),   # axes from --adaptive_axes
    # filter on the COMMANDED action a_pre instead of the measured executed action:
    "est_cmd":       ("fixed",      (0, 1, 2),    False),
    "est_cmd_seq":   ("sequential", (0, 1, 2),    False),
}
CMD_FILTER_METHODS = {"est_cmd", "est_cmd_seq"}
ALL_METHODS = ["oracle", "residual_only", "pos_only"] + list(ESTIMATOR_METHODS)


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


def load_model(algorithm, model_path, env):
    algorithm = algorithm.upper()
    model_path = Path(model_path)
    print(f"Loading model ({algorithm}): {model_path}  exists={model_path.exists()}")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if algorithm in ["TD3_BC", "TD3_HER_BC"]:
        from RL_algo.td3_BC import TD3_BC
        return TD3_BC.load(str(model_path), env=env)
    if algorithm == "BC":
        from RL_algo.BC import BC
        return BC.load(str(model_path), env=env)
    if algorithm in ["TD3", "TD3_HER"]:
        try:
            from RL_algo.td3 import TD3
        except Exception:
            from stable_baselines3 import TD3
        return TD3.load(str(model_path), env=env)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def get_rot(obs):
    """Rotation part of the current PSM pose (rad). observation[3:6] is not rescaled."""
    if isinstance(obs, dict):
        return np.asarray(obs["observation"][3:6], dtype=np.float64).copy()
    return np.asarray(obs[3:6], dtype=np.float64).copy()


def sign_name(x):
    return "theta_pos" if x > 0 else "theta_neg"


class Estimator:
    """
    Collects per-step bias samples d_hat = executed_rot_action - a_pre_rot
    from unsaturated steps on the selected axes.
    """

    def __init__(self, axes, sat_margin, window=None, filter_mode="measured", cmd_margin=0.02):
        """
        filter_mode
          "measured": keep a sample when the MEASURED executed action is not at +-1
                      (v1 rule). With pose noise, steps that were truly saturated can
                      leak through; if a_pre was already beyond +-1 those leaked
                      samples carry the WRONG sign.
          "commanded": keep a sample when the COMMANDED action |a_pre| < 1 - cmd_margin,
                      whatever the executed action did. Then
                      sample = clip(a_pre + d) - a_pre always has the sign of d
                      (it is only shrunk when clipping happens), so the pooled mean
                      is sign-consistent even with noise.
        """
        self.axes = tuple(axes)
        self.sat_margin = sat_margin
        self.filter_mode = filter_mode
        self.cmd_margin = cmd_margin
        self.all_samples = []                       # every kept sample (pooled)
        self.window = deque(maxlen=window) if window else None
        self.kept = np.zeros(3)
        self.seen = np.zeros(3)

    def update(self, rot_before, rot_after, a_pre_rot, angle_step):
        delta = (rot_after - rot_before) / angle_step
        sample = delta - a_pre_rot
        for i in self.axes:
            self.seen[i] += 1
            if self.filter_mode == "commanded":
                keep = abs(a_pre_rot[i]) < 1.0 - self.cmd_margin
            else:
                keep = abs(delta[i]) < self.sat_margin
            if keep:
                self.kept[i] += 1
                self.all_samples.append(float(sample[i]))
                if self.window is not None:
                    self.window.append(float(sample[i]))

    @staticmethod
    def stats(samples, se_floor):
        n = len(samples)
        if n == 0:
            return 0, np.nan, 0.0
        x = np.asarray(samples)
        m = float(x.mean())
        if n < 2:
            return n, m, 0.0
        se = max(float(x.std(ddof=1)) / np.sqrt(n), se_floor)
        return n, m, m / se

    def global_stats(self, se_floor):
        return self.stats(self.all_samples, se_floor)

    def window_stats(self, se_floor):
        return self.stats(list(self.window) if self.window is not None else self.all_samples, se_floor)

    def kept_rate(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return [None if self.seen[i] == 0 else float(self.kept[i] / self.seen[i]) for i in range(3)]


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(env, base_model, residual_model, method, base_bias_deg, theta_pos, theta_neg,
                args, ep_seed, rng):
    angle_step = float(env.unwrapped.step_size[3])
    base_bias_action = np.deg2rad(np.asarray(base_bias_deg, dtype=np.float64)) / angle_step

    def bias_at(step):  # step counts from 1
        if args.flip_step > 0 and step > args.flip_step:
            return -base_bias_action
        return base_bias_action

    def true_dir_at(step):
        return float(np.sign(np.sum(bias_at(step))))

    def vecs_of(theta):
        theta = clip_theta(theta, "rot_scale_bias", args.scale_min, args.scale_max, args.bias_bound)
        return theta_to_scale_bias(theta, "rot_scale_bias")

    vecs = {"neutral": vecs_of(NEUTRAL_THETA), "theta_pos": vecs_of(theta_pos), "theta_neg": vecs_of(theta_neg)}

    is_est = method in ESTIMATOR_METHODS
    if is_est:
        decision_rule, axes, keep_updating = ESTIMATOR_METHODS[method]
        if axes is None:
            axes = tuple(int(a) for a in args.adaptive_axes.split(","))
        est = Estimator(axes, args.sat_margin, window=args.window if keep_updating else None,
                        filter_mode="commanded" if method in CMD_FILTER_METHODS else args.adaptive_filter,
                        cmd_margin=args.cmd_margin)
    else:
        decision_rule, axes, keep_updating, est = None, None, False, None

    if method == "oracle":
        theta_name = "theta_pos" if true_dir_at(1) >= 0 else "theta_neg"
    elif method == "pos_only":
        theta_name = "theta_pos"
    else:
        theta_name = "neutral"

    theta_changes = [(0, theta_name)]
    decided = not is_est
    decision_step = decision_estimate = None
    chosen_dir = None
    forced = False

    sat_pos = np.zeros(3)
    sat_neg = np.zeros(3)
    pre_pos = np.zeros(3)
    pre_neg = np.zeros(3)

    obs, info = env.reset(seed=ep_seed)
    done = truncated = False
    steps = 0
    traj_mm = 0.0
    ep_return = 0.0

    while not (done or truncated):
        step = steps + 1
        if method == "oracle":
            want = "theta_pos" if true_dir_at(step) >= 0 else "theta_neg"
            if want != theta_name:
                theta_name = want
                theta_changes.append((steps, theta_name))
        scale_vec, bias_vec = vecs[theta_name]

        base_action, _ = base_model.predict(obs, deterministic=True)
        residual_action, _ = residual_model.predict(obs, deterministic=True)
        base_action = np.asarray(base_action, dtype=np.float32)
        residual_action = np.asarray(residual_action, dtype=np.float32)

        a_pre = base_action + args.residual_scale * scale_vec * residual_action + bias_vec
        action = a_pre.copy()
        action[3:6] += bias_at(step).astype(np.float32)
        if args.action_noise_std > 0:
            action += rng.normal(0.0, args.action_noise_std, size=action.shape).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        sat_pos += (action[3:6] >= 0.999)
        sat_neg += (action[3:6] <= -0.999)
        pre_pos += (a_pre[3:6] >= 0.999)
        pre_neg += (a_pre[3:6] <= -0.999)

        rot_before = get_rot(obs)
        obs, reward, done, truncated, info = env.step(action.copy())
        rot_after = get_rot(obs)

        steps += 1
        ep_return += float(reward)
        traj_mm += float(np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000))

        if is_est:
            if args.est_noise_deg > 0:
                rb = rot_before + np.deg2rad(rng.normal(0.0, args.est_noise_deg, size=3))
                ra = rot_after + np.deg2rad(rng.normal(0.0, args.est_noise_deg, size=3))
            else:
                rb, ra = rot_before, rot_after
            est.update(rb, ra, a_pre[3:6].astype(np.float64), angle_step)

            if not decided:
                n, m, t = est.global_stats(args.se_floor)
                go = False
                if decision_rule == "fixed":
                    go = steps >= args.warmup_steps and n > 0
                else:
                    if n >= args.min_samples and abs(t) >= args.z_thresh:
                        go = True
                    elif steps >= args.max_wait:
                        go, forced = True, True
                if go:
                    decided = True
                    decision_step = steps
                    decision_estimate = None if np.isnan(m) else m
                    if decision_estimate is None or decision_estimate == 0:
                        chosen_dir = 0.0          # no information: stay neutral
                    else:
                        chosen_dir = float(np.sign(decision_estimate))
                        theta_name = sign_name(decision_estimate)
                        theta_changes.append((steps, theta_name))
            elif keep_updating:
                n, m, t = est.window_stats(args.se_floor)
                if n >= args.min_samples and abs(t) >= args.z_thresh:
                    want = sign_name(m)
                    if want != theta_name:
                        theta_name = want
                        theta_changes.append((steps, theta_name))

    success = bool(info.get("is_success", False))
    final_true = true_dir_at(steps)

    def theta_matches(name, d):
        return (name == "theta_pos" and d > 0) or (name == "theta_neg" and d < 0)

    switch_latency = None
    if args.flip_step > 0 and steps > args.flip_step:
        new_dir = true_dir_at(args.flip_step + 1)
        current = theta_changes[0][1]
        # theta in force at the first post-flip step
        for s, name in theta_changes:
            if s <= args.flip_step:
                current = name
        if theta_matches(current, new_dir):
            switch_latency = 0
        else:
            for s, name in theta_changes:
                if s > args.flip_step and theta_matches(name, new_dir):
                    switch_latency = s - args.flip_step
                    break

    n_steps = max(steps, 1)
    rec = {
        "seed": int(ep_seed),
        "bias_deg": [float(x) for x in base_bias_deg],
        "success": success,
        "steps": int(steps),
        "traj_mm": float(traj_mm),
        "return": float(ep_return),
        "final_theta": theta_name,
        "theta_changes": theta_changes,
        "n_theta_changes": len(theta_changes) - 1,
        "true_dir": true_dir_at(1),
        "final_true_dir": final_true,
        "final_theta_correct": None if final_true == 0 else theta_matches(theta_name, final_true),
        "switch_latency": switch_latency,
        "flipped_in_episode": bool(args.flip_step > 0 and steps > args.flip_step),
        "sat_pos": (sat_pos / n_steps).tolist(),
        "sat_neg": (sat_neg / n_steps).tolist(),
        "pre_sat_pos": (pre_pos / n_steps).tolist(),
        "pre_sat_neg": (pre_neg / n_steps).tolist(),
    }
    if is_est:
        rec.update({
            "decision_step": decision_step,
            "decision_forced": forced,
            "decision_estimate_action": decision_estimate,
            "decision_estimate_deg": None if decision_estimate is None
            else float(np.rad2deg(decision_estimate * angle_step)),
            "chosen_dir": chosen_dir,
            "true_dir_at_decision": None if decision_step is None else true_dir_at(decision_step),
            "est_kept_rate": est.kept_rate(),
        })
    return rec


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(records):
    succ = np.array([r["success"] for r in records], dtype=float)
    steps = np.array([r["steps"] for r in records], dtype=float)
    s = {
        "episodes": len(records),
        "success_rate": float(succ.mean()),
        "mean_steps": float(steps.mean()),
        "mean_traj_mm": float(np.mean([r["traj_mm"] for r in records])),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "sat_pos": np.mean([r["sat_pos"] for r in records], axis=0).tolist(),
        "sat_neg": np.mean([r["sat_neg"] for r in records], axis=0).tolist(),
        "pre_sat_pos": np.mean([r["pre_sat_pos"] for r in records], axis=0).tolist(),
        "pre_sat_neg": np.mean([r["pre_sat_neg"] for r in records], axis=0).tolist(),
        "final_theta_accuracy": _mean([None if r["final_theta_correct"] is None else float(r["final_theta_correct"])
                                       for r in records]),
        "mean_theta_changes": _mean([r["n_theta_changes"] for r in records]),
    }
    flipped = [r for r in records if r["flipped_in_episode"]]
    if flipped:
        lat = [r["switch_latency"] for r in flipped]
        s.update({
            "flipped_episodes": len(flipped),
            "switched_rate": float(np.mean([x is not None for x in lat])),
            "mean_switch_latency": _mean(lat),
            "success_rate_flipped": float(np.mean([r["success"] for r in flipped])),
        })
    if "chosen_dir" in records[0]:
        dec = [r for r in records if r["decision_step"] is not None]
        correct = [float(r["chosen_dir"] == r["true_dir_at_decision"]) for r in dec
                   if r["true_dir_at_decision"] != 0]
        est_deg = [r["decision_estimate_deg"] for r in dec if r["decision_estimate_deg"] is not None]
        kept = np.array([[np.nan if v is None else v for v in r["est_kept_rate"]] for r in records], dtype=float)
        s.update({
            "decided_rate": len(dec) / len(records),
            "direction_accuracy": _mean(correct),
            "mean_decision_step": _mean([r["decision_step"] for r in dec]),
            "forced_rate": _mean([float(r["decision_forced"]) for r in dec]),
            "decision_estimate_deg_mean": _mean(est_deg),
            "decision_estimate_deg_std": float(np.std(est_deg)) if est_deg else None,
            "est_kept_rate": [None if np.all(np.isnan(kept[:, i])) else float(np.nanmean(kept[:, i]))
                              for i in range(3)],
        })
    return s


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", type=str, default="TD3_HER_BC", help="algorithm of the RESIDUAL model")
    p.add_argument("--task_name", type=str, default="Insert")
    p.add_argument("--reward_type", type=str, default="sparse")
    p.add_argument("--trans_error", type=float, default=0.5)
    p.add_argument("--angle_error", type=float, default=10)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=100)

    p.add_argument("--bias_mode", choices=["scalar", "random3d"], default="scalar")
    p.add_argument("--bias_list", type=str, default="-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5")
    p.add_argument("--random3d_low", type=float, default=-2.0)
    p.add_argument("--random3d_high", type=float, default=2.0)
    p.add_argument("--flip_step", type=int, default=0, help="bias changes sign after this step (0 = never)")

    p.add_argument("--methods", type=str, default="oracle,est_v1,est_seq,est_seq_pitch")

    p.add_argument("--warmup_steps", type=int, default=20, help="fixed rule")
    p.add_argument("--z_thresh", type=float, default=3.0, help="sequential rule")
    p.add_argument("--min_samples", type=int, default=10, help="sequential rule")
    p.add_argument("--max_wait", type=int, default=40, help="sequential rule: forced decision step")
    p.add_argument("--se_floor", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=10, help="adaptive: samples in the sliding window")
    p.add_argument("--adaptive_axes", type=str, default="0,1,2")
    p.add_argument("--sat_margin", type=float, default=0.98)
    p.add_argument("--cmd_margin", type=float, default=0.02, help="est_cmd*: keep sample if |a_pre| < 1 - cmd_margin")
    p.add_argument("--adaptive_filter", choices=["measured", "commanded"], default="measured",
                   help="sample filter used by est_v1/est_pitch/est_seq*/adaptive")
    p.add_argument("--est_noise_deg", type=float, default=0.0)
    p.add_argument("--action_noise_std", type=float, default=0.0)

    p.add_argument("--model_path", type=str, required=True, help="residual model")
    p.add_argument("--base_insert_model_path", type=str, required=True)
    p.add_argument("--base_insert_algo", type=str, default="TD3_HER_BC")
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--postprocess_mode", type=str, default="rot_scale_bias", choices=["rot_scale_bias"])
    p.add_argument("--positive_params_path", type=str, required=True)
    p.add_argument("--negative_params_path", type=str, required=True)
    p.add_argument("--scale_min", type=float, default=0.0)
    p.add_argument("--scale_max", type=float, default=2.0)
    p.add_argument("--bias_bound", type=float, default=1.0)
    p.add_argument("--save_json", type=str, required=True)
    p.add_argument("--save_episodes", type=int, default=1, help="1 = keep per-episode records in the json")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"unknown method {m}; choose from {ALL_METHODS}")

    theta_pos = load_theta(args.positive_params_path)
    theta_neg = load_theta(args.negative_params_path)

    # Seed blocks follow v1 (eval_seed + idx*1000 per scalar bias) so results with
    # the same bias_list order are matched with the 10/15 main experiment.
    if args.bias_mode == "scalar":
        conditions = [
            {"key": f"{b:g}", "fn": (lambda ep, b=b: [b, b, b]), "seed0": args.eval_seed + i * 1000}
            for i, b in enumerate(parse_float_list(args.bias_list))
        ]
    else:
        vecs = np.random.default_rng(args.eval_seed + 777).uniform(
            args.random3d_low, args.random3d_high, size=(args.episodes, 3))
        conditions = [{"key": "random3d", "fn": (lambda ep: vecs[ep].tolist()), "seed0": args.eval_seed}]

    args.angle_bias_deg = 0.0
    env, _, _, _ = setup_environment(args)
    residual_model = load_model(args.algorithm, args.model_path, env)
    base_model = load_model(args.base_insert_algo, args.base_insert_model_path, env)

    print("\n========== Estimated-Direction CEM v2 ==========")
    print(f"residual       : {args.algorithm}  scale={args.residual_scale}")
    print(f"bias_mode      : {args.bias_mode}   flip_step: {args.flip_step}")
    print(f"conditions     : {[c['key'] for c in conditions]}")
    print(f"methods        : {methods}")
    print(f"fixed warmup   : {args.warmup_steps}   seq z={args.z_thresh} min_n={args.min_samples} "
          f"max_wait={args.max_wait}   window={args.window}")
    print(f"est_noise_deg  : {args.est_noise_deg}")
    print("================================================\n", flush=True)

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"method": "estimated_direction_cem_v2", "config": vars(args),
              "theta_pos": theta_pos.tolist(), "theta_neg": theta_neg.tolist(), "results": {}}

    t0 = time.time()
    for cond in conditions:
        output["results"][cond["key"]] = {}
        for method in methods:
            rng = np.random.default_rng(cond["seed0"] + 12345)
            records = [run_episode(env, base_model, residual_model, method, cond["fn"](ep),
                                   theta_pos, theta_neg, args, cond["seed0"] + ep, rng)
                       for ep in range(args.episodes)]
            summ = summarize(records)
            entry = {"summary": summ}
            if args.save_episodes:
                entry["episodes"] = records
            output["results"][cond["key"]][method] = entry

            line = (f"bias={cond['key']:>8} | {method:>13} | success={summ['success_rate']*100:6.2f}% "
                    f"| steps={summ['mean_steps']:6.2f}")
            if "direction_accuracy" in summ:
                acc = summ["direction_accuracy"]
                line += f" | dir_acc={'n/a' if acc is None else f'{acc*100:.1f}%'}"
                if summ["mean_decision_step"] is not None:
                    line += f" | decide@{summ['mean_decision_step']:.1f}"
                if summ.get("decision_estimate_deg_mean") is not None:
                    line += f" | d_hat={summ['decision_estimate_deg_mean']:+.3f}deg"
            if "mean_switch_latency" in summ:
                lat = summ["mean_switch_latency"]
                line += (f" | switched={summ['switched_rate']*100:.0f}%"
                         f" | latency={'n/a' if lat is None else f'{lat:.1f}'}")
            print(line, flush=True)
            save_path.write_text(json.dumps(output, indent=1))

    print(f"\nElapsed: {(time.time() - t0) / 60:.1f} min")
    print("\n========== Summary (success %) ==========")
    print(f"{'bias':>8} | " + " | ".join(f"{m:>13}" for m in methods))
    for key, res in output["results"].items():
        print(f"{key:>8} | " + " | ".join(f"{res[m]['summary']['success_rate']*100:13.1f}" for m in methods))
    print(f"\nSaved to: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
