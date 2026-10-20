import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Model_evaluation import setup_environment
from evolve_residual_postprocess import load_model, rollout_score


# =========================================================
# Experiment config
# =========================================================

BIAS_LIST = [-2.0, -1.5, -1.0, -0.5]

# 建議跑 3 個 seed
SEEDS = [0, 100, 200]

EPISODES = 100

THETA_POS = np.array([
    0.94038433,
    0.97939765,
    1.10951865,
   -0.48590052,
   -0.24643701,
   -0.92954612
], dtype=np.float32)


BASE_MODEL = (
    "/home/suzixun/SurgicAI/RL/"
    "Insert/TD3_HER_BC/sparse/seed_0/final_model.zip"
)

RESIDUAL_MODEL = (
    "/home/suzixun/SurgicAI/RL/"
    "insert_residual/TD3_HER_BC/sparse/seed_0/final_model.zip"
)

OUTPUT_PATH = Path(
    "/home/suzixun/SurgicAI/RL/"
    "evolution_results/negative_multiseed_eval.json"
)


# =========================================================
# Args required by environment / rollout_score
# =========================================================

args = SimpleNamespace(
    algorithm="TD3_HER_BC",
    task_name="Insert",
    reward_type="sparse",

    trans_error=0.5,
    angle_error=10,

    eval_seed=0,

    action_noise_std=0.0,
    angle_bias_deg=0.0,

    residual_scale=1.0,

    postprocess_mode="rot_scale_bias",
    scale_min=0.0,
    scale_max=2.0,
    bias_bound=1.0,
)


# =========================================================
# Setup
# =========================================================

env, step_size, threshold, max_episode_steps = setup_environment(args)

residual_model = load_model(
    args.algorithm,
    RESIDUAL_MODEL,
    env,
)

base_model = load_model(
    "TD3_HER_BC",
    BASE_MODEL,
    env,
)


all_results = {}


# =========================================================
# Evaluation
# =========================================================

for seed in SEEDS:

    print("\n")
    print("==============================================")
    print(f" Eval Seed = {seed}")
    print("==============================================")

    seed_results = {}

    for bias in BIAS_LIST:

        args.angle_bias_deg = bias

        result = rollout_score(
            env=env,
            base_model=base_model,
            residual_model=residual_model,
            theta=THETA_POS,
            args=args,
            eval_seed=seed,
            episodes=EPISODES,
            verbose=False,
        )

        success_pct = result["success_rate"] * 100.0

        seed_results[str(bias)] = {
            "success_rate": success_pct,
            "mean_steps": result["mean_steps"],
            "mean_traj_mm": result["mean_traj_mm"],
            "mean_return": result["mean_return"],
        }

        print(
            f"bias={bias:+.1f} | "
            f"success={success_pct:6.2f}% | "
            f"steps={result['mean_steps']:7.2f} | "
            f"traj={result['mean_traj_mm']:7.2f} mm"
        )

    all_results[str(seed)] = seed_results


# =========================================================
# Aggregate statistics across seeds
# =========================================================

summary = {}

print("\n")
print("==============================================================")
print(" Aggregate Statistics Across Seeds")
print("==============================================================")

for bias in BIAS_LIST:

    key = str(bias)

    successes = np.array([
        all_results[str(seed)][key]["success_rate"]
        for seed in SEEDS
    ])

    steps = np.array([
        all_results[str(seed)][key]["mean_steps"]
        for seed in SEEDS
    ])

    traj = np.array([
        all_results[str(seed)][key]["mean_traj_mm"]
        for seed in SEEDS
    ])

    summary[key] = {
        "success_mean": float(np.mean(successes)),
        "success_min": float(np.min(successes)),
        "success_max": float(np.max(successes)),

        "steps_mean": float(np.mean(steps)),
        "steps_min": float(np.min(steps)),
        "steps_max": float(np.max(steps)),

        "traj_mean": float(np.mean(traj)),
        "traj_min": float(np.min(traj)),
        "traj_max": float(np.max(traj)),
    }

    print(
        f"bias={bias:+.1f} | "
        f"success mean={np.mean(successes):6.2f}% "
        f"[{np.min(successes):.2f}, {np.max(successes):.2f}] | "
        f"steps mean={np.mean(steps):7.2f} "
        f"[{np.min(steps):.2f}, {np.max(steps):.2f}]"
    )


# =========================================================
# Save JSON
# =========================================================

output = {
    "bias_list": BIAS_LIST,
    "seeds": SEEDS,
    "episodes_per_bias": EPISODES,
    "theta_pos": THETA_POS.tolist(),
    "per_seed_results": all_results,
    "summary": summary,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(output, indent=2)
)

print("\nSaved to:")
print(OUTPUT_PATH)

env.close()
