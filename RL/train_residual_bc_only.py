"""
Residual BC-only baseline (exp. 4).

Trains a residual policy with ONLY behaviour cloning on the same residual
targets the TD3_HER_BC residual used:

    a_res_target = clip((a_expert - a_base) / alpha_train, -1, 1)

- same expert data   : Expert_traj/Insert/all_episodes_merged.pkl
- same base policy   : Insert/TD3_HER_BC/sparse/seed_0/final_model.zip
- same alpha_train   : 0.1
- same network       : actor 4 x 256 (DEFAULT_NET_ARCH), lr 3e-4, batch 512
- NO environment interaction, NO RL loss, NO HER.

Because BC never needs a rollout, training is a plain loop of gradient steps
on the demo set (minutes on GPU). AMBF must still be running, because the
residual env is constructed to load the frozen base policy and to convert the
expert actions into residual targets.

Output: insert_residual/BC/sparse/seed_<seed>/final_model.zip
        (+ checkpoints and a loss log). Load it with --algorithm BC.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

import RL_training_online_residual as tr
from RL_algo.BC import BC

DEFAULT_NET_ARCH = dict(pi=[256, 256, 256, 256], qf=[256, 256, 256, 256])


def flatten_transitions(data):
    return [t for t in tr._iter_transition_containers(data) if "obs" in t and "action" in t]


def to_tensors(transitions, device):
    obs = {k: th.tensor(np.array([t["obs"][k] for t in transitions]), dtype=th.float32, device=device)
           for k in transitions[0]["obs"].keys()}
    act = th.tensor(np.array([t["action"] for t in transitions]), dtype=th.float32, device=device)
    return obs, act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual_scale", type=float, default=0.1, help="alpha_train (target scale)")
    p.add_argument("--base_insert_model_path", type=str,
                   default=os.path.join(tr.Base_directory, "Insert/TD3_HER_BC/sparse/seed_0/final_model.zip"))
    p.add_argument("--base_insert_algo", type=str, default="TD3_HER_BC")
    p.add_argument("--gradient_steps", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=25000)
    p.add_argument("--trans_error", type=float, default=0.5)
    p.add_argument("--angle_error", type=float, default=10)
    args = p.parse_args()

    set_random_seed(args.seed)

    # Build the residual env exactly like RL_training_online_residual.py does.
    env_args = argparse.Namespace(
        algorithm="BC", task_name="insert_residual", reward_type="sparse", seed=args.seed,
        trans_error=args.trans_error, angle_error=args.angle_error,
        base_insert_model_path=args.base_insert_model_path, base_insert_algo=args.base_insert_algo,
        residual_scale=args.residual_scale, enable_residual_jaw=False,
    )
    env, _, _, _ = tr.setup_environment(env_args)

    expert = None
    for name in ["Insert", "insert"]:          # folder on disk is Expert_traj/Insert
        if os.path.exists(os.path.join(tr.Base_directory, "Expert_traj", name, "all_episodes_merged.pkl")):
            expert = tr.load_expert_data(name)
            break
    if expert is None:
        raise FileNotFoundError("Expert_traj/Insert/all_episodes_merged.pkl not found")
    expert = tr.transform_expert_actions_to_residual(expert, env)
    transitions = flatten_transitions(expert)
    print(f"Residual-target transitions: {len(transitions)}")

    acts = np.array([t["action"] for t in transitions])
    clipped = np.mean(np.abs(acts[:, :6]) >= 0.999, axis=0)
    print("Fraction of residual targets clipped at +-1 per axis [x y z roll pitch yaw]:",
          np.round(clipped, 3))

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(transitions))
    n_val = max(1, int(len(transitions) * args.val_frac))
    val_t = [transitions[i] for i in idx[:n_val]]
    train_t = [transitions[i] for i in idx[n_val:]]

    out_dir = os.path.join(tr.Base_directory, "insert_residual", "BC", "sparse", f"seed_{args.seed}")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    model = BC(policy="MultiInputPolicy", env=env, learning_rate=args.learning_rate,
               batch_size=args.batch_size, verbose=0, seed=args.seed,
               policy_kwargs=dict(net_arch=DEFAULT_NET_ARCH), episode_transitions=train_t)
    model.set_logger(configure(os.path.join(out_dir, "sb3_log"), ["csv"]))
    device = model.device
    val_obs, val_act = to_tensors(val_t, device)

    def val_loss():
        model.policy.set_training_mode(False)
        with th.no_grad():
            return float(F.mse_loss(model.actor(val_obs), val_act).item())

    log_path = os.path.join(out_dir, "bc_loss.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gradient_step", "train_mse", "val_mse"])
        t0 = time.time()
        done = 0
        print(f"step 0 | val_mse={val_loss():.5f}")
        while done < args.gradient_steps:
            n = min(args.log_every, args.gradient_steps - done)
            model.train(gradient_steps=n, batch_size=args.batch_size)
            done += n
            train_mse = float(model.logger.name_to_value.get("train/actor_loss", np.nan))
            vm = val_loss()
            w.writerow([done, train_mse, vm])
            f.flush()
            model.logger.dump(done)
            if done % (args.log_every * 10) == 0 or done == args.gradient_steps:
                print(f"step {done:>7} | train_mse={train_mse:.5f} | val_mse={vm:.5f} "
                      f"| {time.time() - t0:.0f}s", flush=True)
            if done % args.ckpt_every == 0:
                model.save(os.path.join(out_dir, "checkpoints", f"bc_{done}_steps"))

    model.save(os.path.join(out_dir, "final_model"))
    print(f"\nSaved: {os.path.join(out_dir, 'final_model.zip')}")
    print(f"Loss log: {log_path}")
    try:
        env.close()
    except RuntimeError as e:   # AMBF publisher thread race on shutdown; model is already saved
        print(f"(ignored on env.close: {e})")


if __name__ == "__main__":
    main()
