"""
Plots for the estimated-direction CEM experiment.

Usage:
    python3 plot_estimated_direction.py --main  evolution_results/estimated_direction/main.json \
                                        --noise_dir evolution_results/estimated_direction/noise \
                                        --random3d evolution_results/estimated_direction/random3d.json \
                                        --out_dir figures/estimated_direction
Any argument may be omitted; the matching figures are skipped.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical order (blue, orange, aqua, yellow); text stays in ink colours.
COLORS = {
    "oracle": "#2a78d6",
    "estimated": "#eb6834",
    "residual_only": "#1baf7a",
    "pos_only": "#eda100",
}
LABELS = {
    "oracle": "Oracle 方向（已知）",
    "estimated": "估計方向（本次）",
    "residual_only": "Residual Only",
    "pos_only": "只用 θ_pos",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_fonts():
    for f in ["Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "PingFang TC",
              "Noto Sans CJK JP", "WenQuanYi Zen Hei", "AR PL UMing TW"]:
        try:
            from matplotlib import font_manager
            if any(f in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = ["DejaVu Sans", f]
                break
        except Exception:
            pass
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 11, "figure.dpi": 150, "axes.axisbelow": True,
    })


def bias_keys_sorted(results):
    keys = [k for k in results if k != "random3d"]
    return sorted(keys, key=float)


def fig_success_vs_bias(data, out):
    res = data["results"]
    keys = bias_keys_sorted(res)
    x = np.array([float(k) for k in keys])
    # oracle drawn last (dashed, hollow markers) so it stays visible when it overlaps "estimated"
    methods = [m for m in ["pos_only", "residual_only", "estimated", "oracle"] if m in res[keys[0]]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in methods:
        sr = [res[k][m]["summary"]["success_rate"] * 100 for k in keys]
        st = [res[k][m]["summary"]["mean_steps"] for k in keys]
        kw = dict(color=COLORS[m], marker="o", label=LABELS[m])
        if m == "oracle":
            kw.update(ls="--", lw=1.8, ms=9, mfc="white", mew=1.8)
        elif m == "estimated":
            kw.update(ls="-", lw=2.6, ms=7)
        else:
            kw.update(ls="-", lw=1.8, ms=6)
        axes[0].plot(x, sr, **kw)
        axes[1].plot(x, st, **kw)
    axes[0].set_ylim(-3, 103)
    axes[0].set_ylabel("成功率 (%)")
    axes[1].set_ylabel("平均步數")
    for ax in axes:
        ax.set_xlabel("旋轉動作偏差 d (deg)")
        ax.set_xticks(x)
        ax.axvline(0, color=INK2, lw=0.8, alpha=0.5)
    axes[0].set_title("成功率：估計方向 vs 已知方向", color=INK)
    axes[1].set_title("控制效率", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "est_dir_success_steps.png", bbox_inches="tight")
    plt.close(fig)


def fig_estimator_quality(data, out):
    res = data["results"]
    keys = bias_keys_sorted(res)
    if "estimated" not in res[keys[0]]:
        return
    true_deg, est_mean, est_std, acc, dstep = [], [], [], [], []
    for k in keys:
        s = res[k]["estimated"]["summary"]
        true_deg.append(float(k))
        est_mean.append(s["decision_estimate_deg_mean"])
        est_std.append(s["decision_estimate_deg_std"] or 0.0)
        acc.append(None if s["direction_accuracy"] is None else s["direction_accuracy"] * 100)
        dstep.append(s["mean_decision_step"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    lim = max(abs(min(true_deg)), abs(max(true_deg))) + 0.5
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls=":", label="y = x")
    ax.errorbar(true_deg, est_mean, yerr=est_std, fmt="o", color=COLORS["estimated"],
                ms=7, capsize=4, lw=1.8, label="決策當下的 d̂（平均 ± 標準差）")
    ax.set_xlabel("真實偏差 d (deg)")
    ax.set_ylabel("估計偏差 d̂ (deg)")
    ax.set_title("估計值 vs 真實值", color=INK)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    xs = [t for t, a in zip(true_deg, acc) if a is not None]
    ys = [a for a in acc if a is not None]
    ax.bar(xs, ys, width=0.35, color=COLORS["estimated"])
    for xv, yv in zip(xs, ys):
        ax.text(xv, yv + 1.5, f"{yv:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, 110)
    ax.set_xticks(true_deg)
    ax.set_xlabel("真實偏差 d (deg)")
    ax.set_ylabel("方向判斷正確率 (%)")
    ms = np.nanmean([d for d in dstep if d is not None]) if any(dstep) else float("nan")
    ax.set_title(f"方向判斷正確率（平均第 {ms:.1f} 步做決定）", color=INK)
    fig.tight_layout()
    fig.savefig(out / "est_dir_estimator_quality.png", bbox_inches="tight")
    plt.close(fig)


def fig_noise(noise_dir, out):
    files = sorted(glob.glob(os.path.join(noise_dir, "*.json")))
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text())
        cfg = d["config"]
        for k, r in d["results"].items():
            if "estimated" not in r:
                continue
            s = r["estimated"]["summary"]
            rows.append((cfg["est_noise_deg"], cfg["warmup_steps"], float(k),
                         s["direction_accuracy"], s["success_rate"], s["decided_rate"]))
    if not rows:
        return
    noises = sorted({r[0] for r in rows})
    warmups = sorted({r[1] for r in rows})
    biases = sorted({r[2] for r in rows if r[2] != 0}, key=lambda b: (abs(b), b))

    fig, axes = plt.subplots(1, len(noises), figsize=(4.4 * len(noises), 4.0), squeeze=False)
    seq = ["#9ec5f4", "#5b9de9", "#2a78d6", "#1a56a3"]  # one hue, light -> dark = longer warm-up
    for ax, nz in zip(axes[0], noises):
        width = 0.8 / max(len(warmups), 1)
        for j, w in enumerate(warmups):
            vals = []
            for b in biases:
                m = [r[3] for r in rows if r[0] == nz and r[1] == w and r[2] == b and r[3] is not None]
                vals.append(np.mean(m) * 100 if m else np.nan)
            xs = np.arange(len(biases)) + (j - (len(warmups) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.92, color=seq[min(j, len(seq) - 1)], label=f"warm-up {w} 步")
            for xv, b, v in zip(xs, biases, vals):
                dec = [r[5] for r in rows if r[0] == nz and r[1] == w and r[2] == b]
                if np.isnan(v):
                    ax.text(xv, 2, "未決定", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
                elif dec and np.mean(dec) < 0.999:
                    ax.text(xv, v + 1, f"{np.mean(dec)*100:.0f}%決定", ha="center", va="bottom",
                            fontsize=7, color=INK2, rotation=90)
        ax.set_xticks(np.arange(len(biases)))
        ax.set_xticklabels([f"{b:+g}°" for b in biases])
        ax.set_ylim(0, 105)
        ax.set_title(f"姿態讀值雜訊 σ = {nz}°", color=INK)
        ax.set_ylabel("方向判斷正確率 (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=len(warmups), bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "est_dir_noise_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def print_table(data, title):
    res = data["results"]
    keys = bias_keys_sorted(res) if "random3d" not in res else ["random3d"]
    methods = list(res[keys[0]].keys())
    print(f"\n== {title} ==")
    print(f"{'bias':>8} | " + " | ".join(f"{m:>13}" for m in methods))
    for k in keys:
        print(f"{k:>8} | " + " | ".join(f"{res[k][m]['summary']['success_rate']*100:13.1f}" for m in methods))
        if "estimated" in res[k]:
            s = res[k]["estimated"]["summary"]
            acc = s["direction_accuracy"]
            print(f"{'':>8}   est: dir_acc={'n/a' if acc is None else f'{acc*100:.1f}%'}, "
                  f"decide@{s['mean_decision_step']}, d̂={s['decision_estimate_deg_mean']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=str, default=None)
    p.add_argument("--noise_dir", type=str, default=None)
    p.add_argument("--random3d", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="figures/estimated_direction")
    a = p.parse_args()

    setup_fonts()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.main:
        d = json.loads(Path(a.main).read_text())
        fig_success_vs_bias(d, out)
        fig_estimator_quality(d, out)
        print_table(d, "Scalar bias")
    if a.noise_dir:
        fig_noise(a.noise_dir, out)
    if a.random3d:
        print_table(json.loads(Path(a.random3d).read_text()), "Random 3-DOF")
    print(f"\nFigures saved to {out.resolve()}")


if __name__ == "__main__":
    main()
