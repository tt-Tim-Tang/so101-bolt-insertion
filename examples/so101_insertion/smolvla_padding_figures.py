"""Plot the expert and paired padding rollouts saved by the Episode 20 audit."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = {
    "expert": dict(label="Expert replay", color="#596674", ls="--"),
    "native": dict(label="Original SmolVLA", color="#ae4e32", ls="-"),
    "corrected": dict(label="Padding-corrected", color="#08766f", ls="-"),
}


def read_rollout(folder):
    info = json.loads((folder / "rollout.json").read_text())
    with np.load(folder / "control.npz") as arrays:
        t = arrays["t_post_s"]
        depth = arrays["post_tip_depth"] * 1000
        wrist = np.linalg.norm(arrays["post_ee_pos"] - arrays["pre_ee_pos"], axis=1) * 1000
    return info, t, depth, wrist


def paired_panel(ax, pairs, fields, *, ylabel, title):
    labels = []
    ticks = []
    for index, (name, *field) in enumerate(fields):
        x = 3 * index + np.arange(2)
        modes = ("native", "corrected")
        columns = []
        for mode in modes:
            values = []
            for pair in pairs:
                value = pair[mode]["phases"]["all"]
                for key in field:
                    value = value[key]
                values.append(value)
            columns.append(values)
        for y in zip(*columns, strict=True):
            ax.plot(x, y, color="#7b8b9a", alpha=.6, lw=1.1, zorder=1)
        for position, values, mode in zip(x, columns, modes, strict=True):
            color = STYLES[mode]["color"]
            ax.scatter(np.full(len(values), position), values, facecolors="white", edgecolors=color, s=24, zorder=2)
            median = np.median(values)
            ax.scatter(position, median, marker="D", color=color, s=28, zorder=3)
            ax.annotate(f"{median:.2f}" if len(fields) == 1 else f"{median:.0f}", (position, median),
                        xytext=(7, -3 if mode == "corrected" else 3), textcoords="offset points", color=color,
                        fontsize=9)
        ticks.extend(x.tolist())
        prefix = f"{name}\n" if name else ""
        labels.extend([f"{prefix}Original", f"{prefix}Corrected"])
    ax.set(title=title, ylabel=ylabel, xticks=ticks, xticklabels=labels,
           xlim=(-.45, ticks[-1] + .58), ylim=(0, ax.get_ylim()[1] * 1.1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, help="Archive directory with native and corrected per-seed controls")
    parser.add_argument("expert", type=Path)
    parser.add_argument("ab_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pairs = json.loads(args.ab_summary.read_text())["models"]["multi-20k"]["closed_loop"]
    seed = pairs[0]["seed"]
    runs = {"expert": read_rollout(args.expert)}
    runs.update({mode: read_rollout(args.rollouts / f"{mode}-{seed}") for mode in ("native", "corrected")})
    if not all(info["episode"] == 20 and info["control_hz"] == 25 for info, *_ in runs.values()):
        raise ValueError("Expected the matched Episode 20 replays at 25 Hz")
    if len({json.dumps(info["initial"], sort_keys=True) for info, *_ in runs.values()}) != 1:
        raise ValueError("Expert and policy initial states differ")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3))
    fig.subplots_adjust(top=.84, bottom=.14, left=.08, right=.97, wspace=.26, hspace=.46)
    fig.suptitle("SmolVLA 20k · motion before and after the padding correction", x=.08, ha="left",
                 y=.98, fontsize=16, fontweight="semibold")
    fig.text(.08, .928, f"SO-101 bolt insertion · Episode 20 · same initial state and noise for the policy pair (seed {seed})",
             fontsize=10, color="#475569")

    for mode, (info, time, depth, wrist) in runs.items():
        style = STYLES[mode]
        contact = info["events"]["target_contact"]["time_s"]
        axes[0, 0].plot(time, wrist, lw=1.05, **style)
        axes[0, 0].axvline(contact, lw=.7, color=style["color"], alpha=.4, ls=":")
        mask = time >= contact - .35
        axes[0, 1].plot(time[mask] - contact, depth[mask], lw=1.45, **style)
        axes[0, 1].scatter(time[-1] - contact, depth[-1], marker="o" if info["end"]["success"] else "x",
                           color=style["color"], s=28, zorder=3)
    axes[0, 0].set(title="Wrist displacement per 40 ms step", xlabel="Time from reset (s)", ylabel="mm")
    axes[0, 1].axvline(0, color="#64748b", ls=":", lw=.8)
    axes[0, 1].set(title="Insertion depth around first contact", xlabel="Time from first bolt–fixture contact (s)",
                   ylabel="Depth below hole mouth (mm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.073, .914), ncol=3, frameon=False, fontsize=9)

    paired_panel(axes[1, 0], pairs, (("", "wrist_step_mm", "p95"),),
                 ylabel="mm per 40 ms step", title="Wrist displacement P95 · all three seeds")
    paired_panel(axes[1, 1], pairs, (("Within", "within_chunk_target_jerk", "rms"),
                                     ("Across", "cross_chunk_target_jerk", "rms")),
                 ylabel="rad/s³", title="Five-axis target joint jerk RMS · all three seeds")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.24)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    fig.text(.08, .045, "Depth endpoints: ● inserted, × failed. Bottom panels: circles are individual seeds; diamonds are medians.",
             fontsize=9, color="#475569")
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"episode20-padding-comparison.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
