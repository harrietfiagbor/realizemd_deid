"""
viz_ex_seed_comparison.py

Consolidated 3-seed comparison: precision and recall at full-res, 256px maxpool,
and 512px maxpool, for all three EX seed checkpoints (42, 123, 7). Shows the
core resolution finding (256px ~= 512px precision) is seed-stable, while
absolute precision/recall levels vary seed to seed (optic-disc contamination).

Numbers are hardcoded from the completed eval runs -- this is a summary/report
chart, not a live eval.

Output: EX_seed_comparison_viz.png

Run (on pod or locally, no GPU needed):
    python viz_ex_seed_comparison.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--out", default="/workspace/EX_seed_comparison_viz.png")
p.add_argument("--drive_dir", default=None)
args = p.parse_args()

# Testing set, maxpool mode, macro-average precision -- from completed eval runs
seeds = ["Seed 42", "Seed 123", "Seed 7"]
od_contamination = [1.1, 2.3, 14.6]  # % of FP on optic disc

recall = {
    "Full-res":  [0.764, 0.757, 0.764],
    "256px":     [0.774, 0.768, 0.797],
    "512px":     [0.795, 0.786, 0.824],
}
precision = {
    "Full-res":  [0.655, 0.659, 0.621],
    "256px":     [0.666, 0.684, 0.642],
    "512px":     [0.666, 0.676, 0.638],
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

x = np.arange(len(seeds))
width = 0.25
colors = {"Full-res": "#888888", "256px": "#4C72B0", "512px": "#DD8452"}

# Panel 1: Recall
ax = axes[0]
for i, (label, vals) in enumerate(recall.items()):
    ax.bar(x + (i-1)*width, vals, width, label=label, color=colors[label])
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel("Recall@IoU=0.3")
ax.set_title("Recall by seed and resolution\n(maxpool, testing set)")
ax.legend()
ax.set_ylim(0, 1)

# Panel 2: Precision
ax = axes[1]
for i, (label, vals) in enumerate(precision.items()):
    ax.bar(x + (i-1)*width, vals, width, label=label, color=colors[label])
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel("Precision (macro-average)")
ax.set_title("Precision by seed and resolution\n(maxpool, testing set)\n"
              "Note: 256px ~= 512px within each seed -- the resolution\n"
              "finding is stable. Absolute level varies by seed.")
ax.legend()
ax.set_ylim(0, 1)

# Panel 3: OD contamination vs precision drop
ax = axes[2]
ax2 = ax.twinx()
bars = ax.bar(x, od_contamination, width=0.5, color="#C44E52", alpha=0.7, label="% FP on optic disc")
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel("% of false positives on optic disc", color="#C44E52")
ax.tick_params(axis='y', labelcolor="#C44E52")
ax.set_title("Optic-disc contamination vs precision\n(explains most of the seed-to-seed spread)")

fullres_pre = precision["Full-res"]
ax2.plot(x, fullres_pre, color="black", marker="o", linewidth=2, label="Full-res precision")
ax2.set_ylabel("Full-res precision")
ax2.set_ylim(0.5, 0.75)

for i, v in enumerate(od_contamination):
    ax.text(i, v + 0.3, f"{v}%", ha="center", fontsize=10)

fig.suptitle("EX Learned Detector: 3-Seed Stability Check\n"
             "Resolution finding (256px viable) is seed-robust; absolute precision is not, "
             "and tracks optic-disc contamination",
             fontsize=13, fontweight="bold", y=1.05)

plt.tight_layout()
plt.savefig(args.out, dpi=110, bbox_inches="tight")
print(f"Saved -> {args.out}")

if args.drive_dir:
    import subprocess as _sp
    r = _sp.run(["rclone", "copy", args.out, args.drive_dir], capture_output=True, text=True)
    print("Drive backup:", "OK" if r.returncode == 0 else r.stderr.strip())
