"""
viz_ex_overfitting.py

Parses an EX training log and plots training loss vs validation metrics per
epoch, to visually show the overfitting: training loss keeps dropping while
validation combined score plateaus/declines after its peak.

Output: <log-stem>_overfitting_viz.png (or --out to override)

Run (on pod):
    python viz_ex_overfitting.py --log /workspace/train_ex_seed123.log
"""
import argparse
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--log", default="/workspace/train_ex.log")
p.add_argument("--out", default=None)
args = p.parse_args()

LOG = Path(args.log)
OUT = Path(args.out) if args.out else LOG.with_name(LOG.stem + "_overfitting_viz.png")

# Epoch NNN | loss=X.XXXX [| recall=X.XXXX (h/g) | preservation=X.XXXX | combined=X.XXXX]
LOSS_RE = re.compile(r"Epoch (\d+) \| loss=([\d.]+)")
VAL_RE  = re.compile(
    r"Epoch (\d+) \| loss=([\d.]+) \| recall=([\d.]+) \(\d+/\d+\) "
    r"\| preservation=([\d.]+) \| combined=([\d.]+)"
)

train_epochs, train_losses = [], []
val_epochs, val_recall, val_pres, val_combined = [], [], [], []

with open(LOG, "r", errors="ignore") as f:
    for line in f:
        vm = VAL_RE.search(line)
        if vm:
            e = int(vm.group(1))
            val_epochs.append(e)
            val_recall.append(float(vm.group(3)))
            val_pres.append(float(vm.group(4)))
            val_combined.append(float(vm.group(5)))
            train_epochs.append(e)
            train_losses.append(float(vm.group(2)))
            continue
        lm = LOSS_RE.search(line)
        if lm:
            e = int(lm.group(1))
            if e not in train_epochs:
                train_epochs.append(e)
                train_losses.append(float(lm.group(2)))

# Sort by epoch (log lines can arrive slightly out of order due to buffering)
order = sorted(range(len(train_epochs)), key=lambda i: train_epochs[i])
train_epochs = [train_epochs[i] for i in order]
train_losses = [train_losses[i] for i in order]

best_idx = max(range(len(val_combined)), key=lambda i: val_combined[i]) if val_combined else None

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(train_epochs, train_losses, color="tab:red", label="Train loss", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss", color="tab:red")
ax1.tick_params(axis='y', labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(val_epochs, val_combined, color="tab:blue", marker="o", label="Val combined (recall x preservation)")
ax2.plot(val_epochs, val_recall, color="tab:green", marker="s", linestyle="--", alpha=0.6, label="Val recall")
ax2.plot(val_epochs, val_pres, color="tab:purple", marker="^", linestyle="--", alpha=0.6, label="Val preservation")
ax2.set_ylabel("Validation metric")

if best_idx is not None:
    be = val_epochs[best_idx]
    bc = val_combined[best_idx]
    ax2.axvline(be, color="gray", linestyle=":", alpha=0.7)
    ax2.annotate(f"best epoch {be}\ncombined={bc:.4f}",
                 xy=(be, bc), xytext=(be + 3, bc + 0.05),
                 fontsize=9, color="black")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

plt.title("EX Learned Detector: Training Loss vs Validation Metrics\n"
          "(train loss keeps dropping while validation plateaus after best epoch = overfitting)")
plt.tight_layout()
plt.savefig(str(OUT), dpi=110, bbox_inches="tight")
print(f"Saved -> {OUT}")
print(f"Best validation epoch: {val_epochs[best_idx] if best_idx is not None else 'N/A'}")
