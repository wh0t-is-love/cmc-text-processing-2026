#!/usr/bin/env python3
"""График обучения: loss базового прогона + сравнение экспериментов.

Usage: python -I plot_history.py <repo_root>
Выход: reports/training_curve.png
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Палитра: референсные слоты 1-2 (blue/orange), один оттенок для столбцов.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

root = sys.argv[1]
out_dir = os.path.join(root, "reports")
os.makedirs(out_dir, exist_ok=True)


def read_log(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


base = read_log(os.path.join(root, "checkpoints/base/log.jsonl"))
steps = [r["step"] for r in base]
train = [r["train_loss"] for r in base]
val = [r["val_loss"] for r in base]

exps = []
for name in ["v4096", "v8192", "v16384", "swa128"]:
    r = read_log(os.path.join(root, f"checkpoints/{name}/log.jsonl"))[-1]
    exps.append((name, r["val_loss"], r["ppl"]))
exps.sort(key=lambda t: t[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)

# --- панель 1: кривые обучения base ---
ax1.plot(steps, train, color=BLUE, lw=2, label="train loss")
ax1.plot(steps, val, color=ORANGE, lw=2, label="val loss")
# прямые подписи на концах линий (текст — чернилами, не цветом серии)
ax1.annotate(f"{train[-1]:.2f}", xy=(steps[-1], train[-1]),
             xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK2)
ax1.annotate(f"{val[-1]:.2f}", xy=(steps[-1], val[-1]),
             xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK2)
best_i = min(range(len(val)), key=lambda i: val[i])
ax1.plot([steps[best_i]], [val[best_i]], "o", color=ORANGE, ms=6)
ax1.annotate(f"best val {val[best_i]:.3f} (ppl {math.exp(val[best_i]):.0f})",
             xy=(steps[best_i], val[best_i]), xytext=(-8, 16),
             textcoords="offset points", ha="right", fontsize=8, color=INK2)
ax1.set_title("Base-модель (словарь 8192): 9 295 шагов, 5 эпох",
              color=INK, fontsize=10, loc="left")
ax1.set_xlabel("шаг", color=INK2, fontsize=9)
ax1.set_ylabel("cross-entropy loss", color=INK2, fontsize=9)
ax1.grid(axis="y", color=GRID, lw=0.8)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

# --- панель 2: эксперименты @1200 шагов ---
names = [e[0] for e in exps]
losses = [e[1] for e in exps]
ppls = [e[2] for e in exps]
bars = ax2.bar(names, losses, width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
for rect, lv, pv in zip(bars, losses, ppls):
    ax2.annotate(f"{lv:.2f}\nppl {pv:.0f}",
                 xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color=INK2)
ax2.set_title("Короткие прогоны (1 200 шагов): val loss",
              color=INK, fontsize=10, loc="left")
ax2.set_ylabel("val loss", color=INK2, fontsize=9)
ax2.grid(axis="y", color=GRID, lw=0.8)
ax2.set_axisbelow(True)
ax2.set_ylim(0, max(losses) * 1.25)
ax2.tick_params(axis="x", labelsize=9)
for lbl in ax2.get_xticklabels():
    lbl.set_color(INK2)

fig.tight_layout()
out = os.path.join(out_dir, "training_curve.png")
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
