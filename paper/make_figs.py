"""Generate the two report figures as vector PDFs next to this script.

Data are transcribed verbatim from the committed artifacts:
  - dose curves: reports/selection/{attribution,dpo}_mistral_lora.json (dev-set ASR sweep)
  - bar chart: reports/safestack_report.md section 6.0 headline table (locked-test ASR)
No numbers are computed here beyond plotting.

Run from anywhere: `uv run --with matplotlib --with numpy python paper/make_figs.py`.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # non-interactive; write files, never open a window
plt.rcParams.update({
    "font.family": "serif",  # sit closer to the paper's Latin Modern body text
    "font.size": 9,
    "axes.linewidth": 0.8,
    "hatch.linewidth": 0.7,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe categorical palette
BLUE = "#0072B2"
VERM = "#D55E00"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#8a8a8a"
INK = "#333333"

# ---- Figure 1: dose-response (H7), dev-set ASR vs continue-training dose ----
# Equal-spaced ordinal x: the dose grid is a discrete set of experimental settings,
# not a linear sweep, so even spacing reads the curve shape honestly (ticks show true doses).
doses = [10, 50, 100, 250, 411]
pos = list(range(len(doses)))
sft = [0.04, 0.25, 0.60, 0.83, 0.83]      # attribution (SFT-on-chosen), LLM-LAT
dpo = [0.04, 0.04, 0.05, 0.05, 0.04]      # DPO, LLM-LAT
base = 0.04

fig, ax = plt.subplots(figsize=(5.9, 3.6))

# aligned-floor reference (C5): DPO essentially never leaves it
ax.axhline(base, color=GREY, ls=(0, (1, 1.6)), lw=1.1, zorder=1)
ax.text(-0.28, base + 0.03, "aligned floor (C5) = 0.04", fontsize=7.5, color=GREY,
        va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

# the whole H7 point: same data, same dose, different objective -> different outcome
ax.fill_between(pos, dpo, sft, color=BLUE, alpha=0.06, zorder=1)
ax.annotate("", xy=(3, 0.83), xytext=(3, 0.07),
            arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.1))
ax.text(3.12, 0.45, "same data,\ndifferent\nobjective", fontsize=7.5, color="#555555", va="center")

ax.plot(pos, sft, "-o", color=BLUE, lw=2, ms=6, mfc=BLUE, mec="white", mew=0.8, zorder=3)
ax.plot(pos, dpo, "-s", color=VERM, lw=2, ms=6, mfc=VERM, mec="white", mew=0.8, zorder=3)

ax.annotate("first rise\nby dose 50", xy=(1, 0.25), xytext=(1.2, 0.52),
            fontsize=8, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
for xi, v in [(2, 0.60), (4, 0.83)]:  # selective labels on the SFT rise + plateau
    ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=BLUE)

# direct end-labels instead of a legend box
ax.text(4.15, 0.83, "SFT-on-chosen\n(C21 family)", color=BLUE, fontsize=8,
        va="center", ha="left", fontweight="bold")
ax.text(4.15, 0.05, "DPO\n(C19 family)", color=VERM, fontsize=8,
        va="center", ha="left", fontweight="bold")

ax.set_xlabel("Continue-training dose (examples; equal-spaced grid)")
ax.set_ylabel("Dev-set ASR (LLM-LAT)")
ax.set_ylim(-0.03, 1.0)
ax.set_xlim(-0.35, 5.7)
ax.set_xticks(pos)
ax.set_xticklabels([str(d) for d in doses])
ax.grid(True, axis="y", color="#e9e9e9", lw=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_dose.pdf", bbox_inches="tight")
print("wrote fig_dose.pdf")

# ---- Figure 2: defense-in-depth, locked-test ASR across load-bearing conditions ----
conds = ["C1\nbase", "C5\nSFT", "C9\nstripped", "C10\nstripped\n+ stack"]
# (suite label, values C1/C5/C9/C10, color, hatch = grayscale/CVD secondary encoding)
suites = [
    ("advbench", [0.548, 0.010, 0.938, 0.000], BLUE, ""),
    ("harmbench", [0.675, 0.035, 0.940, 0.000], ORANGE, "//"),
    ("dual-use", [0.740, 0.100, 0.940, 0.240], GREEN, ".."),
]

x = np.arange(len(conds))
w = 0.26
fig2, ax2 = plt.subplots(figsize=(6.4, 3.7))
for i, (name, vals, color, hatch) in enumerate(suites):
    xpos = x + (i - 1) * w
    ax2.bar(xpos, vals, w, color=color, hatch=hatch, edgecolor="white", linewidth=0.8,
            label=name, zorder=3)
    for xj, v in zip(xpos, vals, strict=True):
        if v >= 0.05:  # selective labels; near-zero bars read as ~0 without a number
            ax2.text(xj, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)

ax2.set_ylabel("Locked-test ASR")
ax2.set_ylim(0, 1.03)
ax2.set_xticks(x)
ax2.set_xticklabels(conds, fontsize=8.5)
ax2.grid(True, axis="y", color="#e9e9e9", lw=0.8, zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 1.01), handlelength=1.5)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig_bars.pdf", bbox_inches="tight")
print("wrote fig_bars.pdf")
