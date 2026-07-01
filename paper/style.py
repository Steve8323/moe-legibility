"""Shared, consistent, professional color palette + matplotlib style for ALL
paper figures. Colorblind-safe (Okabe-Ito based). Import and call setup() at the
top of every figure script so colors are identical across graphs.

Semantic convention (kept consistent across figures):
  gray   = baseline / control (direct, none, original, canonical)
  blue   = reasoner / normal CoT / primary series
  green  = our method (paraphrase bottleneck, atomize, batched, full)
  red    = paraphraser cost / leakage / warning / negative
  orange = secondary condition (interleaved, deterministic, paragraph-SBP, chunked)
  purple = tertiary (back-translation, extra series)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "gray":   "#6C757D",
    "light":  "#E9EEF3",
}
CYCLE = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["gray"]]
# sequential colormap for heatmaps, tuned to the palette (white -> deep blue)
SEQ = "GnBu"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.prop_cycle": cycler(color=CYCLE),
        "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
        "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
        "legend.frameon": False,
    })
    return C
