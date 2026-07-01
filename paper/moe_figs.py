"""Result figures for the MoE-legibility paper, using the shared palette."""
import json
import numpy as np
import matplotlib.pyplot as plt
from style import C, setup
setup()

D = "/Users/steven/moe_legibility"
d = np.load(f"{D}/moe_data.npz", allow_pickle=True)
spec, prop, counts, cats = d["spec"], d["prop"], d["counts"], list(d["cats"])
NL, NE, NC = prop.shape

# ---- Fig A: specialization across depth ---------------------------------- #
layers = np.arange(NL)
fig, ax = plt.subplots(figsize=(6.0, 3.7))
ax.plot(layers, spec.mean(1), "o-", color=C["blue"], lw=1.9, ms=5, label="mean over experts")
ax.plot(layers, spec.max(1), "s-", color=C["green"], lw=1.9, ms=5, label="most specialized expert")
ax.set_xlabel("layer (depth)"); ax.set_ylabel("specialization score")
ax.set_ylim(0, 1.0); ax.legend(fontsize=9)
ax.set_title("Expert specialization concentrates in late layers")
fig.tight_layout(); fig.savefig("figures/fig_a_depth.pdf"); fig.savefig("figures/fig_a_depth.png"); plt.close(fig)

# ---- Fig B: category x expert routing heatmap (most specialized layer) ---- #
L = int(spec.mean(1).argmax())
M = prop[L].T                                  # (NC, NE)
fig, ax = plt.subplots(figsize=(9.2, 3.2))
im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=min(0.6, M.max()))
ax.set_yticks(range(NC)); ax.set_yticklabels(cats, fontsize=9)
ax.set_xlabel(f"expert index (0-{NE-1}),  layer {L}")
ax.set_title(f"Routing propensity by category and expert (layer {L}): bright cells = specialized experts")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("P(expert | category)", fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_b_heatmap.pdf"); fig.savefig("figures/fig_b_heatmap.png"); plt.close(fig)

# ---- Fig C: distribution of specialization ------------------------------- #
fig, ax = plt.subplots(figsize=(5.6, 3.7))
ax.hist(spec.flatten(), bins=30, color=C["blue"], edgecolor="white", lw=0.4)
ax.axvline(spec.mean(), ls="--", color=C["orange"], lw=1.6,
           label=f"mean = {spec.mean():.2f}")
ax.set_xlabel("specialization score (0 = balanced, 1 = one category)")
ax.set_ylabel("number of (layer, expert) units")
ax.legend(fontsize=9)
ax.set_title("Most experts are balanced; a specialized tail is legible")
fig.tight_layout(); fig.savefig("figures/fig_c_hist.pdf"); fig.savefig("figures/fig_c_hist.png"); plt.close(fig)

# ---- Fig D: peak expert propensity per category -------------------------- #
maxprop = prop.max(axis=(0, 1))                # strongest dedicated expert per category
order = np.argsort(maxprop)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.bar(range(NC), maxprop[order], color=C["blue"], edgecolor="black", lw=0.5)
ax.axhline(8 / NE, ls=":", color=C["gray"], lw=1.2)            # uniform routing baseline (top-8 of 64)
ax.text(NC - 1, 8 / NE + 0.01, "uniform routing", fontsize=7.5, ha="right", color=C["gray"])
ax.set_xticks(range(NC)); ax.set_xticklabels([cats[i] for i in order], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("peak P(expert | category)")
ax.set_title("Every category has a strongly dedicated expert")
fig.tight_layout(); fig.savefig("figures/fig_d_peak.pdf"); fig.savefig("figures/fig_d_peak.png"); plt.close(fig)

print(f"wrote 4 figures; most-specialized layer={L}; mean spec={spec.mean():.3f}; "
      f"peak per-category propensity range {maxprop.min():.2f}-{maxprop.max():.2f}")
