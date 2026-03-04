from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("outputs_gemma2_2b/features")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

cats = ["sports", "code", "safety", "uncertainty"]


def load_category(cat: str):
    s = pd.read_csv(ROOT / f"gemma_{cat}_200pairs_with_controls_summary.csv")
    c = pd.read_csv(ROOT / f"gemma_{cat}_200pairs_controls_aggregate.csv")
    target = s[s["is_target"] == 1].copy().sort_values("alpha")
    return target, c.sort_values("alpha")


# 1) Small-multiples target vs control ribbon
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, cat in zip(axes.flatten(), cats):
    t, c = load_category(cat)
    ax.plot(t["alpha"], t["mean_delta_contrast"], color="#1f77b4", lw=2.5, label="Target feature")
    ax.fill_between(t["alpha"], t["ci95_lo"], t["ci95_hi"], color="#1f77b4", alpha=0.18)

    ax.plot(c["alpha"], c["controls_mean_delta_contrast"], color="#ff7f0e", lw=2.0, ls="--", label="Controls (agg)")
    ax.fill_between(c["alpha"], c["controls_ci95_lo"], c["controls_ci95_hi"], color="#ff7f0e", alpha=0.18)

    ax.axhline(0.0, color="gray", lw=1)
    ax.set_title(cat.title())
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Δcontrast")

handles, labels = axes.flatten()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
fig.suptitle("Gemma SAE Steering: Target vs Matched Random Controls (200 pairs)", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "gemma_target_vs_controls_small_multiples.png", dpi=180)
plt.close(fig)


# 2) Forest-style plot at key alphas
key_alphas = [0.5, 1.5, 2.0]
rows = []
for cat in cats:
    t, c = load_category(cat)
    for a in key_alphas:
        tr = t[t["alpha"] == a].iloc[0]
        cr = c[c["alpha"] == a].iloc[0]
        rows.append({
            "category": cat,
            "alpha": a,
            "target_mean": tr["mean_delta_contrast"],
            "target_lo": tr["ci95_lo"],
            "target_hi": tr["ci95_hi"],
            "control_mean": cr["controls_mean_delta_contrast"],
            "control_lo": cr["controls_ci95_lo"],
            "control_hi": cr["controls_ci95_hi"],
        })
F = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for ax, a in zip(axes, key_alphas):
    d = F[F["alpha"] == a].reset_index(drop=True)
    y = np.arange(len(d))

    ax.errorbar(
        d["target_mean"], y + 0.12,
        xerr=[d["target_mean"] - d["target_lo"], d["target_hi"] - d["target_mean"]],
        fmt="o", color="#1f77b4", label="Target" if a == key_alphas[0] else None
    )
    ax.errorbar(
        d["control_mean"], y - 0.12,
        xerr=[d["control_mean"] - d["control_lo"], d["control_hi"] - d["control_mean"]],
        fmt="s", color="#ff7f0e", label="Controls (agg)" if a == key_alphas[0] else None
    )
    ax.axvline(0.0, color="gray", lw=1)
    ax.set_title(f"alpha={a}")
    ax.set_yticks(y)
    ax.set_yticklabels(d["category"].str.title())
    ax.set_xlabel("Δcontrast")

fig.legend(loc="upper center", ncol=2, frameon=False)
fig.suptitle("Target vs Controls at Key Steering Strengths", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "gemma_forest_key_alphas.png", dpi=180)
plt.close(fig)


# 3) Heatmap-like table plot: feature x alpha for each category
for cat in cats:
    s = pd.read_csv(ROOT / f"gemma_{cat}_200pairs_with_controls_summary.csv")
    piv = s.pivot_table(index="feature", columns="alpha", values="mean_delta_contrast")
    fig, ax = plt.subplots(figsize=(8, max(3, len(piv) * 0.35)))
    im = ax.imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-abs(piv.values).max(), vmax=abs(piv.values).max())
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(x) for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(int(x)) for x in piv.index])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Feature")
    ax.set_title(f"{cat.title()}: mean Δcontrast by feature and alpha")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean Δcontrast")
    fig.tight_layout()
    fig.savefig(OUT / f"gemma_{cat}_feature_alpha_heatmap.png", dpi=180)
    plt.close(fig)

print(f"Wrote figures to: {OUT}")
