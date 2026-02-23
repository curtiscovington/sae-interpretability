from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    fig_dir = Path(cfg.outputs.figures_dir)
    tab_dir = Path(cfg.outputs.tables_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) training curves
    plt.figure(figsize=(8, 5))
    for label in ["A", "B"]:
        df = pd.read_csv(tab_dir / f"train_log_{label}.csv")
        plt.plot(df["epoch"], df["train_recon"], marker="o", label=f"train_recon_{label}")
        plt.plot(df["epoch"], df["val_recon"], marker="x", linestyle="--", label=f"val_recon_{label}")
    plt.xlabel("epoch")
    plt.ylabel("reconstruction MSE")
    plt.title("SAE reconstruction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=160)
    plt.close()

    # 2) sparsity histogram
    plt.figure(figsize=(8, 5))
    h = pd.read_csv(tab_dir / "l0_hist_trainA_evalA.csv")
    centers = (h["bin_left"] + h["bin_right"]) / 2
    plt.bar(centers, h["count"], width=(h["bin_right"] - h["bin_left"]).iloc[0] * 0.9)
    plt.xlabel("L0 non-zeros per token")
    plt.ylabel("count")
    plt.title("Sparsity histogram (train A, eval A)")
    plt.tight_layout()
    plt.savefig(fig_dir / "sparsity_hist.png", dpi=160)
    plt.close()

    # 3) feature frequency vs magnitude
    fm = pd.read_csv(tab_dir / "feature_freq_mag_trainA_evalA.csv")
    plt.figure(figsize=(8, 5))
    plt.scatter(fm["frequency"], fm["magnitude"], s=8, alpha=0.6)
    plt.xlabel("activation frequency")
    plt.ylabel("mean activation magnitude")
    plt.title("Feature frequency vs magnitude")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_freq_vs_mag.png", dpi=160)
    plt.close()

    # 4) generalization bar chart
    sm = pd.read_csv(tab_dir / "summary_metrics.csv")
    labels = [f"train{r.train}->eval{r.eval}" for r in sm.itertuples()]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, sm["mse"])
    plt.xticks(rotation=25)
    plt.ylabel("MSE")
    plt.title("Generalization (cross-dataset reconstruction)")
    plt.tight_layout()
    plt.savefig(fig_dir / "generalization_bar.png", dpi=160)
    plt.close()

    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
