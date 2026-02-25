from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import load_config
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def _load_acts(cfg, label: str):
    with open(Path(cfg.collection.output_dir) / f"meta_{label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    mmap = np.memmap(meta["acts_path"], mode="r", dtype=np.float16, shape=(cfg.collection.__dict__[f"tokens_{label.lower()}"], d_model))
    return np.array(mmap[:n], dtype=np.float32), d_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--sample-per-domain", type=int, default=6000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    xA, d_model = _load_acts(cfg, "A")
    xB, _ = _load_acts(cfg, "B")

    n = min(args.sample_per_domain, len(xA), len(xB))
    rng = np.random.default_rng(cfg.seed)
    xa = xA[rng.choice(len(xA), size=n, replace=False)]
    xb = xB[rng.choice(len(xB), size=n, replace=False)]

    device = get_device(cfg.device_preference)
    sae = SparseAutoencoder(d_model=d_model, d_sae=cfg.sae.d_sae, sparsity_mode=cfg.sae.sparsity_mode, topk=cfg.sae.topk).to(device)
    ckpt = Path(cfg.outputs.checkpoints_dir) / "sae_A.pt"
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()

    with torch.no_grad():
        hA = sae.encode(torch.from_numpy(xa).to(device)).cpu().numpy()
        hB = sae.encode(torch.from_numpy(xb).to(device)).cpu().numpy()

    freqA = (hA > 0).mean(axis=0)
    freqB = (hB > 0).mean(axis=0)
    magA = hA.mean(axis=0)
    magB = hB.mean(axis=0)

    # feature vectors: decoder row + selectivity stats
    Wd = sae.decoder.weight.detach().cpu().numpy().T  # [d_sae, d_model]
    feats = np.concatenate(
        [Wd, (freqA - freqB)[:, None], (magA - magB)[:, None], freqA[:, None], freqB[:, None]],
        axis=1,
    )

    pca3 = PCA(n_components=3, random_state=cfg.seed)
    z = pca3.fit_transform(feats)

    kmeans = KMeans(n_clusters=args.clusters, random_state=cfg.seed, n_init=10)
    cluster = kmeans.fit_predict(z)

    out_fig = Path(cfg.outputs.figures_dir)
    out_tab = Path(cfg.outputs.tables_dir)
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "feature": np.arange(feats.shape[0]),
            "pc1": z[:, 0],
            "pc2": z[:, 1],
            "pc3": z[:, 2],
            "cluster": cluster,
            "freqA": freqA,
            "freqB": freqB,
            "magA": magA,
            "magB": magB,
            "selectivity_freq_AminusB": freqA - freqB,
            "selectivity_mag_AminusB": magA - magB,
        }
    )
    df.to_csv(out_tab / "feature_map_points.csv", index=False)

    # 2D feature map
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df.pc1, df.pc2, c=df.cluster, cmap="tab10", s=18, alpha=0.85)
    plt.xlabel("Feature PC1")
    plt.ylabel("Feature PC2")
    plt.title("SAE Feature Map (2D PCA + clusters)")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    plt.savefig(out_fig / "feature_map_2d.png", dpi=180)
    plt.close()

    # 3D feature map
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(df.pc1, df.pc2, df.pc3, c=df.cluster, cmap="tab10", s=16, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("SAE Feature Map (3D PCA + clusters)")
    fig.colorbar(p, ax=ax, label="cluster", shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_fig / "feature_map_3d.png", dpi=180)
    plt.close()

    # cluster summary for concept grouping
    summary = (
        df.groupby("cluster")
        .agg(
            n_features=("feature", "count"),
            mean_freqA=("freqA", "mean"),
            mean_freqB=("freqB", "mean"),
            mean_sel_freq=("selectivity_freq_AminusB", "mean"),
            mean_sel_mag=("selectivity_mag_AminusB", "mean"),
        )
        .reset_index()
        .sort_values("mean_sel_mag", ascending=False)
    )
    summary.to_csv(out_tab / "feature_cluster_summary.csv", index=False)

    print("Wrote:")
    print(out_fig / "feature_map_2d.png")
    print(out_fig / "feature_map_3d.png")
    print(out_tab / "feature_map_points.csv")
    print(out_tab / "feature_cluster_summary.csv")


if __name__ == "__main__":
    main()
