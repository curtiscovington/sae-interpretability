from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .config import load_config
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def _load_dataset(cfg, label: str):
    with open(Path(cfg.collection.output_dir) / f"meta_{label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    mmap = np.memmap(
        meta["acts_path"], mode="r", dtype=np.float16, shape=(cfg.collection.__dict__[f"tokens_{label.lower()}"], d_model)
    )
    x = np.array(mmap[:n], dtype=np.float32)
    return x, d_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-points", type=int, default=6000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    xA, d_model = _load_dataset(cfg, "A")
    xB, _ = _load_dataset(cfg, "B")

    # balanced sampling
    n_each = min(len(xA), len(xB), args.max_points // 2)
    rng = np.random.default_rng(cfg.seed)
    idxA = rng.choice(len(xA), size=n_each, replace=False)
    idxB = rng.choice(len(xB), size=n_each, replace=False)

    x = np.concatenate([xA[idxA], xB[idxB]], axis=0)
    domain = np.array([0] * n_each + [1] * n_each)

    device = get_device(cfg.device_preference)
    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    ckpt = Path(cfg.outputs.checkpoints_dir) / "sae_A.pt"
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()

    with torch.no_grad():
        h = sae.encode(torch.from_numpy(x).to(device)).cpu().numpy()

    # use PCA for robust + fast 2D/3D projections
    pca3 = PCA(n_components=3, random_state=cfg.seed)
    z3 = pca3.fit_transform(h)
    z2 = z3[:, :2]

    out_fig = Path(cfg.outputs.figures_dir)
    out_tab = Path(cfg.outputs.tables_dir)
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    # 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(z2[domain == 0, 0], z2[domain == 0, 1], s=6, alpha=0.35, label="A (wiki)")
    plt.scatter(z2[domain == 1, 0], z2[domain == 1, 1], s=6, alpha=0.35, label="B (code)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("SAE latent projection (2D PCA)")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_fig / "latent_2d_pca.png", dpi=180)
    plt.close()

    # 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(z3[domain == 0, 0], z3[domain == 0, 1], z3[domain == 0, 2], s=4, alpha=0.28, label="A (wiki)")
    ax.scatter(z3[domain == 1, 0], z3[domain == 1, 1], z3[domain == 1, 2], s=4, alpha=0.28, label="B (code)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("SAE latent projection (3D PCA)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_fig / "latent_3d_pca.png", dpi=180)
    plt.close()

    df = pd.DataFrame(
        {
            "pc1": z3[:, 0],
            "pc2": z3[:, 1],
            "pc3": z3[:, 2],
            "domain": np.where(domain == 0, "A", "B"),
        }
    )
    df.to_csv(out_tab / "latent_projection_points.csv", index=False)

    print("Wrote:")
    print(out_fig / "latent_2d_pca.png")
    print(out_fig / "latent_3d_pca.png")
    print(out_tab / "latent_projection_points.csv")


if __name__ == "__main__":
    main()
