from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def load_acts(cfg, label: str):
    with open(Path(cfg.collection.output_dir) / f"meta_{label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    mmap = np.memmap(meta["acts_path"], mode="r", dtype=np.float16, shape=(cfg.collection.__dict__[f'tokens_{label.lower()}'], d_model))
    return torch.from_numpy(np.array(mmap[:n], dtype=np.float32)), d_model


def eval_pair(model, x: torch.Tensor, device: torch.device):
    with torch.no_grad():
        xb = x.to(device)
        recon, h = model(xb)
        err = recon - xb
        mse = float((err.pow(2).mean()).item())
        var = float(((xb - xb.mean(dim=0)).pow(2).mean()).item())
        r2 = float(1.0 - mse / max(var, 1e-8))

        l0 = (h > 0).sum(dim=1).float()
        l1 = h.abs().sum(dim=1)
        freq = (h > 0).float().mean(dim=0)
        mag = h.mean(dim=0)

    return {
        "mse": mse,
        "r2": r2,
        "avg_l0": float(l0.mean().item()),
        "avg_l1": float(l1.mean().item()),
        "l0_values": l0.cpu().numpy(),
        "freq": freq.cpu().numpy(),
        "mag": mag.cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device_preference)

    xA, d_model = load_acts(cfg, "A")
    xB, _ = load_acts(cfg, "B")

    rows = []
    all_results = {}

    for train_label in ["A", "B"]:
        model = SparseAutoencoder(
            d_model=d_model,
            d_sae=cfg.sae.d_sae,
            sparsity_mode=cfg.sae.sparsity_mode,
            topk=cfg.sae.topk,
        ).to(device)
        ckpt = Path(cfg.outputs.checkpoints_dir) / f"sae_{train_label}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        for eval_label, x in [("A", xA), ("B", xB)]:
            metrics = eval_pair(model, x, device)
            key = f"train{train_label}_eval{eval_label}"
            all_results[key] = {
                k: v for k, v in metrics.items() if k not in {"l0_values", "freq", "mag"}
            }
            rows.append(
                {
                    "train": train_label,
                    "eval": eval_label,
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "avg_l0": metrics["avg_l0"],
                    "avg_l1": metrics["avg_l1"],
                }
            )

            hist, bin_edges = np.histogram(metrics["l0_values"], bins=20)
            hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "bin_right": bin_edges[1:], "count": hist})
            hist_df.to_csv(Path(cfg.outputs.tables_dir) / f"l0_hist_train{train_label}_eval{eval_label}.csv", index=False)

            freq_mag_df = pd.DataFrame({"feature": np.arange(len(metrics["freq"])), "frequency": metrics["freq"], "magnitude": metrics["mag"]})
            freq_mag_df.to_csv(Path(cfg.outputs.tables_dir) / f"feature_freq_mag_train{train_label}_eval{eval_label}.csv", index=False)

    # selectivity proxy using model A features
    modelA = SparseAutoencoder(
        d_model=d_model,
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    modelA.load_state_dict(torch.load(Path(cfg.outputs.checkpoints_dir) / "sae_A.pt", map_location=device))
    modelA.eval()
    with torch.no_grad():
        hA = modelA.encode(xA.to(device)).cpu().numpy()
        hB = modelA.encode(xB.to(device)).cpu().numpy()
    sel = hA.mean(axis=0) - hB.mean(axis=0)
    pd.DataFrame({"feature": np.arange(len(sel)), "selectivity_A_minus_B": sel}).to_csv(
        Path(cfg.outputs.tables_dir) / "feature_selectivity_AminusB.csv", index=False
    )

    table = pd.DataFrame(rows)
    table.to_csv(Path(cfg.outputs.tables_dir) / "summary_metrics.csv", index=False)

    # degradation
    gen = {
        "A_to_B_mse_ratio": float(all_results["trainA_evalB"]["mse"] / max(all_results["trainA_evalA"]["mse"], 1e-8)),
        "B_to_A_mse_ratio": float(all_results["trainB_evalA"]["mse"] / max(all_results["trainB_evalB"]["mse"], 1e-8)),
    }
    all_results["generalization_degradation"] = gen

    Path(cfg.outputs.root).mkdir(parents=True, exist_ok=True)
    with open(cfg.outputs.results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(table.to_string(index=False))
    print(f"\nWrote {cfg.outputs.results_json}")


if __name__ == "__main__":
    main()
