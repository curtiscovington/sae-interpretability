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
    mmap = np.memmap(
        meta["acts_path"],
        mode="r",
        dtype=np.float16,
        shape=(cfg.collection.__dict__[f"tokens_{label.lower()}"], d_model),
    )
    x = np.array(mmap[:n], dtype=np.float32)
    return torch.from_numpy(x), d_model


def parse_alphas(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No alpha values provided")
    return vals


def pick_random_controls(
    rng: np.random.Generator,
    d_sae: int,
    selected: list[int],
    n: int,
    freq_map: dict[int, float] | None = None,
) -> list[int]:
    pool = [i for i in range(d_sae) if i not in set(selected)]
    if not pool:
        return []

    if not freq_map:
        pool_arr = np.array(pool, dtype=np.int64)
        n = min(n, len(pool_arr))
        return rng.choice(pool_arr, size=n, replace=False).tolist()

    picked: list[int] = []
    used = set()
    for s in selected[:n]:
        sf = float(freq_map.get(s, 0.0))
        best = None
        best_dist = float("inf")
        for c in pool:
            if c in used:
                continue
            cf = float(freq_map.get(c, 0.0))
            d = abs(cf - sf)
            if d < best_dist:
                best_dist = d
                best = c
        if best is not None:
            used.add(best)
            picked.append(best)

    if len(picked) < n:
        remaining = [x for x in pool if x not in used]
        if remaining:
            extra = rng.choice(np.array(remaining), size=min(n - len(picked), len(remaining)), replace=False).tolist()
            picked.extend(extra)

    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep per-feature SAE gain knobs (alpha scaling).")
    p.add_argument("--config", required=True)
    p.add_argument("--label", default="A", choices=["A", "B"])
    p.add_argument("--candidate-json", required=True)
    p.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--ranking-csv", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--out-prefix", default="outputs/features/knob_sweep_A")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    x_all, d_model = load_acts(cfg, args.label)
    n = min(len(x_all), args.max_tokens)
    x = x_all[:n]

    device = get_device(cfg.device_preference)
    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    ckpt = Path(cfg.outputs.checkpoints_dir) / f"sae_{args.label}.pt"
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()

    candidates_raw = json.loads(Path(args.candidate_json).read_text())
    selected = [int(x["feature"]) for x in candidates_raw["selected"]]

    freq_map = None
    if args.ranking_csv:
        rdf = pd.read_csv(args.ranking_csv)
        if "feature" in rdf.columns and "frequency" in rdf.columns:
            freq_map = {int(r.feature): float(r.frequency) for r in rdf.itertuples(index=False)}

    rng = np.random.default_rng(args.seed)
    controls = pick_random_controls(rng, cfg.sae.d_sae, selected, args.random_controls, freq_map=freq_map)
    alphas = parse_alphas(args.alphas)

    xb = x.to(device)

    with torch.no_grad():
        h_base = sae.encode(xb)
        recon_base = sae.decoder(h_base)

    base_err = recon_base - xb
    base_mse = float((base_err.pow(2).mean()).item())
    var = float(((xb - xb.mean(dim=0)).pow(2).mean()).item())
    base_r2 = float(1.0 - base_mse / max(var, 1e-8))

    rows = []

    def run_one(feature_idx: int, group: str):
        base_feature_mean = float(h_base[:, feature_idx].mean().item())
        base_feature_freq = float((h_base[:, feature_idx] > 0).float().mean().item())

        for a in alphas:
            with torch.no_grad():
                h = h_base.clone()
                h[:, feature_idx] = h[:, feature_idx] * a
                recon = sae.decoder(h)

            err = recon - xb
            mse = float((err.pow(2).mean()).item())
            r2 = float(1.0 - mse / max(var, 1e-8))

            # latent perturbation magnitude for this intervention
            delta_latent = float((h - h_base).abs().mean().item())
            target_delta = float((h[:, feature_idx] - h_base[:, feature_idx]).abs().mean().item())
            target_mean = float(h[:, feature_idx].mean().item())

            rows.append(
                {
                    "group": group,
                    "feature": feature_idx,
                    "alpha": a,
                    "mse": mse,
                    "r2": r2,
                    "delta_mse": mse - base_mse,
                    "delta_r2": r2 - base_r2,
                    "latent_delta_mean": delta_latent,
                    "target_feature_delta_mean": target_delta,
                    "target_feature_mean": target_mean,
                    "target_feature_mean_baseline": base_feature_mean,
                    "target_feature_freq_baseline": base_feature_freq,
                    "target_feature_mean_ratio": (target_mean / base_feature_mean) if base_feature_mean > 1e-12 else 0.0,
                }
            )

    for f in selected:
        run_one(f, "selected")
    for f in controls:
        run_one(f, "random_control")

    df = pd.DataFrame(rows)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_path = out_prefix.with_suffix(".md")
    meta_path = out_prefix.with_suffix(".json")

    df.to_csv(csv_path, index=False)

    # average dose-response by group
    summary = (
        df.groupby(["group", "alpha"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            delta_mse_mean=("delta_mse", "mean"),
            delta_mse_std=("delta_mse", "std"),
            delta_r2_mean=("delta_r2", "mean"),
        )
        .sort_values(["group", "alpha"], ascending=[True, False])
    )
    summary.to_csv(summary_path, index=False)

    meta = {
        "label": args.label,
        "base_mse": base_mse,
        "base_r2": base_r2,
        "tokens_evaluated": int(n),
        "selected_features": selected,
        "random_controls": controls,
        "alphas": alphas,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    top_impacts = df[df["alpha"] == 0.0].sort_values("delta_mse", ascending=False).head(10)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Feature knob sweep ({args.label})\n\n")
        f.write(f"- tokens evaluated: {n}\n")
        f.write(f"- baseline mse: {base_mse:.6f}\n")
        f.write(f"- baseline r2: {base_r2:.6f}\n")
        f.write(f"- selected features: {selected}\n")
        f.write(f"- random controls: {controls}\n")
        f.write(f"- alphas: {alphas}\n\n")

        f.write("## Group-level dose response (delta_mse)\n\n")
        for g in ["selected", "random_control"]:
            sub = summary[summary["group"] == g]
            if sub.empty:
                continue
            f.write(f"### {g}\n")
            for _, r in sub.iterrows():
                f.write(
                    f"- alpha={r['alpha']:.2f}: ΔMSE={r['delta_mse_mean']:.6f} (std={r['delta_mse_std'] if pd.notna(r['delta_mse_std']) else 0.0:.6f}), "
                    f"ΔR2={r['delta_r2_mean']:.6f}\n"
                )
            f.write("\n")

        f.write("## Largest single-feature impacts at alpha=0.0\n\n")
        for _, r in top_impacts.iterrows():
            f.write(
                f"- {r['group']} feature {int(r['feature'])}: ΔMSE={r['delta_mse']:.6f}, ΔR2={r['delta_r2']:.6f}\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
