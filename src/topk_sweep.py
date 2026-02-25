from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_cfg(base: dict, layer: int, topk: int) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["model"]["layer_index"] = layer
    cfg["sae"]["sparsity_mode"] = "topk"
    cfg["sae"]["topk"] = int(topk)

    # Reuse activation collection from prior layer sweep run
    cfg["collection"]["output_dir"] = f"artifacts/layer_sweep/layer_{layer}/activations"

    out_root = Path(f"outputs/topk_sweep/k{topk}/layer_{layer}")
    art_root = Path(f"artifacts/topk_sweep/k{topk}/layer_{layer}")
    cfg["outputs"]["root"] = str(out_root)
    cfg["outputs"]["results_json"] = str(out_root / "results.json")
    cfg["outputs"]["figures_dir"] = str(out_root / "figures")
    cfg["outputs"]["tables_dir"] = str(out_root / "tables")
    cfg["outputs"]["features_dir"] = str(out_root / "features")
    cfg["outputs"]["checkpoints_dir"] = str(art_root / "checkpoints")

    p = Path(f"artifacts/topk_sweep/configs/k{topk}_layer_{layer}.yaml")
    write_yaml(p, cfg)
    return p


def summarize(topks: list[int], layers: list[int]) -> None:
    rows = []
    for k in topks:
        for layer in layers:
            rp = Path(f"outputs/topk_sweep/k{k}/layer_{layer}/results.json")
            if not rp.exists():
                continue
            with open(rp, "r", encoding="utf-8") as f:
                r = json.load(f)
            row = {
                "topk": k,
                "layer": layer,
                "A_A_r2": r["trainA_evalA"]["r2"],
                "B_B_r2": r["trainB_evalB"]["r2"],
                "A_to_B_ratio": r["generalization_degradation"]["A_to_B_mse_ratio"],
                "B_to_A_ratio": r["generalization_degradation"]["B_to_A_mse_ratio"],
                "A_A_l0": r["trainA_evalA"]["avg_l0"],
                "B_B_l0": r["trainB_evalB"]["avg_l0"],
            }
            row["score"] = (
                0.5 * (row["A_A_r2"] + row["B_B_r2"])
                - 0.15 * (row["A_to_B_ratio"] + row["B_to_A_ratio"])
                - 0.00025 * (row["A_A_l0"] + row["B_B_l0"])
            )
            rows.append(row)

    out = Path("outputs/topk_sweep")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(out / "leaderboard.csv", index=False)
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", default="0,1,2,3,4,5")
    ap.add_argument("--topks", default="24,16")
    args = ap.parse_args()

    base = read_yaml(Path(args.config))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]

    for k in topks:
        for layer in layers:
            cfgp = make_cfg(base, layer, k)
            run(["python", "-m", "src.train_sae", "--config", str(cfgp)])
            run(["python", "-m", "src.interpret", "--config", str(cfgp), "--label", "A"])
            run(["python", "-m", "src.interpret", "--config", str(cfgp), "--label", "B"])
            run(["python", "-m", "src.eval", "--config", str(cfgp)])
            run(["python", "-m", "src.viz", "--config", str(cfgp)])

    summarize(topks, layers)
