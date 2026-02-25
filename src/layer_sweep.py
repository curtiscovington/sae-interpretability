from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _layer_cfg(base: dict, layer: int) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["model"]["layer_index"] = int(layer)

    art_root = Path("artifacts/layer_sweep") / f"layer_{layer}"
    out_root = Path("outputs/layer_sweep") / f"layer_{layer}"

    cfg["collection"]["output_dir"] = str(art_root / "activations")
    cfg["outputs"]["root"] = str(out_root)
    cfg["outputs"]["results_json"] = str(out_root / "results.json")
    cfg["outputs"]["figures_dir"] = str(out_root / "figures")
    cfg["outputs"]["tables_dir"] = str(out_root / "tables")
    cfg["outputs"]["features_dir"] = str(out_root / "features")
    cfg["outputs"]["checkpoints_dir"] = str(art_root / "checkpoints")

    out_cfg = Path("artifacts/layer_sweep/configs") / f"layer_{layer}.yaml"
    _write_yaml(out_cfg, cfg)
    return out_cfg


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _collect_summary(layers: list[int]) -> None:
    rows = []
    for layer in layers:
        path = Path("outputs/layer_sweep") / f"layer_{layer}" / "results.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)
        row = {
            "layer": layer,
            "A_A_r2": r["trainA_evalA"]["r2"],
            "A_B_r2": r["trainA_evalB"]["r2"],
            "B_B_r2": r["trainB_evalB"]["r2"],
            "B_A_r2": r["trainB_evalA"]["r2"],
            "A_A_l0": r["trainA_evalA"]["avg_l0"],
            "B_B_l0": r["trainB_evalB"]["avg_l0"],
            "A_to_B_mse_ratio": r["generalization_degradation"]["A_to_B_mse_ratio"],
            "B_to_A_mse_ratio": r["generalization_degradation"]["B_to_A_mse_ratio"],
        }
        row["score"] = (
            0.5 * (row["A_A_r2"] + row["B_B_r2"])
            - 0.15 * (row["A_to_B_mse_ratio"] + row["B_to_A_mse_ratio"])
            - 0.00025 * (row["A_A_l0"] + row["B_B_l0"])
        )
        rows.append(row)

    out_dir = Path("outputs/layer_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(out_dir / "leaderboard.csv", index=False)
    with open(out_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\nLayer leaderboard")
    print(df[["layer", "score", "A_A_r2", "B_B_r2", "A_to_B_mse_ratio", "B_to_A_mse_ratio", "A_A_l0", "B_B_l0"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", default="1,3,5,7,9,11")
    parser.add_argument("--stage", default="all", choices=["collect", "train", "eval", "all"])
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    base = _read_yaml(Path(args.config))

    for layer in layers:
        lc = _layer_cfg(base, layer)
        if args.stage in {"collect", "all"}:
            _run(["python", "-m", "src.collect_acts", "--config", str(lc)])
        if args.stage in {"train", "all"}:
            _run(["python", "-m", "src.train_sae", "--config", str(lc)])
            _run(["python", "-m", "src.interpret", "--config", str(lc), "--label", "A"])
            _run(["python", "-m", "src.interpret", "--config", str(lc), "--label", "B"])
        if args.stage in {"eval", "all"}:
            _run(["python", "-m", "src.eval", "--config", str(lc)])
            _run(["python", "-m", "src.viz", "--config", str(lc)])

    _collect_summary(layers)


if __name__ == "__main__":
    main()
