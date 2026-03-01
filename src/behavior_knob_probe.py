from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .model import load_model_and_tokenizer
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def parse_alphas(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No alpha values provided")
    return vals


def one_token_target_ids(tokenizer, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        toks = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def intervention_hook_factory(sae: SparseAutoencoder, feature_idx: int, alpha: float):
    def hook(_module, _inp, out):
        x = out
        orig_dtype = x.dtype
        with torch.no_grad():
            h = sae.encode(x.float())
            h[..., feature_idx] = h[..., feature_idx] * alpha
            recon = sae.decoder(h)
        return recon.to(orig_dtype)

    return hook


def prompt_theme_score(model, tokenizer, prompt: str, target_ids: list[int], device: torch.device) -> float:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, -1, :]  # next-token distribution

    if not target_ids:
        return float("nan")

    tid = torch.tensor(target_ids, device=device, dtype=torch.long)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logprob_mass = torch.logsumexp(log_probs[:, tid], dim=-1).mean().item()
    return float(target_logprob_mass)


def main() -> None:
    p = argparse.ArgumentParser(description="Behavioral probe sweep for feature gain knobs.")
    p.add_argument("--config", required=True)
    p.add_argument("--candidate-json", required=True)
    p.add_argument("--probe-json", required=True)
    p.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    p.add_argument("--out-prefix", default="outputs/features/behavior_knob_probe_A")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device_preference)

    hooked = load_model_and_tokenizer(cfg.model.model_name, cfg.model.dtype, device)
    model = hooked.model
    tokenizer = hooked.tokenizer

    sae = SparseAutoencoder(
        d_model=int(model.config.hidden_size),
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    ckpt = Path(cfg.outputs.checkpoints_dir) / "sae_A.pt"
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()

    candidates = json.loads(Path(args.candidate_json).read_text())
    selected = [int(x["feature"]) for x in candidates["selected"]]
    controls = json.loads(Path("outputs/features/knob_sweep_A.json").read_text()).get("random_controls", [])

    probes = json.loads(Path(args.probe_json).read_text())["themes"]
    alphas = parse_alphas(args.alphas)

    # baseline without hook
    baseline_rows = []
    for theme in probes:
        target_ids = one_token_target_ids(tokenizer, theme["targets"])
        for prompt in theme["prompts"]:
            score = prompt_theme_score(model, tokenizer, prompt, target_ids, device)
            baseline_rows.append({
                "theme": theme["name"],
                "prompt": prompt,
                "group": "baseline",
                "feature": -1,
                "alpha": 1.0,
                "score": score,
            })

    rows = []

    block = model.gpt_neox.layers[cfg.model.layer_index]

    def run_group(group_name: str, features: list[int]):
        for f_idx in features:
            for alpha in alphas:
                hook = intervention_hook_factory(sae, f_idx, alpha)
                handle = block.mlp.register_forward_hook(hook)
                try:
                    for theme in probes:
                        target_ids = one_token_target_ids(tokenizer, theme["targets"])
                        for prompt in theme["prompts"]:
                            score = prompt_theme_score(model, tokenizer, prompt, target_ids, device)
                            rows.append(
                                {
                                    "theme": theme["name"],
                                    "prompt": prompt,
                                    "group": group_name,
                                    "feature": int(f_idx),
                                    "alpha": float(alpha),
                                    "score": float(score),
                                }
                            )
                finally:
                    handle.remove()

    run_group("selected", selected)
    run_group("random_control", controls)

    base_df = pd.DataFrame(baseline_rows)
    df = pd.DataFrame(rows)

    base_theme = base_df.groupby("theme", as_index=False).agg(baseline_score=("score", "mean"))
    out = df.merge(base_theme, on="theme", how="left")
    out["delta_score"] = out["score"] - out["baseline_score"]

    summary = (
        out.groupby(["group", "feature", "alpha", "theme"], as_index=False)
        .agg(score_mean=("score", "mean"), delta_score_mean=("delta_score", "mean"))
        .sort_values(["group", "feature", "theme", "alpha"], ascending=[True, True, True, False])
    )

    group_summary = (
        out.groupby(["group", "alpha", "theme"], as_index=False)
        .agg(delta_score_mean=("delta_score", "mean"), delta_score_std=("delta_score", "std"))
        .sort_values(["group", "theme", "alpha"], ascending=[True, True, False])
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_suffix(".csv")
    sum_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    group_csv = out_prefix.with_name(out_prefix.name + "_group_summary.csv")
    md_path = out_prefix.with_suffix(".md")
    meta_path = out_prefix.with_suffix(".json")

    out.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    group_summary.to_csv(group_csv, index=False)

    meta = {
        "selected_features": selected,
        "random_controls": controls,
        "alphas": alphas,
        "themes": [t["name"] for t in probes],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Behavioral knob probe (wiki themes)\n\n")
        f.write(f"- selected features: {selected}\n")
        f.write(f"- random controls: {controls}\n")
        f.write(f"- alphas: {alphas}\n\n")

        for theme in [t["name"] for t in probes]:
            f.write(f"## Theme: {theme}\n")
            for grp in ["selected", "random_control"]:
                sub = group_summary[(group_summary["theme"] == theme) & (group_summary["group"] == grp)]
                if sub.empty:
                    continue
                f.write(f"### {grp}\n")
                for _, r in sub.iterrows():
                    std = r["delta_score_std"] if pd.notna(r["delta_score_std"]) else 0.0
                    f.write(f"- alpha={r['alpha']:.2f}: Δscore={r['delta_score_mean']:.4f} (std={std:.4f})\n")
            f.write("\n")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {md_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
