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


def simple_keyword_hit_rate(texts: list[str], keywords: list[str]) -> float:
    kws = [k.lower() for k in keywords if k]
    if not kws:
        return 0.0
    hits = 0
    total = 0
    for t in texts:
        low = t.lower()
        for k in kws:
            total += 1
            if k in low:
                hits += 1
    return hits / total if total else 0.0


def extract_keywords(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def generate_continuations(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 40, n: int = 3) -> list[str]:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            g = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            txt = tokenizer.decode(g[0], skip_special_tokens=True)
            outs.append(txt)
    return outs


def collect_feature_activation(model, tokenizer, sae: SparseAutoencoder, layer_idx: int, prompt: str, feature_idx: int, device: torch.device) -> tuple[float, float]:
    collected = {}

    def probe_hook(_module, _inp, out):
        x = out.float()
        with torch.no_grad():
            h = sae.encode(x)
        # mean + max over batch/seq for this feature
        feat = h[..., feature_idx]
        collected["mean"] = float(feat.mean().item())
        collected["max"] = float(feat.max().item())
        return out

    block = model.gpt_neox.layers[layer_idx]
    handle = block.mlp.register_forward_hook(probe_hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    finally:
        handle.remove()

    return collected.get("mean", 0.0), collected.get("max", 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature-anchored behavioral probe with knob sweeps.")
    p.add_argument("--config", required=True)
    p.add_argument("--ranking-json", required=True)
    p.add_argument("--candidate-json", required=True)
    p.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    p.add_argument("--contexts-per-feature", type=int, default=4)
    p.add_argument("--gen-per-context", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=36)
    p.add_argument("--out-prefix", default="outputs/features/anchored_probe_A")
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

    ranking = json.loads(Path(args.ranking_json).read_text())
    cards = ranking.get("feature_cards", {})
    candidates = json.loads(Path(args.candidate_json).read_text())
    selected = [int(x["feature"]) for x in candidates["selected"]]

    alphas = parse_alphas(args.alphas)
    block = model.gpt_neox.layers[cfg.model.layer_index]

    rows = []

    for f_idx in selected:
        card = cards.get(str(f_idx), {})
        contexts = card.get("top_contexts", [])[: args.contexts_per_feature]
        keywords = extract_keywords(card.get("keywords", ""))

        if not contexts:
            continue

        # baseline activations for each context (no intervention)
        for ci, ctx in enumerate(contexts):
            base_act_mean, base_act_max = collect_feature_activation(
                model, tokenizer, sae, cfg.model.layer_index, ctx, f_idx, device
            )

            base_gens = generate_continuations(
                model, tokenizer, ctx, device, max_new_tokens=args.max_new_tokens, n=args.gen_per_context
            )
            base_hit = simple_keyword_hit_rate(base_gens, keywords)

            rows.append(
                {
                    "feature": f_idx,
                    "context_idx": ci,
                    "alpha": 1.0,
                    "mode": "baseline",
                    "feature_act_mean": base_act_mean,
                    "feature_act_max": base_act_max,
                    "keyword_hit_rate": base_hit,
                    "keywords": ", ".join(keywords),
                }
            )

            for a in alphas:
                hook = intervention_hook_factory(sae, f_idx, a)
                handle = block.mlp.register_forward_hook(hook)
                try:
                    act_mean, act_max = collect_feature_activation(
                        model, tokenizer, sae, cfg.model.layer_index, ctx, f_idx, device
                    )
                    gens = generate_continuations(
                        model, tokenizer, ctx, device, max_new_tokens=args.max_new_tokens, n=args.gen_per_context
                    )
                    hit = simple_keyword_hit_rate(gens, keywords)
                finally:
                    handle.remove()

                rows.append(
                    {
                        "feature": f_idx,
                        "context_idx": ci,
                        "alpha": float(a),
                        "mode": "intervened",
                        "feature_act_mean": act_mean,
                        "feature_act_max": act_max,
                        "keyword_hit_rate": hit,
                        "keywords": ", ".join(keywords),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected. Check candidates/ranking contexts.")

    # create deltas relative to baseline row per feature+context
    base = (
        df[df["mode"] == "baseline"][
            ["feature", "context_idx", "feature_act_mean", "feature_act_max", "keyword_hit_rate"]
        ]
        .rename(
            columns={
                "feature_act_mean": "base_feature_act_mean",
                "feature_act_max": "base_feature_act_max",
                "keyword_hit_rate": "base_keyword_hit_rate",
            }
        )
    )

    out = df.merge(base, on=["feature", "context_idx"], how="left")
    out["delta_feature_act_mean"] = out["feature_act_mean"] - out["base_feature_act_mean"]
    out["delta_feature_act_max"] = out["feature_act_max"] - out["base_feature_act_max"]
    out["delta_keyword_hit_rate"] = out["keyword_hit_rate"] - out["base_keyword_hit_rate"]

    summary = (
        out[out["mode"] == "intervened"]
        .groupby(["feature", "alpha"], as_index=False)
        .agg(
            feature_act_mean=("feature_act_mean", "mean"),
            base_feature_act_mean=("base_feature_act_mean", "mean"),
            delta_feature_act_mean=("delta_feature_act_mean", "mean"),
            keyword_hit_rate=("keyword_hit_rate", "mean"),
            base_keyword_hit_rate=("base_keyword_hit_rate", "mean"),
            delta_keyword_hit_rate=("delta_keyword_hit_rate", "mean"),
        )
        .sort_values(["feature", "alpha"], ascending=[True, False])
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_suffix(".csv")
    sum_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_path = out_prefix.with_suffix(".md")

    out.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Feature-anchored knob probe\n\n")
        f.write(f"Selected features: {selected}\n")
        f.write(f"Alphas: {alphas}\n\n")
        for feat in sorted(summary["feature"].unique().tolist()):
            f.write(f"## Feature {feat}\n")
            sub = summary[summary["feature"] == feat]
            for _, r in sub.iterrows():
                f.write(
                    f"- alpha={r['alpha']:.2f}: act={r['feature_act_mean']:.6f} (base {r['base_feature_act_mean']:.6f}, Δ {r['delta_feature_act_mean']:.6f}); "
                    f"hit={r['keyword_hit_rate']:.3f} (base {r['base_keyword_hit_rate']:.3f}, Δ {r['delta_keyword_hit_rate']:.3f})\n"
                )
            f.write("\n")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
