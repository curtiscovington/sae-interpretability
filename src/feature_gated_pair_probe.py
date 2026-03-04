from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .model import load_model_and_tokenizer, register_mlp_output_hook
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def parse_alphas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def target_ids_from_words(tokenizer, words: list[str]) -> list[int]:
    """Use first-token ids for each target word/phrase (works for multi-token words too)."""
    out = []
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) >= 1:
            out.append(int(ids[0]))
    return sorted(set(out))


def rollout_logprob_mass(
    model,
    tokenizer,
    prompt: str,
    target_ids: list[int],
    device: torch.device,
    horizon: int = 24,
) -> float:
    """Greedy rollout; accumulate target logprob mass over multiple generated steps."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    total = 0.0
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(horizon):
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]
            lp = torch.log_softmax(logits, dim=-1)
            total += float(torch.logsumexp(lp[:, tid], dim=-1).item())

            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

    return total / max(horizon, 1)


def gated_residual_hook_factory(sae: SparseAutoencoder, feature_idx: int, alpha: float, gate_quantile: float, strength: float):
    def hook(_module, _inp, out):
        x = out
        dt = x.dtype
        with torch.no_grad():
            xf = x.float()
            h0 = sae.encode(xf)
            h1 = h0.clone()

            feat = h0[..., feature_idx]
            thr = torch.quantile(feat.flatten(), gate_quantile)
            mask = (feat >= thr).float()

            # only intervene where feature is active above threshold
            h1[..., feature_idx] = h0[..., feature_idx] * (1.0 - mask + mask * alpha)

            delta = sae.decoder(h1) - sae.decoder(h0)
            x_new = xf + strength * delta
        return x_new.to(dt)

    return hook


def main() -> None:
    ap = argparse.ArgumentParser(description="Activation-gated minimal-pair probe for feature steering")
    ap.add_argument("--config", required=True)
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--alphas", default="0.0,0.25,0.5,1.0,1.5,2.0")
    ap.add_argument("--gate-quantile", type=float, default=0.90)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out-prefix", default="outputs/features/feature640_gated_pairs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device_preference)

    hooked = load_model_and_tokenizer(cfg.model.model_name, cfg.model.dtype, device)
    model, tok = hooked.model, hooked.tokenizer

    sae = SparseAutoencoder(
        d_model=int(model.config.hidden_size),
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    sae.load_state_dict(torch.load(Path(cfg.outputs.checkpoints_dir) / "sae_A.pt", map_location=device))
    sae.eval()

    probe = json.loads(Path(args.probe_json).read_text())
    alphas = parse_alphas(args.alphas)

    # expected schema:
    # {"targets": [...], "pairs": [{"id":"...","a":"...","b":"..."}, ...]}
    target_ids = target_ids_from_words(tok, probe["targets"])
    if not target_ids:
        raise RuntimeError("No target ids could be derived from probe targets.")
    pairs = probe["pairs"]

    rows = []
    for pair in pairs:
        pa = pair["a"]
        pb = pair["b"]

        base_a = rollout_logprob_mass(model, tok, pa, target_ids, device, horizon=args.horizon)
        base_b = rollout_logprob_mass(model, tok, pb, target_ids, device, horizon=args.horizon)
        base_contrast = base_a - base_b

        for alpha in alphas:
            h = register_mlp_output_hook(model, cfg.model.layer_index, 
                gated_residual_hook_factory(
                    sae=sae,
                    feature_idx=args.feature,
                    alpha=alpha,
                    gate_quantile=args.gate_quantile,
                    strength=args.strength,
                )
            )
            try:
                s_a = rollout_logprob_mass(model, tok, pa, target_ids, device, horizon=args.horizon)
                s_b = rollout_logprob_mass(model, tok, pb, target_ids, device, horizon=args.horizon)
            finally:
                h.remove()

            contrast = s_a - s_b
            rows.append(
                {
                    "pair_id": pair.get("id", "pair"),
                    "feature": args.feature,
                    "alpha": alpha,
                    "gate_quantile": args.gate_quantile,
                    "strength": args.strength,
                    "base_a": base_a,
                    "base_b": base_b,
                    "base_contrast": base_contrast,
                    "score_a": s_a,
                    "score_b": s_b,
                    "contrast": contrast,
                    "delta_contrast": contrast - base_contrast,
                }
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["feature", "alpha", "gate_quantile", "strength"], as_index=False)
        .agg(
            delta_contrast_mean=("delta_contrast", "mean"),
            delta_contrast_std=("delta_contrast", "std"),
            n_pairs=("delta_contrast", "count"),
        )
        .sort_values(["alpha"], ascending=[False])
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_suffix(".csv")
    sum_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_path = out_prefix.with_suffix(".md")
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Activation-gated minimal-pair probe\n\n")
        f.write(f"Feature: {args.feature}\n")
        f.write(f"Gate quantile: {args.gate_quantile}\n")
        f.write(f"Strength: {args.strength}\n")
        f.write(f"Pairs: {len(pairs)}\n\n")
        for _, r in summary.iterrows():
            std = r.delta_contrast_std if pd.notna(r.delta_contrast_std) else 0.0
            f.write(
                f"- alpha={r.alpha:.2f}: Δcontrast={r.delta_contrast_mean:.6f} (std={std:.6f}, n={int(r.n_pairs)})\n"
            )

    print(f"Wrote {raw_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
