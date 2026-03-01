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
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def residual_intervention_hook_factory(
    sae: SparseAutoencoder,
    feature_idx: int,
    alpha: float,
    strength: float = 1.0,
):
    """
    Residualized SAE intervention:
      h0 = encode(x)
      h1 = h0 with one feature scaled
      x' = x + strength * (decode(h1) - decode(h0))

    This removes the global decode(encode(x)) replacement confound.
    """

    def hook(_module, _inp, out):
        x = out
        dt = x.dtype
        with torch.no_grad():
            xf = x.float()
            h0 = sae.encode(xf)
            h1 = h0.clone()
            h1[..., feature_idx] = h1[..., feature_idx] * alpha
            delta = sae.decoder(h1) - sae.decoder(h0)
            x_new = xf + strength * delta
        return x_new.to(dt)

    return hook


def one_token_ids(tokenizer, words: list[str]) -> list[int]:
    out = []
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
    return sorted(set(out))


def target_logprob_mass(model, tokenizer, prompt: str, target_ids: list[int], device: torch.device) -> float:
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[:, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
        tid = torch.tensor(target_ids, dtype=torch.long, device=device)
        return float(torch.logsumexp(lp[:, tid], dim=-1).item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired token-level logprob probe under feature knob intervention")
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="outputs/features/paired_logprob_probe_A")
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

    selected = [int(x["feature"]) for x in json.loads(Path(args.candidate_json).read_text())["selected"]]
    probes = json.loads(Path(args.probe_json).read_text())["themes"]
    alphas = parse_alphas(args.alphas)

    block = model.gpt_neox.layers[cfg.model.layer_index]

    rows = []
    for theme in probes:
        tids = one_token_ids(tok, theme["targets"])
        if not tids:
            continue
        for prompt in theme["prompts"]:
            base = target_logprob_mass(model, tok, prompt, tids, device)
            for feat in selected:
                for a in alphas:
                    h = block.mlp.register_forward_hook(
                        residual_intervention_hook_factory(sae, feat, a, strength=args.strength)
                    )
                    try:
                        score = target_logprob_mass(model, tok, prompt, tids, device)
                    finally:
                        h.remove()
                    rows.append(
                        {
                            "theme": theme["name"],
                            "prompt": prompt,
                            "feature": feat,
                            "alpha": a,
                            "strength": args.strength,
                            "baseline_logprob_mass": base,
                            "logprob_mass": score,
                            "delta_logprob_mass": score - base,
                        }
                    )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["feature", "theme", "alpha", "strength"], as_index=False)
        .agg(delta_mean=("delta_logprob_mass", "mean"), delta_std=("delta_logprob_mass", "std"))
        .sort_values(["feature", "theme", "alpha"], ascending=[True, True, False])
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_suffix(".csv")
    sum_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_path = out_prefix.with_suffix(".md")
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Paired logprob probe (residual intervention)\n\n")
        f.write(f"Intervention strength: {args.strength}\n\n")
        for feat in sorted(summary.feature.unique().tolist()):
            f.write(f"## Feature {feat}\n")
            subf = summary[summary.feature == feat]
            for theme in sorted(subf.theme.unique().tolist()):
                f.write(f"### {theme}\n")
                subt = subf[subf.theme == theme]
                for _, r in subt.iterrows():
                    std = r.delta_std if pd.notna(r.delta_std) else 0.0
                    f.write(f"- alpha={r.alpha:.2f}: Δlogprob={r.delta_mean:.6f} (std={std:.6f})\n")
            f.write("\n")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
