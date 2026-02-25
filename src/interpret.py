from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .model import load_model_and_tokenizer
from .sae import SparseAutoencoder
from .utils import get_device, set_seed


def heuristic_label(contexts: list[str]) -> str:
    words = []
    for c in contexts:
        parts = [w.strip('.,:;!?()[]{}"\'`').lower() for w in c.split()]
        words.extend([w for w in parts if len(w) >= 4])
    common = [w for w, _ in Counter(words).most_common(3)]
    return ", ".join(common) if common else "misc-pattern"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default="A", choices=["A", "B"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    acts_dir = Path(cfg.collection.output_dir)
    feats_dir = Path(cfg.outputs.features_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)

    with open(acts_dir / f"meta_{args.label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    acts = np.memmap(meta["acts_path"], mode="r", dtype=np.float16, shape=(cfg.collection.__dict__[f'tokens_{args.label.lower()}'], d_model))
    x = torch.from_numpy(np.array(acts[:n], dtype=np.float32))

    token_ids = np.load(meta["tokens_path"])

    device = get_device(cfg.device_preference)
    hooked = load_model_and_tokenizer(cfg.model.model_name, cfg.model.dtype, device)

    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)
    ckpt = Path(cfg.outputs.checkpoints_dir) / f"sae_{args.label}.pt"
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()

    with torch.no_grad():
        h = sae.encode(x.to(device)).cpu().numpy()

    freq = (h > 0).mean(axis=0)
    mag = h.mean(axis=0)
    score = freq * mag
    top_features = np.argsort(score)[::-1][: cfg.interpret.top_features]

    results = {}
    win = cfg.interpret.context_window_tokens

    for f_idx in top_features:
        col = h[:, f_idx]
        top_idx = np.argsort(col)[::-1][: cfg.interpret.top_contexts]
        contexts = []
        vals = []
        for i in top_idx:
            lo = max(0, i - win)
            hi = min(len(token_ids), i + win + 1)
            snippet = hooked.tokenizer.decode(token_ids[lo:hi], skip_special_tokens=True)
            contexts.append(snippet)
            vals.append(float(col[i]))
        results[str(int(f_idx))] = {
            "feature_index": int(f_idx),
            "activation_mean": float(col.mean()),
            "activation_max": float(col.max()),
            "activation_frequency": float((col > 0).mean()),
            "heuristic_label": heuristic_label(contexts),
            "top_contexts": contexts,
            "top_values": vals,
        }

    json_path = feats_dir / f"features_{args.label}.json"
    md_path = feats_dir / f"features_{args.label}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Feature summaries ({args.label})\n\n")
        for v in results.values():
            f.write(f"## Feature {v['feature_index']} — {v['heuristic_label']}\n")
            f.write(f"- mean: {v['activation_mean']:.4f}\n")
            f.write(f"- max: {v['activation_max']:.4f}\n")
            f.write(f"- frequency: {v['activation_frequency']:.4f}\n")
            for c in v["top_contexts"][:5]:
                f.write(f"  - {c[:220].replace(chr(10), ' ')}\n")
            f.write("\n")

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
