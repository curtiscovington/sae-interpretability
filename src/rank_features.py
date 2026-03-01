from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .model import load_model_and_tokenizer
from .sae import SparseAutoencoder
from .utils import get_device, set_seed

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
CAPWORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
STOPWORDS = {
    "the", "and", "for", "with", "that", "from", "this", "were", "was", "are", "has", "had", "have",
    "into", "their", "about", "after", "before", "during", "between", "over", "under", "they", "them",
    "his", "her", "she", "him", "its", "you", "your", "our", "ours", "out", "not", "but", "than",
    "then", "also", "can", "could", "would", "should", "may", "might", "one", "two", "three", "new",
    "used", "using", "use", "such", "most", "some", "many", "more", "less", "very", "only", "other",
    "who", "what", "when", "where", "which", "while", "because", "just", "like", "through", "within",
    "unk",
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _tokenize_words(contexts: list[str]) -> list[str]:
    words: list[str] = []
    for c in contexts:
        words.extend([w.lower() for w in WORD_RE.findall(c)])
    return words


def _coherence_score(words: list[str], top_n_words: int = 20) -> tuple[float, list[str], list[tuple[str, int]]]:
    if not words:
        return 0.0, [], []

    content_words = [w for w in words if w not in STOPWORDS]
    chosen = content_words if content_words else words

    counts = Counter(chosen)
    most_common = counts.most_common(top_n_words)
    top_total = sum(v for _, v in most_common)
    if top_total == 0:
        return 0.0, [], []

    probs = np.array([v / top_total for _, v in most_common], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    coherence = float(1.0 - entropy / max(max_entropy, 1e-8))
    keywords = [w for w, _ in most_common[:5]]
    return coherence, keywords, most_common


def _genericity_penalty(words: list[str]) -> float:
    total = len(words)
    if total == 0:
        return 0.0
    stop = sum(1 for w in words if w in STOPWORDS)
    return float(stop / total)


def _entity_ratio(contexts: list[str]) -> float:
    all_tokens = 0
    entity_like = 0
    for c in contexts:
        toks = c.split()
        all_tokens += len(toks)
        entity_like += len(CAPWORD_RE.findall(c))
    if all_tokens == 0:
        return 0.0
    return float(entity_like / all_tokens)


def _context_diversity(contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    buckets = set()
    for c in contexts:
        compact = " ".join(c.split())
        buckets.add(compact[:120].lower())
    return float(len(buckets) / len(contexts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank SAE features for intervention triage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default="A", choices=["A", "B"])
    parser.add_argument("--top-features", type=int, default=240)
    parser.add_argument("--top-contexts", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--max-frequency", type=float, default=0.95)
    parser.add_argument("--min-frequency", type=float, default=0.001)

    parser.add_argument("--activity-weight", type=float, default=0.30)
    parser.add_argument("--specificity-weight", type=float, default=0.20)
    parser.add_argument("--coherence-weight", type=float, default=0.20)
    parser.add_argument("--diversity-weight", type=float, default=0.10)
    parser.add_argument("--entity-weight", type=float, default=0.25)
    parser.add_argument("--genericity-weight", type=float, default=0.35)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    acts_dir = Path(cfg.collection.output_dir)
    out_dir = Path(cfg.outputs.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(acts_dir / f"meta_{args.label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    total_alloc = cfg.collection.__dict__[f"tokens_{args.label.lower()}"]
    acts = np.memmap(meta["acts_path"], mode="r", dtype=np.float16, shape=(total_alloc, d_model))
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
    activity = freq * mag
    specificity = 1.0 - freq

    # shortlist from larger pool by intervention relevance (avoid always-on / ultra-rare features)
    mask = (freq <= args.max_frequency) & (freq >= args.min_frequency)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        raise RuntimeError(
            f"No features within frequency range [{args.min_frequency}, {args.max_frequency}]. "
            "Try relaxing --max-frequency/--min-frequency."
        )

    pre_score = activity[valid_idx] * np.sqrt(np.clip(specificity[valid_idx], 1e-8, 1.0))
    shortlist = valid_idx[np.argsort(pre_score)[::-1][: args.top_features]]

    rows: list[dict] = []
    cards: dict[str, dict] = {}

    for f_idx in shortlist:
        col = h[:, f_idx]
        top_idx = np.argsort(col)[::-1][: args.top_contexts]

        contexts: list[str] = []
        vals: list[float] = []
        for i in top_idx:
            lo = max(0, int(i) - args.window)
            hi = min(len(token_ids), int(i) + args.window + 1)
            snippet = hooked.tokenizer.decode(token_ids[lo:hi], skip_special_tokens=True)
            contexts.append(snippet)
            vals.append(float(col[i]))

        words = _tokenize_words(contexts)
        coherence, keywords, top_words = _coherence_score(words)
        genericity = _genericity_penalty(words)
        entity_ratio = _entity_ratio(contexts)
        diversity = _context_diversity(contexts)

        row = {
            "feature": int(f_idx),
            "activity": float(activity[f_idx]),
            "frequency": float(freq[f_idx]),
            "mean_activation": float(mag[f_idx]),
            "specificity": float(specificity[f_idx]),
            "coherence": float(coherence),
            "genericity": float(genericity),
            "entity_ratio": float(entity_ratio),
            "diversity": float(diversity),
            "keywords": ", ".join(keywords) if keywords else "misc",
        }
        rows.append(row)

        cards[str(int(f_idx))] = {
            **row,
            "top_words": top_words,
            "top_values": vals,
            "top_contexts": contexts,
        }

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No features ranked. Check activations/checkpoint availability.")

    for col in ["activity", "specificity", "coherence", "entity_ratio", "diversity", "genericity"]:
        df[f"{col}_n"] = minmax(df[col].to_numpy())

    # reward medium sparsity (not always-on, not completely dead)
    freq = df["frequency"].to_numpy()
    target = 0.08
    sigma = 0.07
    freq_peak = np.exp(-((freq - target) ** 2) / (2 * sigma**2))
    df["freq_peak"] = freq_peak
    df["freq_peak_n"] = minmax(freq_peak)

    df["brain_surgery_score"] = (
        args.activity_weight * df["activity_n"]
        + args.specificity_weight * df["specificity_n"]
        + args.coherence_weight * df["coherence_n"]
        + args.diversity_weight * df["diversity_n"]
        + args.entity_weight * df["entity_ratio_n"]
        + 0.25 * df["freq_peak_n"]
        - args.genericity_weight * df["genericity_n"]
    )

    df = df.sort_values("brain_surgery_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    stem = f"feature_ranking_{args.label}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "weights": {
                    "activity": args.activity_weight,
                    "specificity": args.specificity_weight,
                    "coherence": args.coherence_weight,
                    "diversity": args.diversity_weight,
                    "entity_ratio": args.entity_weight,
                    "genericity_penalty": args.genericity_weight,
                },
                "frequency_filter": {
                    "min_frequency": args.min_frequency,
                    "max_frequency": args.max_frequency,
                },
                "rows": df.to_dict(orient="records"),
                "feature_cards": cards,
            },
            f,
            indent=2,
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Feature ranking ({args.label})\n\n")
        f.write("Composite triage score (brain-surgery style):\n")
        f.write("- reward: activity, specificity, coherence, context diversity, entity ratio\n")
        f.write("- penalty: genericity (stopword-heavy top words)\n\n")

        topk = min(25, len(df))
        f.write(f"## Top {topk} candidates\n\n")
        for _, r in df.head(topk).iterrows():
            f.write(
                f"- #{int(r['rank'])} feature {int(r['feature'])}: score={r['brain_surgery_score']:.3f}, "
                f"act={r['activity']:.4f}, spec={r['specificity']:.4f}, coh={r['coherence']:.3f}, "
                f"div={r['diversity']:.3f}, entity={r['entity_ratio']:.3f}, genericity={r['genericity']:.3f}, "
                f"keywords={r['keywords']}\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
