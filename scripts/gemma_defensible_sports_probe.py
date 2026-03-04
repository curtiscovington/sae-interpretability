from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
TARGET_FEATURE = 16242  # from quick discovery
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

OUT_DIR = Path("outputs_gemma2_2b/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPORTS_STEMS = [
    "The team finished the season with",
    "In the final game, the player scored",
    "After the trade deadline, the coach said",
    "The league announced a new rule before",
    "Fans expected the team to make the playoffs after",
    "A veteran forward led the team in",
    "The manager praised the defense after",
    "The club rebuilt its roster during",
    "The coach adjusted the lineup after",
    "The player returned from injury and helped",
    "The quarterback read the defense and threw",
    "The striker found space and scored",
    "The pitcher controlled the inning with",
    "The referee reviewed the play before",
    "The captain rallied the team when",
    "The crowd erupted after",
    "The rookie improved each week because",
    "The championship run depended on",
    "The bench unit changed momentum when",
    "The trainer cleared the athlete after",
]

NEUTRAL_STEMS = [
    "The company finished the quarter with",
    "In the final chapter, the writer explained",
    "After the policy deadline, the minister said",
    "The publisher announced a new edition before",
    "Readers expected the novel to become popular after",
    "A veteran scholar led the committee in",
    "The editor praised the argument after",
    "The studio rebuilt its pipeline during",
    "The curator adjusted the exhibit after",
    "The author returned from leave and finished",
    "The analyst reviewed the data and wrote",
    "The designer found space and placed",
    "The researcher controlled the experiment with",
    "The auditor reviewed the report before",
    "The director rallied the group when",
    "The audience reacted after",
    "The student improved each week because",
    "The project launch depended on",
    "The support team changed momentum when",
    "The doctor cleared the patient after",
]

ENDINGS = [
    "the difficult stretch.",
    "a narrow victory.",
    "months of uncertainty.",
    "the opening week.",
    "a surprising setback.",
    "careful preparation.",
    "an unexpected turn.",
    "a critical review.",
    "a long delay.",
    "steady progress.",
]

TARGET_WORDS = [" game", " team", " season", " coach", " match", " playoffs", " score", " league"]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_RANDOM_CONTROLS = 8
BOOTSTRAPS = 3000


def make_pairs() -> list[dict]:
    pairs = []
    idx = 1
    for s, n in zip(SPORTS_STEMS, NEUTRAL_STEMS):
        for e in ENDINGS:
            pairs.append({"id": f"p{idx}", "a": f"{s} {e}", "b": f"{n} {e}"})
            idx += 1
    assert len(pairs) == 200
    return pairs


def first_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if t:
            ids.append(int(t[0]))
    return sorted(set(ids))


def get_resid_post(model, tok, prompt: str, device: torch.device):
    holder = {}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        holder["h"] = h.detach()
        return out

    h = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    finally:
        h.remove()
    return holder["h"]


def next_token_mass(model, tok, prompt: str, target_ids: list[int], device: torch.device) -> float:
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[:, -1, :]
    lp = torch.log_softmax(logits, dim=-1)
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)
    return float(torch.logsumexp(lp[:, tid], dim=-1).item())


def residual_hook_factory(sae: SAE, feature_idx: int, alpha: float):
    def hook(_module, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        dt = x.dtype
        xf = x.float()
        with torch.no_grad():
            h0 = sae.encode(xf)
            h1 = h0.clone()
            h1[..., feature_idx] = h1[..., feature_idx] * alpha
            delta = sae.decode(h1) - sae.decode(h0)
            x_new = xf + delta
        if isinstance(out, tuple):
            out = list(out)
            out[0] = x_new.to(dt)
            return tuple(out)
        return x_new.to(dt)

    return hook


def choose_matched_random_controls(sae: SAE, model, tok, pairs, device: torch.device):
    # Match by baseline mean activation on sports prompts.
    feats = []
    with torch.no_grad():
        for p in pairs[:60]:  # sample subset for speed
            resid = get_resid_post(model, tok, p["a"], device)
            h = sae.encode(resid.float()).mean(dim=1).squeeze(0).cpu().numpy()
            feats.append(h)
    m = np.stack(feats, axis=0).mean(axis=0)
    target_mag = float(m[TARGET_FEATURE])

    idx_all = np.arange(m.shape[0])
    diff = np.abs(m - target_mag)
    order = idx_all[np.argsort(diff)]

    chosen = []
    for i in order:
        if int(i) == TARGET_FEATURE:
            continue
        if len(chosen) >= N_RANDOM_CONTROLS:
            break
        chosen.append(int(i))
    return chosen, target_mag


def bootstrap_ci(vals: np.ndarray, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = rng.choice(vals, size=len(vals), replace=True)
        boots[i] = s.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    pairs = make_pairs()
    target_ids = first_token_ids(tok, TARGET_WORDS)

    random_controls, target_mag = choose_matched_random_controls(sae, model, tok, pairs, device)
    features = [TARGET_FEATURE] + random_controls

    rows = []
    for pair in pairs:
        pa, pb = pair["a"], pair["b"]
        base_a = next_token_mass(model, tok, pa, target_ids, device)
        base_b = next_token_mass(model, tok, pb, target_ids, device)
        base_contrast = base_a - base_b

        for feat in features:
            for alpha in ALPHAS:
                h = model.model.layers[LAYER].register_forward_hook(residual_hook_factory(sae, feat, alpha))
                try:
                    s_a = next_token_mass(model, tok, pa, target_ids, device)
                    s_b = next_token_mass(model, tok, pb, target_ids, device)
                finally:
                    h.remove()

                contrast = s_a - s_b
                rows.append(
                    {
                        "pair_id": pair["id"],
                        "feature": int(feat),
                        "is_target": int(feat == TARGET_FEATURE),
                        "alpha": float(alpha),
                        "base_contrast": float(base_contrast),
                        "contrast": float(contrast),
                        "delta_contrast": float(contrast - base_contrast),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gemma_sports_200pairs_with_controls.csv", index=False)

    # per-feature summary with bootstrap CI
    sum_rows = []
    for (feat, alpha), g in df.groupby(["feature", "alpha"]):
        vals = g["delta_contrast"].to_numpy()
        lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=42 + int(alpha * 100) + int(feat) % 997)
        sum_rows.append(
            {
                "feature": int(feat),
                "is_target": int(int(feat) == TARGET_FEATURE),
                "alpha": float(alpha),
                "mean_delta_contrast": float(vals.mean()),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_pairs": int(len(vals)),
            }
        )

    sdf = pd.DataFrame(sum_rows).sort_values(["is_target", "feature", "alpha"], ascending=[False, True, True])
    sdf.to_csv(OUT_DIR / "gemma_sports_200pairs_with_controls_summary.csv", index=False)

    # aggregate controls
    cdf = df[df["is_target"] == 0].groupby(["alpha", "pair_id"], as_index=False).agg(delta_contrast=("delta_contrast", "mean"))
    agg = []
    for alpha, g in cdf.groupby("alpha"):
        vals = g["delta_contrast"].to_numpy()
        lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=777 + int(alpha * 100))
        agg.append({
            "alpha": float(alpha),
            "controls_mean_delta_contrast": float(vals.mean()),
            "controls_ci95_lo": lo,
            "controls_ci95_hi": hi,
            "n_pairs": int(len(vals)),
        })
    adf = pd.DataFrame(agg).sort_values("alpha")
    adf.to_csv(OUT_DIR / "gemma_sports_200pairs_controls_aggregate.csv", index=False)

    report = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "pairs": len(pairs),
        "target_feature": TARGET_FEATURE,
        "target_feature_mean_activation": target_mag,
        "matched_random_controls": random_controls,
        "alphas": ALPHAS,
        "artifacts": {
            "raw": str(OUT_DIR / "gemma_sports_200pairs_with_controls.csv"),
            "summary": str(OUT_DIR / "gemma_sports_200pairs_with_controls_summary.csv"),
            "controls_aggregate": str(OUT_DIR / "gemma_sports_200pairs_controls_aggregate.csv"),
        },
    }
    (OUT_DIR / "gemma_sports_200pairs_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Gemma Sports Probe (200 Minimal Pairs + Matched Random Controls)",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- SAE: `{SAE_RELEASE}` / `{SAE_ID}`",
        f"- Layer: `{LAYER}`",
        f"- Pairs: `{len(pairs)}`",
        f"- Target feature: `{TARGET_FEATURE}`",
        f"- Matched random controls: `{random_controls}`",
        "",
        "## Target feature summary",
    ]
    tsub = sdf[sdf["is_target"] == 1]
    for _, r in tsub.iterrows():
        md.append(f"- alpha={r['alpha']:.2f}: Δcontrast={r['mean_delta_contrast']:.6f} (95% CI [{r['ci95_lo']:.6f}, {r['ci95_hi']:.6f}], n={int(r['n_pairs'])})")

    md.append("\n## Aggregate random-control summary")
    for _, r in adf.iterrows():
        md.append(f"- alpha={r['alpha']:.2f}: controls Δcontrast={r['controls_mean_delta_contrast']:.6f} (95% CI [{r['controls_ci95_lo']:.6f}, {r['controls_ci95_hi']:.6f}], n={int(r['n_pairs'])})")

    (OUT_DIR / "gemma_sports_200pairs_report.md").write_text("\n".join(md))
    print("Wrote", OUT_DIR / "gemma_sports_200pairs_report.md")


if __name__ == "__main__":
    main()
