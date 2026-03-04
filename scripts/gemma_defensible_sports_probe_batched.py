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
TARGET_FEATURE = 16242
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

OUT_DIR = Path("outputs_gemma2_2b/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_RANDOM_CONTROLS = 8
BOOTSTRAPS = 3000
BATCH = 20

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


def make_pairs():
    pairs = []
    idx = 1
    for s, n in zip(SPORTS_STEMS, NEUTRAL_STEMS):
        for e in ENDINGS:
            pairs.append({"id": f"p{idx}", "a": f"{s} {e}", "b": f"{n} {e}"})
            idx += 1
    return pairs


def first_token_ids(tokenizer, words):
    ids = []
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if t:
            ids.append(int(t[0]))
    return sorted(set(ids))


def batch_next_token_mass(model, tok, prompts, target_ids, device):
    out = []
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True)
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            lengths = enc["attention_mask"].sum(dim=1) - 1
            row = torch.arange(logits.shape[0], device=device)
            last_logits = logits[row, lengths.to(device), :]
            lp = torch.log_softmax(last_logits, dim=-1)
            mass = torch.logsumexp(lp[:, tid], dim=-1)
            out.extend([float(x) for x in mass.cpu().tolist()])
    return np.array(out, dtype=np.float64)


def get_activation_mean_feature(model, tok, sae, prompts, device):
    holder = []

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        holder.append(h.detach())
        return out

    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            hnd = model.model.layers[LAYER].register_forward_hook(hook)
            try:
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True)
                _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            finally:
                hnd.remove()

    means = []
    for act in holder:
        z = sae.encode(act.float())  # [b,t,f]
        m = z.mean(dim=(0, 1)).detach().cpu().numpy()
        means.append(m)
    return np.stack(means).mean(axis=0)


def residual_hook_factory(sae, feature_idx, alpha):
    def hook(_m, _i, out):
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


def bootstrap_ci(vals, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for i in range(n):
        s = rng.choice(vals, size=len(vals), replace=True)
        boots[i] = s.mean()
    return tuple(np.percentile(boots, [2.5, 97.5]).tolist())


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    pairs = make_pairs()
    pa = [p["a"] for p in pairs]
    pb = [p["b"] for p in pairs]
    ids = [p["id"] for p in pairs]

    target_ids = first_token_ids(tok, TARGET_WORDS)

    # matched random controls by mean activation proximity on sports prompts
    feat_means = get_activation_mean_feature(model, tok, sae, pa[:80], device)
    target_mag = float(feat_means[TARGET_FEATURE])
    order = np.argsort(np.abs(feat_means - target_mag))
    controls = []
    for i in order:
        i = int(i)
        if i == TARGET_FEATURE:
            continue
        controls.append(i)
        if len(controls) >= N_RANDOM_CONTROLS:
            break

    features = [TARGET_FEATURE] + controls

    base_a = batch_next_token_mass(model, tok, pa, target_ids, device)
    base_b = batch_next_token_mass(model, tok, pb, target_ids, device)
    base_contrast = base_a - base_b

    rows = []
    for feat in features:
        for alpha in ALPHAS:
            h = model.model.layers[LAYER].register_forward_hook(residual_hook_factory(sae, feat, alpha))
            try:
                s_a = batch_next_token_mass(model, tok, pa, target_ids, device)
                s_b = batch_next_token_mass(model, tok, pb, target_ids, device)
            finally:
                h.remove()
            contrast = s_a - s_b
            delta = contrast - base_contrast
            for i in range(len(ids)):
                rows.append(
                    {
                        "pair_id": ids[i],
                        "feature": feat,
                        "is_target": int(feat == TARGET_FEATURE),
                        "alpha": alpha,
                        "base_contrast": float(base_contrast[i]),
                        "contrast": float(contrast[i]),
                        "delta_contrast": float(delta[i]),
                    }
                )

    df = pd.DataFrame(rows)
    raw = OUT_DIR / "gemma_sports_200pairs_with_controls.csv"
    df.to_csv(raw, index=False)

    sums = []
    for (feat, alpha), g in df.groupby(["feature", "alpha"]):
        vals = g["delta_contrast"].to_numpy()
        lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=1000 + int(feat) % 991 + int(alpha * 100))
        sums.append(
            {
                "feature": int(feat),
                "is_target": int(feat == TARGET_FEATURE),
                "alpha": float(alpha),
                "mean_delta_contrast": float(vals.mean()),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "n_pairs": int(len(vals)),
            }
        )
    sdf = pd.DataFrame(sums).sort_values(["is_target", "feature", "alpha"], ascending=[False, True, True])
    summary = OUT_DIR / "gemma_sports_200pairs_with_controls_summary.csv"
    sdf.to_csv(summary, index=False)

    cdf = df[df.is_target == 0].groupby(["alpha", "pair_id"], as_index=False).agg(delta_contrast=("delta_contrast", "mean"))
    agg = []
    for alpha, g in cdf.groupby("alpha"):
        vals = g["delta_contrast"].to_numpy()
        lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=700 + int(alpha * 100))
        agg.append(
            {
                "alpha": float(alpha),
                "controls_mean_delta_contrast": float(vals.mean()),
                "controls_ci95_lo": float(lo),
                "controls_ci95_hi": float(hi),
                "n_pairs": int(len(vals)),
            }
        )
    adf = pd.DataFrame(agg).sort_values("alpha")
    controls_agg = OUT_DIR / "gemma_sports_200pairs_controls_aggregate.csv"
    adf.to_csv(controls_agg, index=False)

    report = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "pairs": len(pairs),
        "target_feature": TARGET_FEATURE,
        "target_feature_mean_activation": target_mag,
        "matched_random_controls": controls,
        "alphas": ALPHAS,
    }
    (OUT_DIR / "gemma_sports_200pairs_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Gemma Sports Probe (200 Minimal Pairs + Matched Random Controls)",
        "",
        f"- Pairs: {len(pairs)}",
        f"- Target feature: {TARGET_FEATURE}",
        f"- Matched random controls: {controls}",
        "",
        "## Target Feature",
    ]
    t = sdf[sdf.is_target == 1]
    for _, r in t.iterrows():
        md.append(
            f"- alpha={r.alpha:.2f}: Δcontrast={r.mean_delta_contrast:.6f} (95% CI [{r.ci95_lo:.6f}, {r.ci95_hi:.6f}], n={int(r.n_pairs)})"
        )
    md.append("\n## Aggregate Controls")
    for _, r in adf.iterrows():
        md.append(
            f"- alpha={r.alpha:.2f}: Δcontrast={r.controls_mean_delta_contrast:.6f} (95% CI [{r.controls_ci95_lo:.6f}, {r.controls_ci95_hi:.6f}], n={int(r.n_pairs)})"
        )

    (OUT_DIR / "gemma_sports_200pairs_report.md").write_text("\n".join(md))
    print("Wrote", raw)
    print("Wrote", summary)
    print("Wrote", controls_agg)


if __name__ == "__main__":
    main()
