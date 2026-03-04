from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
WRONG_LAYER = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

OUT = Path("outputs_gemma2_2b/features")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CATEGORY = "code"
TARGET_FEATURE = 6631
CONTROL_FEATURES = [743, 1692, 3518, 5052]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEEDS = [7, 17, 27, 37, 47, 57, 67]
BATCH = 16

TARGET_WORDS = [" code", " function", " python", " bug", " test", " api", " class", " query"]
PAIRS = [
    ("Write a Python function that validates user input", "Write a clear procedure that validates user input"),
    ("Refactor this class to reduce duplicated code", "Refactor this section to reduce duplicated wording"),
    ("Add unit tests for this function", "Add checklist items for this process"),
    ("Debug the failing test in this module", "Review the failing argument in this section"),
    ("Implement an API endpoint for this feature", "Implement a policy update for this feature"),
    ("Parse this JSON payload safely", "Parse this report summary carefully"),
    ("Handle exceptions in this service", "Handle objections in this discussion"),
    ("Write a SQL query to find duplicates", "Write a table summary to find duplicates"),
]
CONTEXT_HOLDOUT = [
    "while triaging a live incident.",
    "for an onboarding walkthrough.",
    "during a postmortem discussion.",
]


def first_token_ids(tok, words):
    return sorted({int(tok.encode(w, add_special_tokens=False)[0]) for w in words if tok.encode(w, add_special_tokens=False)})


def build_prompts():
    ids, a, b = [], [], []
    n = 0
    for pidx, (pa, pb) in enumerate(PAIRS):
        for c in CONTEXT_HOLDOUT:
            n += 1
            ids.append(f"p{pidx+1}_c{n}")
            a.append(f"{pa} {c}")
            b.append(f"{pb} {c}")
    return ids, a, b


def batch_last_logits(model, tok, prompts, device):
    all_last = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i+BATCH], return_tensors="pt", padding=True)
            out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            idx = enc["attention_mask"].sum(dim=1) - 1
            row = torch.arange(out.shape[0], device=device)
            all_last.append(out[row, idx.to(device), :].float().cpu())
    return torch.cat(all_last, dim=0)


def target_logmass_from_last(last_logits, target_ids):
    lp = torch.log_softmax(last_logits, dim=-1)
    tid = torch.tensor(target_ids, dtype=torch.long)
    return torch.logsumexp(lp[:, tid], dim=-1).numpy()


def evaluate_contrast(model, tok, prompts_a, prompts_b, target_ids, hook_reg, device):
    h = hook_reg() if hook_reg else None
    try:
        la = batch_last_logits(model, tok, prompts_a, device)
        lb = batch_last_logits(model, tok, prompts_b, device)
    finally:
        if h:
            h.remove()
    return target_logmass_from_last(la, target_ids) - target_logmass_from_last(lb, target_ids)


def steer_hook(sae, feat, alpha):
    def hk(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        dt = x.dtype
        xf = x.float()
        with torch.no_grad():
            h0 = sae.encode(xf)
            h1 = h0.clone()
            h1[..., feat] = h1[..., feat] * alpha
            xnew = xf + (sae.decode(h1) - sae.decode(h0))
        if isinstance(out, tuple):
            o = list(out)
            o[0] = xnew.to(dt)
            return tuple(o)
        return xnew.to(dt)

    return hk


def bootstrap_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    return np.percentile(boots, [2.5, 97.5]).tolist()


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    _, a_ho, b_ho = build_prompts()
    tids = first_token_ids(tok, TARGET_WORDS)

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        base = evaluate_contrast(model, tok, a_ho, b_ho, tids, None, device)

        for alpha in ALPHAS:
            # target
            d = evaluate_contrast(
                model, tok, a_ho, b_ho, tids,
                lambda: model.model.layers[LAYER].register_forward_hook(steer_hook(sae, TARGET_FEATURE, alpha)),
                device,
            ) - base
            rows.append({
                "seed": seed, "feature": TARGET_FEATURE, "group": "target", "condition": "target", "alpha": alpha,
                "mean_delta": float(d.mean())
            })

            # wrong-layer target
            d_wl = evaluate_contrast(
                model, tok, a_ho, b_ho, tids,
                lambda: model.model.layers[WRONG_LAYER].register_forward_hook(steer_hook(sae, TARGET_FEATURE, alpha)),
                device,
            ) - base
            rows.append({
                "seed": seed, "feature": TARGET_FEATURE, "group": "target", "condition": "wrong_layer", "alpha": alpha,
                "mean_delta": float(d_wl.mean())
            })

            # wrong-hook target (mlp)
            d_wh = evaluate_contrast(
                model, tok, a_ho, b_ho, tids,
                lambda: model.model.layers[LAYER].mlp.register_forward_hook(steer_hook(sae, TARGET_FEATURE, alpha)),
                device,
            ) - base
            rows.append({
                "seed": seed, "feature": TARGET_FEATURE, "group": "target", "condition": "wrong_hook", "alpha": alpha,
                "mean_delta": float(d_wh.mean())
            })

            # controls (target condition only)
            for cf in CONTROL_FEATURES:
                dc = evaluate_contrast(
                    model, tok, a_ho, b_ho, tids,
                    lambda f=cf: model.model.layers[LAYER].register_forward_hook(steer_hook(sae, f, alpha)),
                    device,
                ) - base
                rows.append({
                    "seed": seed, "feature": cf, "group": "control", "condition": "target", "alpha": alpha,
                    "mean_delta": float(dc.mean())
                })

    df = pd.DataFrame(rows)
    raw_csv = OUT / "gemma_code_seeded_effects_rows.csv"
    df.to_csv(raw_csv, index=False)

    # aggregate with CI across seeds
    summary_rows = []
    for (group, condition, alpha), g in df.groupby(["group", "condition", "alpha"]):
        vals = g["mean_delta"].to_numpy()
        lo, hi = bootstrap_ci(vals, seed=42 + int(alpha * 100) + len(vals))
        summary_rows.append({
            "group": group,
            "condition": condition,
            "alpha": float(alpha),
            "mean_of_seed_means": float(vals.mean()),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "seed_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": int(len(vals)),
        })

    sdf = pd.DataFrame(summary_rows).sort_values(["group", "condition", "alpha"])
    summary_csv = OUT / "gemma_code_seeded_effects_summary.csv"
    sdf.to_csv(summary_csv, index=False)

    # pass/fail recompute at alpha=1.5
    a = 1.5
    tgt = sdf[(sdf.group == "target") & (sdf.condition == "target") & (sdf.alpha == a)].iloc[0]
    ctr_vals = df[(df.group == "control") & (df.condition == "target") & (df.alpha == a)].groupby("seed", as_index=False)["mean_delta"].mean()
    ctr_mean = float(ctr_vals["mean_delta"].mean())
    ctr_std = float(ctr_vals["mean_delta"].std(ddof=1))
    wl = sdf[(sdf.group == "target") & (sdf.condition == "wrong_layer") & (sdf.alpha == a)].iloc[0]
    wh = sdf[(sdf.group == "target") & (sdf.condition == "wrong_hook") & (sdf.alpha == a)].iloc[0]

    seed_cv = float(abs(tgt["seed_std"]) / max(abs(tgt["mean_of_seed_means"]), 1e-6))

    criteria = {
        "category": CATEGORY,
        "alpha": a,
        "target_mean": float(tgt["mean_of_seed_means"]),
        "target_ci95": [float(tgt["ci95_lo"]), float(tgt["ci95_hi"])],
        "controls_mean": ctr_mean,
        "controls_seed_std": ctr_std,
        "target_vs_controls_pass": bool(tgt["mean_of_seed_means"] > ctr_mean + 0.01),
        "wrong_layer_weaker_pass": bool(abs(float(wl["mean_of_seed_means"])) < abs(float(tgt["mean_of_seed_means"])) * 0.7),
        "wrong_hook_weaker_pass": bool(abs(float(wh["mean_of_seed_means"])) < abs(float(tgt["mean_of_seed_means"])) * 0.7),
        "seed_cv_effect": seed_cv,
        "seed_stability_pass_cv_lt_0p2": bool(seed_cv < 0.2),
        "n_seeds": len(SEEDS),
    }

    criteria_json = OUT / "gemma_code_seeded_effects_criteria.json"
    criteria_json.write_text(json.dumps(criteria, indent=2))

    # figure: per-seed alpha curves (target vs avg controls)
    seed_ctrl = (
        df[(df.group == "control") & (df.condition == "target")]
        .groupby(["seed", "alpha"], as_index=False)["mean_delta"].mean()
    )
    seed_tgt = df[(df.group == "target") & (df.condition == "target")][["seed", "alpha", "mean_delta"]].copy()

    plt.figure(figsize=(7, 4.6))
    for s in SEEDS:
        t = seed_tgt[seed_tgt.seed == s].sort_values("alpha")
        c = seed_ctrl[seed_ctrl.seed == s].sort_values("alpha")
        plt.plot(t.alpha, t.mean_delta, color="#1f77b4", alpha=0.28, lw=1)
        plt.plot(c.alpha, c.mean_delta, color="#ff7f0e", alpha=0.22, lw=1, ls="--")

    t_mean = seed_tgt.groupby("alpha", as_index=False)["mean_delta"].mean()
    c_mean = seed_ctrl.groupby("alpha", as_index=False)["mean_delta"].mean()
    plt.plot(t_mean.alpha, t_mean.mean_delta, color="#1f77b4", lw=2.6, label="Target (seed mean)")
    plt.plot(c_mean.alpha, c_mean.mean_delta, color="#ff7f0e", lw=2.4, ls="--", label="Controls avg (seed mean)")
    plt.axhline(0, color="gray", lw=1)
    plt.title("Code holdout: seeded steering curves")
    plt.xlabel("alpha")
    plt.ylabel("mean Δcontrast")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig_path = FIG / "gemma_code_seeded_effects_curves.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print(json.dumps({
        "rows_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "criteria_json": str(criteria_json),
        "figure": str(fig_path),
        "criteria": criteria,
    }, indent=2))


if __name__ == "__main__":
    main()
