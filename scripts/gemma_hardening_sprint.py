from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
WRONG_LAYER = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
OUT_DIR = Path("outputs_gemma2_2b/features")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 1.5]
N_MATCHED_CONTROLS = 4
N_ABLATION_NEIGHBORS = 4
SEEDS = [7, 17, 27]
BATCH = 16
GEN_TOKENS = 32

CONTEXT_TRAIN = [
    "in the first draft.", "during a planning meeting.", "for a short internal memo.",
]
CONTEXT_HOLDOUT = [
    "while triaging a live incident.", "for an onboarding walkthrough.", "during a postmortem discussion.",
]

CATEGORY_CFG = {
    "sports": {
        "target_feature": 16242,
        "target_words": [" game", " team", " season", " coach", " match", " playoffs", " score", " league"],
        "pairs": [
            ("The team finished the season with a strong defense", "The company finished the quarter with a clear plan"),
            ("The coach adjusted the lineup before the match", "The manager adjusted the agenda before the meeting"),
            ("The player scored twice in the final game", "The speaker repeated twice in the final section"),
            ("Fans expected a playoff run after the trade", "Readers expected strong sales after the launch"),
            ("The league announced a rule change for safety", "The board announced a policy change for clarity"),
            ("The striker found space and scored early", "The presenter found space and spoke early"),
            ("The defense held the lead late in the game", "The team held the timeline late in the project"),
            ("The rookie improved each week of the season", "The trainee improved each week of the program"),
        ],
    },
    "code": {
        "target_feature": 6631,
        "target_words": [" code", " function", " python", " bug", " test", " api", " class", " query"],
        "pairs": [
            ("Write a Python function that validates user input", "Write a clear procedure that validates user input"),
            ("Refactor this class to reduce duplicated code", "Refactor this section to reduce duplicated wording"),
            ("Add unit tests for this function", "Add checklist items for this process"),
            ("Debug the failing test in this module", "Review the failing argument in this section"),
            ("Implement an API endpoint for this feature", "Implement a policy update for this feature"),
            ("Parse this JSON payload safely", "Parse this report summary carefully"),
            ("Handle exceptions in this service", "Handle objections in this discussion"),
            ("Write a SQL query to find duplicates", "Write a table summary to find duplicates"),
        ],
    },
    "safety": {
        "target_feature": 14033,
        "target_words": [" safe", " policy", " harmful", " refuse", " risk", " unsafe", " legal", " guidance"],
        "pairs": [
            ("Provide a safe response that refuses harmful instructions", "Provide a direct response that addresses the request"),
            ("Explain why this request is unsafe and offer alternatives", "Explain why this request is complex and offer alternatives"),
            ("Apply policy constraints before answering", "Apply style constraints before answering"),
            ("Decline to provide dangerous procedural details", "Decline to provide unnecessary procedural details"),
            ("Redirect the user to a safer option", "Redirect the user to a practical option"),
            ("Warn about misuse risk in this scenario", "Warn about confusion risk in this scenario"),
            ("Use harm-minimizing language in the reply", "Use concise language in the reply"),
            ("Recommend legal and safe alternatives", "Recommend clear and practical alternatives"),
        ],
    },
    "uncertainty": {
        "target_feature": 4223,
        "target_words": [" maybe", " might", " uncertain", " likely", " assume", " estimate", " probably", " confidence"],
        "pairs": [
            ("I might be wrong, but this estimate could change", "This estimate is stable and should hold"),
            ("Given limited evidence, we should be cautious", "Given available evidence, we should proceed"),
            ("This conclusion is uncertain and needs validation", "This conclusion is clear and ready"),
            ("A likely explanation is possible but unconfirmed", "An explanation is clear and confirmed"),
            ("Confidence is moderate due to missing data", "Confidence is strong with current data"),
            ("Assumptions may break under new conditions", "Assumptions hold under current conditions"),
            ("There is uncertainty around this prediction", "There is clarity around this prediction"),
            ("I would report this with caveats", "I would report this without caveats"),
        ],
    },
}


@dataclass
class HookSpec:
    name: str
    register: Callable


def first_token_ids(tok, words):
    return sorted({int(tok.encode(w, add_special_tokens=False)[0]) for w in words if tok.encode(w, add_special_tokens=False)})


def build_prompts(pairs, contexts):
    ids, a, b = [], [], []
    n = 0
    for pidx, (pa, pb) in enumerate(pairs):
        for c in contexts:
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


def feature_stats(model, tok, sae, prompts_a, prompts_b, layer_idx, target_feature, device):
    feats_a, feats_b = [], []

    def collect(prompts, bucket):
        with torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                acts = []

                def hk(_m, _i, out):
                    acts.append((out[0] if isinstance(out, tuple) else out).detach())
                    return out

                h = model.model.layers[layer_idx].register_forward_hook(hk)
                try:
                    enc = tok(prompts[i:i+BATCH], return_tensors="pt", padding=True)
                    _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                finally:
                    h.remove()
                z = sae.encode(acts[0].float()).mean(dim=(0, 1)).cpu().numpy()
                bucket.append(z)

    collect(prompts_a, feats_a)
    collect(prompts_b, feats_b)
    A = np.stack(feats_a)
    B = np.stack(feats_b)
    mu = A.mean(axis=0)
    var = A.var(axis=0)
    sel = (A.mean(axis=0) - B.mean(axis=0)) / (A.std(axis=0) + B.std(axis=0) + 1e-6)
    target_vec = np.array([mu[target_feature], var[target_feature], sel[target_feature]])
    cand = np.stack([mu, var, sel], axis=1)
    d = ((cand - target_vec) / (cand.std(axis=0) + 1e-8)) ** 2
    score = d.sum(axis=1)
    order = np.argsort(score)
    ctr = []
    for i in order:
        i = int(i)
        if i == target_feature:
            continue
        ctr.append(i)
        if len(ctr) >= N_MATCHED_CONTROLS:
            break
    return ctr, {"mu": float(target_vec[0]), "var": float(target_vec[1]), "sel": float(target_vec[2])}


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


def get_hooks(model, sae, feature, alpha):
    return {
        "target": HookSpec("target", lambda: model.model.layers[LAYER].register_forward_hook(steer_hook(sae, feature, alpha))),
        "wrong_layer": HookSpec("wrong_layer", lambda: model.model.layers[WRONG_LAYER].register_forward_hook(steer_hook(sae, feature, alpha))),
        "wrong_hook": HookSpec("wrong_hook", lambda: model.model.layers[LAYER].mlp.register_forward_hook(steer_hook(sae, feature, alpha))),
    }


def evaluate_contrast(model, tok, prompts_a, prompts_b, target_ids, hook: HookSpec | None, device):
    h = hook.register() if hook else None
    try:
        la = batch_last_logits(model, tok, prompts_a, device)
        lb = batch_last_logits(model, tok, prompts_b, device)
    finally:
        if h:
            h.remove()
    return target_logmass_from_last(la, target_ids) - target_logmass_from_last(lb, target_ids)


def quality_metrics(model, tok, prompts, hook, seed, device):
    torch.manual_seed(seed)
    out_rows = []
    h = hook.register() if hook else None
    try:
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i:i+BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=GEN_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
                logits = model(**enc).logits
            for j, p in enumerate(chunk):
                completion_ids = gen_ids[j, enc["input_ids"].shape[1]:]
                completion = tok.decode(completion_ids, skip_special_tokens=True)
                words = [w.strip(".,:;!?").lower() for w in completion.split() if w.strip()]
                uniq = len(set(words))
                coherence = uniq / max(len(words), 1)
                lp = torch.log_softmax(logits[j, -1, :].float(), dim=-1)
                entropy = float((-(lp.exp() * lp)).sum().item())
                prompt_kw = {w.lower().strip(".,:;!?") for w in p.split() if len(w) > 5}
                keep = sum(int(k in completion.lower()) for k in list(prompt_kw)[:6]) / max(min(len(prompt_kw), 6), 1)
                out_rows.append({"prompt": p, "coherence_proxy": coherence, "entropy_proxy": entropy, "task_retention": keep})
    finally:
        if h:
            h.remove()
    return pd.DataFrame(out_rows)


def bootstrap_ci(v, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    return np.percentile(boots, [2.5, 97.5]).tolist()


def nearest_neighbors_from_decoder(sae, feature, k):
    W = sae.W_dec.float()
    v = F.normalize(W[feature], dim=0)
    sims = F.normalize(W, dim=1) @ v
    vals, idx = torch.topk(sims, k=k + 1)
    out = []
    for i, s in zip(idx.tolist(), vals.tolist()):
        if i == feature:
            continue
        out.append((int(i), float(s)))
        if len(out) >= k:
            break
    return out


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    all_rows = []
    quality_rows = []
    ablation_rows = []
    criteria_rows = []

    for cat, cfg in CATEGORY_CFG.items():
        ids_tr, a_tr, b_tr = build_prompts(cfg["pairs"], CONTEXT_TRAIN)
        ids_ho, a_ho, b_ho = build_prompts(cfg["pairs"], CONTEXT_HOLDOUT)
        tids = first_token_ids(tok, cfg["target_words"])
        ctr, stats = feature_stats(model, tok, sae, a_tr[:24], b_tr[:24], LAYER, cfg["target_feature"], device)

        base_tr = evaluate_contrast(model, tok, a_tr, b_tr, tids, None, device)
        base_ho = evaluate_contrast(model, tok, a_ho, b_ho, tids, None, device)

        # target + matched random controls with wrong-layer/wrong-hook controls
        for feat in [cfg["target_feature"]] + ctr:
            for alpha in ALPHAS:
                hooks = get_hooks(model, sae, feat, alpha)
                for cond in ["target", "wrong_layer", "wrong_hook"]:
                    dtr = evaluate_contrast(model, tok, a_tr, b_tr, tids, hooks[cond], device) - base_tr
                    dho = evaluate_contrast(model, tok, a_ho, b_ho, tids, hooks[cond], device) - base_ho
                    lo_tr, hi_tr = bootstrap_ci(dtr, seed=11 + feat % 97 + int(alpha * 10))
                    lo_ho, hi_ho = bootstrap_ci(dho, seed=31 + feat % 97 + int(alpha * 10))
                    all_rows.append({
                        "category": cat,
                        "feature": int(feat),
                        "is_target": int(feat == cfg["target_feature"]),
                        "alpha": float(alpha),
                        "condition": cond,
                        "split": "train",
                        "mean_delta": float(dtr.mean()),
                        "ci_lo": float(lo_tr),
                        "ci_hi": float(hi_tr),
                    })
                    all_rows.append({
                        "category": cat,
                        "feature": int(feat),
                        "is_target": int(feat == cfg["target_feature"]),
                        "alpha": float(alpha),
                        "condition": cond,
                        "split": "holdout",
                        "mean_delta": float(dho.mean()),
                        "ci_lo": float(lo_ho),
                        "ci_hi": float(hi_ho),
                    })

        # quality + seed stability at key alpha
        alpha_key = 1.5
        target_hook = get_hooks(model, sae, cfg["target_feature"], alpha_key)["target"]
        ctrl_hook = get_hooks(model, sae, ctr[0], alpha_key)["target"]
        for s in SEEDS:
            q0 = quality_metrics(model, tok, a_ho[:8], None, s, device)
            qt = quality_metrics(model, tok, a_ho[:8], target_hook, s, device)
            qc = quality_metrics(model, tok, a_ho[:8], ctrl_hook, s, device)
            for name, q in [("baseline", q0), ("target", qt), ("control", qc)]:
                quality_rows.append({
                    "category": cat,
                    "seed": s,
                    "arm": name,
                    "coherence_proxy": float(q["coherence_proxy"].mean()),
                    "entropy_proxy": float(q["entropy_proxy"].mean()),
                    "task_retention": float(q["task_retention"].mean()),
                })

        # ablation panel target vs decoder-neighbors vs matched control at key alpha
        neighbors = nearest_neighbors_from_decoder(sae, cfg["target_feature"], N_ABLATION_NEIGHBORS)
        panel = [(cfg["target_feature"], 1.0, "target")] + [(f, c, "neighbor") for f, c in neighbors] + [(ctr[0], np.nan, "matched_control")]
        for feat, cos, kind in panel:
            h = get_hooks(model, sae, feat, alpha_key)["target"]
            d = evaluate_contrast(model, tok, a_ho, b_ho, tids, h, device) - base_ho
            ablation_rows.append({
                "category": cat,
                "feature": int(feat),
                "kind": kind,
                "decoder_cosine_to_target": float(cos),
                "mean_delta_holdout": float(d.mean()),
            })

        # criterion flags
        R = pd.DataFrame([r for r in all_rows if r["category"] == cat and r["split"] == "holdout" and r["alpha"] == 1.5])
        tgt = R[(R["is_target"] == 1) & (R["condition"] == "target")]["mean_delta"].mean()
        ctrl = R[(R["is_target"] == 0) & (R["condition"] == "target")]["mean_delta"].mean()
        wl = R[(R["is_target"] == 1) & (R["condition"] == "wrong_layer")]["mean_delta"].mean()
        wh = R[(R["is_target"] == 1) & (R["condition"] == "wrong_hook")]["mean_delta"].mean()

        qdf = pd.DataFrame([q for q in quality_rows if q["category"] == cat])
        q_target = qdf[qdf["arm"] == "target"]
        q_ctrl = qdf[qdf["arm"] == "control"]
        q_base = qdf[qdf["arm"] == "baseline"]
        coh_drop = q_target["coherence_proxy"].mean() - q_base["coherence_proxy"].mean()
        ent_shift = abs(q_target["entropy_proxy"].mean() - q_base["entropy_proxy"].mean())
        task_ret = q_target["task_retention"].mean() - q_ctrl["task_retention"].mean()
        seed_cv = q_target["task_retention"].std() / max(abs(q_target["task_retention"].mean()), 1e-6)

        criteria_rows.append({
            "category": cat,
            "target_vs_control": bool(tgt > ctrl + 0.01),
            "wrong_layer_weaker": bool(abs(wl) < abs(tgt) * 0.7),
            "wrong_hook_weaker": bool(abs(wh) < abs(tgt) * 0.7),
            "holdout_transfer": bool(tgt > 0.0),
            "quality_ok": bool(coh_drop > -0.08 and ent_shift < 0.8 and task_ret > -0.02),
            "multi_seed_stable": bool(seed_cv < 0.2),
            "ablation_specific": bool(
                (pd.DataFrame([a for a in ablation_rows if a["category"] == cat and a["kind"] == "target"])["mean_delta_holdout"].mean())
                > (pd.DataFrame([a for a in ablation_rows if a["category"] == cat and a["kind"] == "neighbor"])["mean_delta_holdout"].mean() + 0.01)
            ),
            "target_mean_delta_holdout_alpha1p5": float(tgt),
            "controls_mean_delta_holdout_alpha1p5": float(ctrl),
            "wrong_layer_mean_delta": float(wl),
            "wrong_hook_mean_delta": float(wh),
        })

        # figure per category
        cdf = pd.DataFrame([r for r in all_rows if r["category"] == cat and r["split"] == "holdout"])
        t = cdf[(cdf["is_target"] == 1) & (cdf["condition"] == "target")].sort_values("alpha")
        c = cdf[(cdf["is_target"] == 0) & (cdf["condition"] == "target")].groupby("alpha", as_index=False)["mean_delta"].mean()
        wl_df = cdf[(cdf["is_target"] == 1) & (cdf["condition"] == "wrong_layer")].sort_values("alpha")
        wh_df = cdf[(cdf["is_target"] == 1) & (cdf["condition"] == "wrong_hook")].sort_values("alpha")
        plt.figure(figsize=(6, 4))
        plt.plot(t["alpha"], t["mean_delta"], label="target")
        plt.plot(c["alpha"], c["mean_delta"], "--", label="matched controls")
        plt.plot(wl_df["alpha"], wl_df["mean_delta"], ":", label="wrong layer")
        plt.plot(wh_df["alpha"], wh_df["mean_delta"], "-.", label="wrong hook")
        plt.axhline(0, color="gray", linewidth=1)
        plt.title(f"{cat.title()} hardening curves (holdout)")
        plt.xlabel("alpha")
        plt.ylabel("mean Δcontrast")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"gemma_hardening_{cat}_controls_holdout.png", dpi=170)
        plt.close()

    all_df = pd.DataFrame(all_rows)
    q_df = pd.DataFrame(quality_rows)
    a_df = pd.DataFrame(ablation_rows)
    c_df = pd.DataFrame(criteria_rows)

    all_df.to_csv(OUT_DIR / "gemma_hardening_full_rows.csv", index=False)
    q_df.to_csv(OUT_DIR / "gemma_hardening_quality_seed_rows.csv", index=False)
    a_df.to_csv(OUT_DIR / "gemma_hardening_ablation_rows.csv", index=False)
    c_df.to_csv(OUT_DIR / "gemma_hardening_criteria_table.csv", index=False)

    report = {
        "model": MODEL_NAME,
        "sae": {"release": SAE_RELEASE, "id": SAE_ID, "layer": LAYER},
        "seeds": SEEDS,
        "alphas": ALPHAS,
        "criteria": c_df.to_dict(orient="records"),
        "artifacts": {
            "full_rows_csv": str(OUT_DIR / "gemma_hardening_full_rows.csv"),
            "quality_csv": str(OUT_DIR / "gemma_hardening_quality_seed_rows.csv"),
            "ablation_csv": str(OUT_DIR / "gemma_hardening_ablation_rows.csv"),
            "criteria_csv": str(OUT_DIR / "gemma_hardening_criteria_table.csv"),
            "figures_dir": str(FIG_DIR),
        },
    }
    (OUT_DIR / "gemma_hardening_report.json").write_text(json.dumps(report, indent=2))

    md = ["# Gemma SAE hardening sprint report", "", "## Summary table (holdout, alpha=1.5)", "", "|Category|Target Δ|Controls Δ|Wrong-layer weaker|Wrong-hook weaker|Transfer|Quality|Seed stability|Ablation specificity|", "|---|---:|---:|---|---|---|---|---|---|"]
    for _, r in c_df.iterrows():
        md.append(
            f"|{r['category']}|{r['target_mean_delta_holdout_alpha1p5']:.4f}|{r['controls_mean_delta_holdout_alpha1p5']:.4f}|{'PASS' if r['wrong_layer_weaker'] else 'FAIL'}|{'PASS' if r['wrong_hook_weaker'] else 'FAIL'}|{'PASS' if r['holdout_transfer'] else 'FAIL'}|{'PASS' if r['quality_ok'] else 'FAIL'}|{'PASS' if r['multi_seed_stable'] else 'FAIL'}|{'PASS' if r['ablation_specific'] else 'FAIL'}|"
        )
    md += ["", "## Notes", "- Controls are matched on activation mean + variance + selectivity profile.", "- Holdout uses unseen context templates only.", "- Quality metrics: coherence proxy (distinct-token ratio), entropy proxy (next-token entropy), task-retention proxy (prompt keyword carryover)."]
    (OUT_DIR / "gemma_hardening_report.md").write_text("\n".join(md))
    print("Wrote hardening artifacts in", OUT_DIR)


if __name__ == "__main__":
    main()
