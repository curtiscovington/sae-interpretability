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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

OUT_DIR = Path("outputs_gemma2_2b/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_RANDOM_CONTROLS = 8
BOOTSTRAPS = 2500
BATCH = 20

CATEGORY_CFG = {
    "code": {
        "target_feature": 6631,
        "target_words": [" code", " function", " python", " bug", " test", " api", " sql", " class"],
        "stems": [
            "Write a function that validates",
            "Fix the bug in this module by",
            "Refactor the class to improve",
            "Add a unit test for",
            "Implement an API endpoint that",
            "Optimize this loop to reduce",
            "Parse this JSON payload and",
            "Handle exceptions in this service by",
            "Document the interface for",
            "Review this pull request and",
            "Create a SQL query that finds",
            "Add type hints to",
            "Write pseudocode for",
            "Benchmark the function that",
            "Debug the failing test where",
            "Use dependency injection to",
            "Improve readability of",
            "Design a schema for",
            "Add input validation so",
            "Split this helper into",
        ],
        "neutral": [
            "Write a summary that validates",
            "Fix the wording in this memo by",
            "Refactor the paragraph to improve",
            "Add a checklist item for",
            "Implement a policy update that",
            "Optimize this process to reduce",
            "Parse this report and",
            "Handle objections in this meeting by",
            "Document the procedure for",
            "Review this proposal and",
            "Create a table that finds",
            "Add labels to",
            "Write an outline for",
            "Benchmark the plan that",
            "Debug the argument where",
            "Use delegation to",
            "Improve readability of",
            "Design a template for",
            "Add guardrails so",
            "Split this section into",
        ],
    },
    "safety": {
        "target_feature": 14033,
        "target_words": [" safe", " policy", " harmful", " refuse", " risk", " legal", " unsafe", " guidance"],
        "stems": [
            "Respond safely when asked to",
            "Decline this harmful request by",
            "Provide policy-compliant guidance for",
            "Address misuse risk in",
            "Refuse instructions that could",
            "Offer a safer alternative to",
            "Set a boundary around",
            "Avoid enabling wrongdoing when",
            "Explain why this request is unsafe because",
            "Redirect the user toward",
            "Handle high-risk content by",
            "Enforce responsible use when",
            "Mitigate potential harm from",
            "Refuse to provide instructions for",
            "Give de-escalation guidance for",
            "Prioritize user safety when",
            "Apply content policy to",
            "Provide legal-safe framing for",
            "Warn about consequences of",
            "Offer support resources if",
        ],
        "neutral": [
            "Respond clearly when asked to",
            "Decline this optional request by",
            "Provide practical guidance for",
            "Address uncertainty in",
            "Refine instructions that could",
            "Offer an alternative to",
            "Set a boundary around",
            "Avoid overpromising when",
            "Explain why this request is complex because",
            "Redirect the user toward",
            "Handle sensitive tone by",
            "Enforce consistent style when",
            "Mitigate confusion from",
            "Refuse to provide details for",
            "Give context guidance for",
            "Prioritize clarity when",
            "Apply formatting rules to",
            "Provide neutral framing for",
            "Warn about tradeoffs of",
            "Offer additional resources if",
        ],
    },
    "uncertainty": {
        "target_feature": 4223,
        "target_words": [" maybe", " might", " uncertain", " likely", " assume", " estimate", " confidence", " probably"],
        "stems": [
            "This estimate might change because",
            "I am not fully certain that",
            "Given limited data, we should assume",
            "The most likely explanation is",
            "There is uncertainty around",
            "A cautious conclusion would be",
            "Confidence is limited when",
            "I might be wrong, but",
            "A probabilistic view suggests",
            "Unknown factors include",
            "A tentative answer is",
            "This likely depends on",
            "The evidence is mixed because",
            "A reasonable assumption is",
            "The forecast is uncertain due to",
            "We should avoid overconfidence when",
            "The estimate probably shifts if",
            "One caveat is that",
            "This may vary by",
            "I would hedge this claim since",
        ],
        "neutral": [
            "This update changes because",
            "I am fully aware that",
            "Given available data, we should report",
            "The explanation is",
            "There is clarity around",
            "A direct conclusion would be",
            "Confidence is high when",
            "I am clear that",
            "A practical view suggests",
            "Key factors include",
            "A concise answer is",
            "This depends on",
            "The evidence is consistent because",
            "A simple assumption is",
            "The forecast is stable due to",
            "We should stay consistent when",
            "The estimate shifts if",
            "One note is that",
            "This varies by",
            "I would state this claim since",
        ],
    },
}

ENDINGS = [
    "the latest evidence.",
    "the user context.",
    "new constraints.",
    "practical limitations.",
    "a recent update.",
    "implementation details.",
    "stakeholder feedback.",
    "edge cases.",
    "timing and scope.",
    "unexpected conditions.",
]


def first_token_ids(tok, words):
    ids = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.append(int(t[0]))
    return sorted(set(ids))


def batch_mass(model, tok, prompts, target_ids, device):
    out = []
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True)
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            lengths = enc["attention_mask"].sum(dim=1) - 1
            row = torch.arange(logits.shape[0], device=device)
            last = logits[row, lengths.to(device), :]
            lp = torch.log_softmax(last, dim=-1)
            mass = torch.logsumexp(lp[:, tid], dim=-1)
            out.extend([float(x) for x in mass.cpu().tolist()])
    return np.array(out)


def feat_means(model, tok, sae, prompts, device):
    holder = []

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        holder.append(h.detach())
        return out

    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            h = model.model.layers[LAYER].register_forward_hook(hook)
            try:
                enc = tok(chunk, return_tensors="pt", padding=True)
                _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            finally:
                h.remove()

    ms = []
    for act in holder:
        z = sae.encode(act.float())
        ms.append(z.mean(dim=(0, 1)).detach().cpu().numpy())
    return np.stack(ms).mean(axis=0)


def hook_factory(sae, feature_idx, alpha):
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


def bootstrap_ci(vals, n=2500, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for i in range(n):
        s = rng.choice(vals, size=len(vals), replace=True)
        boots[i] = s.mean()
    return tuple(np.percentile(boots, [2.5, 97.5]).tolist())


def build_pairs(stems, neutral):
    pairs = []
    idx = 1
    for s, n in zip(stems, neutral):
        for e in ENDINGS:
            pairs.append({"id": f"p{idx}", "a": f"{s} {e}", "b": f"{n} {e}"})
            idx += 1
    return pairs


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    all_reports = {}
    for cat, cfg in CATEGORY_CFG.items():
        pairs = build_pairs(cfg["stems"], cfg["neutral"])
        pa = [p["a"] for p in pairs]
        pb = [p["b"] for p in pairs]
        pids = [p["id"] for p in pairs]
        tids = first_token_ids(tok, cfg["target_words"])

        m = feat_means(model, tok, sae, pa[:80], device)
        tgt = cfg["target_feature"]
        ord_idx = np.argsort(np.abs(m - float(m[tgt])))
        controls = []
        for i in ord_idx:
            i = int(i)
            if i == tgt:
                continue
            controls.append(i)
            if len(controls) >= N_RANDOM_CONTROLS:
                break
        features = [tgt] + controls

        base_a = batch_mass(model, tok, pa, tids, device)
        base_b = batch_mass(model, tok, pb, tids, device)
        base = base_a - base_b

        rows = []
        for feat in features:
            for alpha in ALPHAS:
                h = model.model.layers[LAYER].register_forward_hook(hook_factory(sae, feat, alpha))
                try:
                    sa = batch_mass(model, tok, pa, tids, device)
                    sb = batch_mass(model, tok, pb, tids, device)
                finally:
                    h.remove()
                delta = (sa - sb) - base
                for i in range(len(pairs)):
                    rows.append({
                        "category": cat,
                        "pair_id": pids[i],
                        "feature": feat,
                        "is_target": int(feat == tgt),
                        "alpha": alpha,
                        "delta_contrast": float(delta[i]),
                    })

        df = pd.DataFrame(rows)
        raw = OUT_DIR / f"gemma_{cat}_200pairs_with_controls.csv"
        df.to_csv(raw, index=False)

        sums = []
        for (feat, alpha), g in df.groupby(["feature", "alpha"]):
            vals = g["delta_contrast"].to_numpy()
            lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=100 + int(feat) % 997 + int(alpha * 100))
            sums.append({
                "category": cat,
                "feature": int(feat),
                "is_target": int(feat == tgt),
                "alpha": float(alpha),
                "mean_delta_contrast": float(vals.mean()),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "n_pairs": int(len(vals)),
            })
        sdf = pd.DataFrame(sums).sort_values(["is_target", "feature", "alpha"], ascending=[False, True, True])
        summary = OUT_DIR / f"gemma_{cat}_200pairs_with_controls_summary.csv"
        sdf.to_csv(summary, index=False)

        cdf = df[df.is_target == 0].groupby(["alpha", "pair_id"], as_index=False).agg(delta_contrast=("delta_contrast", "mean"))
        agg = []
        for alpha, g in cdf.groupby("alpha"):
            vals = g["delta_contrast"].to_numpy()
            lo, hi = bootstrap_ci(vals, n=BOOTSTRAPS, seed=777 + int(alpha * 100))
            agg.append({
                "category": cat,
                "alpha": float(alpha),
                "controls_mean_delta_contrast": float(vals.mean()),
                "controls_ci95_lo": float(lo),
                "controls_ci95_hi": float(hi),
                "n_pairs": int(len(vals)),
            })
        adf = pd.DataFrame(agg).sort_values("alpha")
        agg_path = OUT_DIR / f"gemma_{cat}_200pairs_controls_aggregate.csv"
        adf.to_csv(agg_path, index=False)

        pair_sample = pairs[:8]
        (OUT_DIR / f"gemma_{cat}_200pairs_pair_sample.json").write_text(json.dumps(pair_sample, indent=2))

        all_reports[cat] = {
            "target_feature": tgt,
            "controls": controls,
            "pairs": len(pairs),
            "raw": str(raw),
            "summary": str(summary),
            "controls_aggregate": str(agg_path),
            "pair_sample": str(OUT_DIR / f"gemma_{cat}_200pairs_pair_sample.json"),
        }

    (OUT_DIR / "gemma_multi_category_200pairs_report.json").write_text(json.dumps(all_reports, indent=2))
    print("Wrote multi-category 200-pair reports")


if __name__ == "__main__":
    main()
