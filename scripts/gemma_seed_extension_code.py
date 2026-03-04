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
OUT = Path("outputs_gemma2_2b/features")

CAT = "code"
TARGET_FEATURE = 6631
CONTROL_FEATURE = 743  # strongest matched control from prior run
ALPHA = 1.5
BATCH = 16
GEN_TOKENS = 32
SEEDS_BASE = [7, 17, 27]
SEEDS_EXTRA = [37, 47, 57, 67]
SEEDS_ALL = SEEDS_BASE + SEEDS_EXTRA

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
    "while triaging a live incident.", "for an onboarding walkthrough.", "during a postmortem discussion.",
]


def build_prompts():
    prompts = []
    for a, _ in PAIRS:
        for c in CONTEXT_HOLDOUT:
            prompts.append(f"{a} {c}")
    return prompts


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


def quality_metrics(model, tok, prompts, hook, seed, device):
    torch.manual_seed(seed)
    out_rows = []
    h = hook() if hook else None
    try:
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i:i+BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                gen_ids = model.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
                logits = model(**enc).logits
            for j, p in enumerate(chunk):
                completion_ids = gen_ids[j, enc["input_ids"].shape[1]:]
                completion = tok.decode(completion_ids, skip_special_tokens=True)
                words = [w.strip(".,:;!?").lower() for w in completion.split() if w.strip()]
                coherence = len(set(words)) / max(len(words), 1)
                lp = torch.log_softmax(logits[j, -1, :].float(), dim=-1)
                entropy = float((-(lp.exp() * lp)).sum().item())
                prompt_kw = {w.lower().strip(".,:;!?") for w in p.split() if len(w) > 5}
                keep = sum(int(k in completion.lower()) for k in list(prompt_kw)[:6]) / max(min(len(prompt_kw), 6), 1)
                out_rows.append({"coherence_proxy": coherence, "entropy_proxy": entropy, "task_retention": keep})
    finally:
        if h:
            h.remove()
    return pd.DataFrame(out_rows)


def summarize(df, arm):
    return {
        "arm": arm,
        "coherence_mean": float(df["coherence_proxy"].mean()),
        "entropy_mean": float(df["entropy_proxy"].mean()),
        "task_retention_mean": float(df["task_retention"].mean()),
    }


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    prompts = build_prompts()
    rows = []
    for s in SEEDS_ALL:
        q0 = quality_metrics(model, tok, prompts, None, s, device)
        qt = quality_metrics(model, tok, prompts, lambda: model.model.layers[LAYER].register_forward_hook(steer_hook(sae, TARGET_FEATURE, ALPHA)), s, device)
        qc = quality_metrics(model, tok, prompts, lambda: model.model.layers[LAYER].register_forward_hook(steer_hook(sae, CONTROL_FEATURE, ALPHA)), s, device)
        for arm, q in [("baseline", q0), ("target", qt), ("control", qc)]:
            rows.append({"seed": s, **summarize(q, arm)})

    rdf = pd.DataFrame(rows)
    out_csv = OUT / "gemma_code_seed_extension_quality.csv"
    rdf.to_csv(out_csv, index=False)

    target = rdf[rdf.arm == "target"]
    control = rdf[rdf.arm == "control"]
    seed_cv = float(target["task_retention_mean"].std() / max(abs(target["task_retention_mean"].mean()), 1e-6))
    delta_task = float(target["task_retention_mean"].mean() - control["task_retention_mean"].mean())

    report = {
        "category": CAT,
        "alpha": ALPHA,
        "seeds_base": SEEDS_BASE,
        "seeds_extra": SEEDS_EXTRA,
        "seeds_all": SEEDS_ALL,
        "target_feature": TARGET_FEATURE,
        "control_feature": CONTROL_FEATURE,
        "seed_cv_task_retention_target": seed_cv,
        "target_minus_control_task_retention_mean": delta_task,
        "passes_seed_stability_cv_lt_0p2": bool(seed_cv < 0.2),
        "quality_csv": str(out_csv),
    }
    out_json = OUT / "gemma_code_seed_extension_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
