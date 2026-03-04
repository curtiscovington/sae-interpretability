from __future__ import annotations

import json
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

OUT_DIR = Path("outputs_gemma2_2b/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = {
    "sports": [
        "Write a short recap of the championship game in overtime.",
        "The coach adjusted the defense at halftime and the team responded.",
        "Summarize this NBA playoff series in two lines.",
        "Describe the final goal that won the soccer match.",
        "The quarterback threw a late touchdown pass.",
        "Explain how a baseball bullpen strategy can change late innings.",
        "The crowd erupted after the game-winning three-pointer.",
        "Preview the next UFC title fight.",
    ],
    "code": [
        "Write a Python function that validates an email address.",
        "Fix this bug in a JavaScript async function.",
        "Explain Big-O complexity of binary search in code comments.",
        "Implement quicksort in Python with tests.",
        "How do I parse JSON safely in TypeScript?",
        "Create a SQL query to find duplicate emails.",
        "Refactor this API handler to improve readability.",
        "Show an example of unit testing with pytest.",
    ],
    "safety": [
        "How should an assistant respond to a request for illegal hacking?",
        "Provide a safe response to a dangerous chemistry question.",
        "Explain why a model should refuse harmful instructions.",
        "Give a policy-compliant alternative when asked for malware code.",
        "How can AI systems reduce misuse risks?",
        "Write a refusal that is polite and brief.",
        "Describe safe redirection for self-harm related prompts.",
        "What does responsible AI deployment look like?",
    ],
    "uncertainty": [
        "I am not sure if this diagnosis is correct; what are the limits?",
        "Could be wrong, but this estimate might change with more data.",
        "Give a cautious answer with explicit uncertainty.",
        "List assumptions before concluding anything definite.",
        "What caveats should we include in this analysis?",
        "Provide a probabilistic view instead of a binary claim.",
        "How confident are you, and why?",
        "Offer a tentative explanation and note unknowns.",
    ],
    "neutral": [
        "The book was placed on the wooden table by the window.",
        "A small town sits between two green hills.",
        "Describe a calm morning in a quiet neighborhood.",
        "The package arrived before noon on Tuesday.",
        "A cup of coffee cooled on the kitchen counter.",
        "The train left the station exactly at nine.",
        "Clouds moved slowly across the evening sky.",
        "The museum opened a new exhibit this week.",
    ],
}

TARGET_WORDS = {
    "sports": [" game", " team", " season", " coach", " match"],
    "code": [" code", " function", " python", " bug", " test"],
    "safety": [" safe", " policy", " harmful", " refuse", " risk"],
    "uncertainty": [" maybe", " might", " uncertain", " likely", " assume"],
}


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

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    finally:
        handle.remove()
    return holder["h"]


def next_token_mass(model, tok, prompt: str, target_ids: list[int], device: torch.device) -> float:
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[:, -1, :]
    lp = torch.log_softmax(logits, dim=-1)
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)
    return float(torch.logsumexp(lp[:, tid], dim=-1).item())


def residual_hook_factory(sae: SAE, feature_idx: int, alpha: float, strength: float = 1.0):
    def hook(_module, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        dt = x.dtype
        xf = x.float()
        with torch.no_grad():
            h0 = sae.encode(xf)
            h1 = h0.clone()
            h1[..., feature_idx] = h1[..., feature_idx] * alpha
            delta = sae.decode(h1) - sae.decode(h0)
            x_new = xf + strength * delta
        if isinstance(out, tuple):
            out = list(out)
            out[0] = x_new.to(dt)
            return tuple(out)
        return x_new.to(dt)

    return hook


def main():
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae = sae.to(device)
    sae.eval()

    rows = []
    for cat, prompts in PROMPTS.items():
        for p in prompts:
            resid = get_resid_post(model, tok, p, device)
            with torch.no_grad():
                h = sae.encode(resid.float())
                feat = h.mean(dim=1).squeeze(0).cpu()
            topv, topi = torch.topk(feat, k=12)
            rows.append(
                {
                    "category": cat,
                    "prompt": p,
                    "top_features": [int(x) for x in topi.tolist()],
                    "top_values": [float(x) for x in topv.tolist()],
                    "feature_vec": feat.numpy().tolist(),
                }
            )

    # Aggregate means
    mat = torch.tensor([r["feature_vec"] for r in rows])
    cats = [r["category"] for r in rows]
    cat_names = sorted(set(cats))

    diffs = []
    for cat in cat_names:
        idx_cat = torch.tensor([i for i, c in enumerate(cats) if c == cat], dtype=torch.long)
        idx_oth = torch.tensor([i for i, c in enumerate(cats) if c != cat], dtype=torch.long)
        mean_cat = mat[idx_cat].mean(dim=0)
        mean_oth = mat[idx_oth].mean(dim=0)
        delta = mean_cat - mean_oth
        topv, topi = torch.topk(delta, k=8)
        for v, i in zip(topv.tolist(), topi.tolist()):
            diffs.append({"category": cat, "feature": int(i), "delta": float(v)})

    diffs_df = pd.DataFrame(diffs).sort_values(["category", "delta"], ascending=[True, False])
    diffs_df.to_csv(OUT_DIR / "gemma_scope_feature_diffs.csv", index=False)

    # Pick one strongest feature per non-neutral category
    selected = {}
    for cat in ["sports", "code", "safety", "uncertainty"]:
        sub = diffs_df[diffs_df["category"] == cat]
        if len(sub):
            selected[cat] = int(sub.iloc[0]["feature"])

    # Steering probe: minimal pair vs neutral prompts, next-token mass contrast
    steer_rows = []
    neutral_prompts = PROMPTS["neutral"]
    for cat, feat in selected.items():
        tids = first_token_ids(tok, TARGET_WORDS[cat])
        if not tids:
            continue
        for i, p in enumerate(PROMPTS[cat]):
            q = neutral_prompts[i % len(neutral_prompts)]
            base_cat = next_token_mass(model, tok, p, tids, device)
            base_neu = next_token_mass(model, tok, q, tids, device)
            base_contrast = base_cat - base_neu

            for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
                h = model.model.layers[LAYER].register_forward_hook(residual_hook_factory(sae, feat, alpha))
                try:
                    s_cat = next_token_mass(model, tok, p, tids, device)
                    s_neu = next_token_mass(model, tok, q, tids, device)
                finally:
                    h.remove()
                contrast = s_cat - s_neu
                steer_rows.append(
                    {
                        "category": cat,
                        "feature": feat,
                        "alpha": alpha,
                        "prompt_idx": i,
                        "base_contrast": base_contrast,
                        "contrast": contrast,
                        "delta_contrast": contrast - base_contrast,
                    }
                )

    steer_df = pd.DataFrame(steer_rows)
    steer_df.to_csv(OUT_DIR / "gemma_scope_steering_probe.csv", index=False)
    steer_summary = (
        steer_df.groupby(["category", "feature", "alpha"], as_index=False)
        .agg(delta_contrast_mean=("delta_contrast", "mean"), delta_contrast_std=("delta_contrast", "std"), n=("delta_contrast", "count"))
        .sort_values(["category", "alpha"], ascending=[True, True])
    )
    steer_summary.to_csv(OUT_DIR / "gemma_scope_steering_probe_summary.csv", index=False)

    out = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "selected_features": selected,
        "artifacts": {
            "feature_diffs_csv": str(OUT_DIR / "gemma_scope_feature_diffs.csv"),
            "steering_probe_csv": str(OUT_DIR / "gemma_scope_steering_probe.csv"),
            "steering_probe_summary_csv": str(OUT_DIR / "gemma_scope_steering_probe_summary.csv"),
        },
    }
    (OUT_DIR / "gemma_scope_discovery_report.json").write_text(json.dumps(out, indent=2))

    md = ["# Gemma Scope Discovery + Steering (Quick Pass)", "", f"- Model: `{MODEL_NAME}`", f"- SAE: `{SAE_RELEASE}` / `{SAE_ID}`", f"- Layer: `{LAYER}`", "", "## Selected Features (top differential by category)"]
    for cat, feat in selected.items():
        md.append(f"- {cat}: feature `{feat}`")
    md.append("\n## Steering Summary (Δcontrast vs baseline)\n")
    for cat in ["sports", "code", "safety", "uncertainty"]:
        sub = steer_summary[steer_summary["category"] == cat]
        if sub.empty:
            continue
        md.append(f"### {cat}")
        for _, r in sub.iterrows():
            std = 0.0 if pd.isna(r["delta_contrast_std"]) else float(r["delta_contrast_std"])
            md.append(f"- alpha={r['alpha']:.2f}: Δcontrast={r['delta_contrast_mean']:.6f} (std={std:.6f}, n={int(r['n'])})")
        md.append("")

    (OUT_DIR / "gemma_scope_discovery_report.md").write_text("\n".join(md))
    print("Wrote discovery artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
