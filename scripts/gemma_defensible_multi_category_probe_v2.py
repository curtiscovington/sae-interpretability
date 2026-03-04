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
BOOTSTRAPS = 2000
BATCH = 20

CONTEXTS = [
    "in the first draft.", "during a planning meeting.", "for a short internal memo.", "before the deadline.",
    "in a technical review.", "as a concise update.", "for a stakeholder briefing.", "in a training example.",
    "during a retrospective.", "for a practical guide.",
]

CATEGORY_CFG = {
    "code": {
        "target_feature": 6631,
        "target_words": [" code", " function", " python", " bug", " test", " api", " class", " query"],
        "pairs": [
            ("Write a Python function that validates user input", "Write a clear procedure that validates user input"),
            ("Refactor this class to reduce duplicated code", "Refactor this section to reduce duplicated wording"),
            ("Add unit tests for this function", "Add checklist items for this process"),
            ("Debug the failing test in this module", "Review the failing argument in this section"),
            ("Implement an API endpoint for this feature", "Implement a policy update for this feature"),
            ("Optimize this loop to improve runtime", "Optimize this workflow to improve speed"),
            ("Parse this JSON payload safely", "Parse this report summary carefully"),
            ("Handle exceptions in this service", "Handle objections in this discussion"),
            ("Write a SQL query to find duplicates", "Write a table summary to find duplicates"),
            ("Document this interface for developers", "Document this procedure for teammates"),
            ("Add type hints to this function", "Add labels to this checklist"),
            ("Benchmark this function with larger inputs", "Benchmark this process with larger samples"),
            ("Split this helper into two functions", "Split this section into two parts"),
            ("Review this pull request for correctness", "Review this proposal for consistency"),
            ("Create test fixtures for edge cases", "Create examples for edge cases"),
            ("Improve code readability in this file", "Improve readability in this paragraph"),
            ("Fix an off-by-one bug in this loop", "Fix an ordering issue in this list"),
            ("Design a schema for this API", "Design a template for this report"),
            ("Implement input validation before execution", "Implement quality checks before publication"),
            ("Add logging around this failure path", "Add notes around this decision path"),
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
            ("Refuse content that enables wrongdoing", "Refuse content that is out of scope"),
            ("Use harm-minimizing language in the reply", "Use concise language in the reply"),
            ("Set a clear safety boundary for this request", "Set a clear scope boundary for this request"),
            ("Offer emergency resources when needed", "Offer additional resources when needed"),
            ("Avoid giving instructions for illegal activity", "Avoid giving instructions for unrelated activity"),
            ("State policy limits before proceeding", "State formatting limits before proceeding"),
            ("Prioritize user safety over completeness", "Prioritize user clarity over completeness"),
            ("De-escalate risky intent in the response", "De-escalate tense tone in the response"),
            ("Prevent dual-use misuse in output", "Prevent misunderstanding in output"),
            ("Flag high-risk requests for refusal", "Flag low-context requests for clarification"),
            ("Provide non-actionable safety guidance", "Provide non-technical background guidance"),
            ("Avoid operational details that cause harm", "Avoid extra details that add noise"),
            ("Give a policy-compliant refusal", "Give a concise refusal"),
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
            ("We should treat this as a tentative forecast", "We should treat this as a final forecast"),
            ("Confidence is moderate due to missing data", "Confidence is strong with current data"),
            ("Assumptions may break under new conditions", "Assumptions hold under current conditions"),
            ("The result probably depends on context", "The result directly applies to context"),
            ("There is uncertainty around this prediction", "There is clarity around this prediction"),
            ("I would hedge this recommendation", "I would state this recommendation directly"),
            ("The evidence is mixed at this stage", "The evidence is consistent at this stage"),
            ("This may vary by environment", "This applies across environments"),
            ("We should avoid overconfidence here", "We can be confident here"),
            ("I am not fully certain about this claim", "I am fully certain about this claim"),
            ("A cautious interpretation is appropriate", "A direct interpretation is appropriate"),
            ("Unknown factors could affect the outcome", "Known factors explain the outcome"),
            ("This is likely but not guaranteed", "This is guaranteed in this setting"),
            ("Treat this as an initial estimate", "Treat this as the final estimate"),
            ("Further testing might revise this result", "Further testing should confirm this result"),
            ("I would report this with caveats", "I would report this without caveats"),
        ],
    },
}


def first_token_ids(tok, words):
    return sorted({int(tok.encode(w, add_special_tokens=False)[0]) for w in words if tok.encode(w, add_special_tokens=False)})


def batch_mass(model, tok, prompts, target_ids, device):
    tid = torch.tensor(target_ids, dtype=torch.long, device=device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i+BATCH], return_tensors="pt", padding=True)
            logits = model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device)).logits
            idx = enc['attention_mask'].sum(dim=1) - 1
            row = torch.arange(logits.shape[0], device=device)
            last = logits[row, idx.to(device), :]
            lp = torch.log_softmax(last, dim=-1)
            vals.extend(torch.logsumexp(lp[:, tid], dim=-1).cpu().tolist())
    return np.array(vals)


def feat_mean(model, tok, sae, prompts, device):
    acts = []
    def hk(_m,_i,out):
        acts.append((out[0] if isinstance(out,tuple) else out).detach())
        return out
    with torch.no_grad():
        for i in range(0,len(prompts),BATCH):
            h = model.model.layers[LAYER].register_forward_hook(hk)
            try:
                enc = tok(prompts[i:i+BATCH], return_tensors='pt', padding=True)
                _ = model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
            finally:
                h.remove()
    mats = [sae.encode(a.float()).mean(dim=(0,1)).detach().cpu().numpy() for a in acts]
    return np.stack(mats).mean(axis=0)


def steer_hook(sae, feat, alpha):
    def hk(_m,_i,out):
        x = out[0] if isinstance(out,tuple) else out
        dt = x.dtype
        xf = x.float()
        with torch.no_grad():
            h0 = sae.encode(xf)
            h1 = h0.clone(); h1[...,feat] = h1[...,feat]*alpha
            xnew = xf + (sae.decode(h1)-sae.decode(h0))
        if isinstance(out,tuple):
            o=list(out); o[0]=xnew.to(dt); return tuple(o)
        return xnew.to(dt)
    return hk


def boot(vals, n=2000, seed=42):
    rng=np.random.default_rng(seed)
    b=np.array([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(n)])
    return np.percentile(b,[2.5,97.5]).tolist()


def build_pairs(base_pairs):
    pairs=[]; idx=1
    for a,b in base_pairs:
        for c in CONTEXTS:
            pairs.append({"id":f"p{idx}","a":f"{a} {c}","b":f"{b} {c}"}); idx+=1
    return pairs


def main():
    device=torch.device(DEVICE)
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device).eval()
    sae=SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID).to(device).eval()

    report={}
    for cat,cfg in CATEGORY_CFG.items():
        pairs=build_pairs(cfg['pairs'])
        pa=[p['a'] for p in pairs]; pb=[p['b'] for p in pairs]; ids=[p['id'] for p in pairs]
        tids=first_token_ids(tok,cfg['target_words'])

        fm=feat_mean(model,tok,sae,pa[:80],device); tgt=cfg['target_feature']
        ord=np.argsort(np.abs(fm-fm[tgt])); ctr=[]
        for i in ord:
            i=int(i)
            if i==tgt: continue
            ctr.append(i)
            if len(ctr)>=N_RANDOM_CONTROLS: break

        base = batch_mass(model,tok,pa,tids,device)-batch_mass(model,tok,pb,tids,device)
        rows=[]
        for feat in [tgt]+ctr:
            for a in ALPHAS:
                h=model.model.layers[LAYER].register_forward_hook(steer_hook(sae,feat,a))
                try:
                    d=(batch_mass(model,tok,pa,tids,device)-batch_mass(model,tok,pb,tids,device))-base
                finally:
                    h.remove()
                for i,v in enumerate(d):
                    rows.append({"category":cat,"pair_id":ids[i],"feature":feat,"is_target":int(feat==tgt),"alpha":a,"delta_contrast":float(v)})
        df=pd.DataFrame(rows)
        raw=OUT_DIR/f"gemma_{cat}_200pairs_with_controls.csv"; df.to_csv(raw,index=False)

        s=[]
        for (feat,a),g in df.groupby(['feature','alpha']):
            vals=g.delta_contrast.to_numpy(); lo,hi=boot(vals,BOOTSTRAPS,100+int(feat)%997+int(a*100))
            s.append({"category":cat,"feature":int(feat),"is_target":int(feat==tgt),"alpha":float(a),"mean_delta_contrast":float(vals.mean()),"ci95_lo":float(lo),"ci95_hi":float(hi),"n_pairs":len(vals)})
        sdf=pd.DataFrame(s).sort_values(['is_target','feature','alpha'], ascending=[False,True,True])
        summary=OUT_DIR/f"gemma_{cat}_200pairs_with_controls_summary.csv"; sdf.to_csv(summary,index=False)

        c=df[df.is_target==0].groupby(['alpha','pair_id'],as_index=False).agg(delta_contrast=('delta_contrast','mean'))
        arows=[]
        for a,g in c.groupby('alpha'):
            vals=g.delta_contrast.to_numpy(); lo,hi=boot(vals,BOOTSTRAPS,777+int(a*100))
            arows.append({"category":cat,"alpha":float(a),"controls_mean_delta_contrast":float(vals.mean()),"controls_ci95_lo":float(lo),"controls_ci95_hi":float(hi),"n_pairs":len(vals)})
        adf=pd.DataFrame(arows).sort_values('alpha')
        agg=OUT_DIR/f"gemma_{cat}_200pairs_controls_aggregate.csv"; adf.to_csv(agg,index=False)

        sample_path=OUT_DIR/f"gemma_{cat}_200pairs_pair_sample.json"
        sample_path.write_text(json.dumps(pairs[:10],indent=2))
        report[cat]={"target_feature":tgt,"controls":ctr,"pairs":len(pairs),"sample_pairs":str(sample_path),"summary":str(summary),"controls":str(agg)}

    (OUT_DIR/"gemma_multi_category_200pairs_report.json").write_text(json.dumps(report,indent=2))
    print('Wrote improved multi-category 200-pair reports')


if __name__=='__main__':
    main()
