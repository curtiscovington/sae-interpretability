# Gemma SAE hardening sprint report

## Summary table (holdout, alpha=1.5)

|Category|Target Δ|Controls Δ|Wrong-layer weaker|Wrong-hook weaker|Transfer|Quality|Seed stability|Ablation specificity|
|---|---:|---:|---|---|---|---|---|---|
|sports|-0.0332|-0.0025|FAIL|PASS|FAIL|PASS|FAIL|FAIL|
|code|0.0288|-0.0446|PASS|PASS|PASS|PASS|FAIL|PASS|
|safety|0.0121|0.0176|PASS|PASS|PASS|PASS|PASS|FAIL|
|uncertainty|0.0065|0.0011|PASS|PASS|PASS|PASS|PASS|FAIL|

## Notes
- Controls are matched on activation mean + variance + selectivity profile.
- Holdout uses unseen context templates only.
- Quality metrics: coherence proxy (distinct-token ratio), entropy proxy (next-token entropy), task-retention proxy (prompt keyword carryover).