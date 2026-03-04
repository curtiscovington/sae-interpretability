# Experiment Plan: Gemma + Pretrained SAE Steering

Date: 2026-03-03  
Status: Planned (not yet executed)

## Objective

Build on prior Pythia-70M SAE intervention work by moving to a larger model family (Gemma) with a pretrained SAE, then test whether selected SAE features provide causal, controllable behavior steering.

## Primary Hypothesis

For a subset of semantically coherent features from a pretrained Gemma SAE:

1. Feature activation is stable on matched prompts.
2. Residualized interventions on that feature produce directional output shifts.
3. The steering effect remains after matched controls and does not catastrophically degrade response quality.

## Model + SAE Scope (v1)

- **Base model target:** `google/gemma-2-2b`
- **SAE source:** pretrained release compatible with Gemma residual stream activations (single layer in v1)
- **Intervention stream:** residualized SAE delta intervention at one chosen layer

Notes:
- v1 intentionally uses a single-layer setup to reduce confounds and accelerate iteration.
- If pretrained SAE compatibility constraints require different stream/layer naming, that mapping will be documented before run.

## Causal Intervention Definition

For activation tensor `x` at the target hook point:

- `h0 = encode(x)`
- `h1 = h0`, except `h1[f] = alpha * h0[f]` for feature `f`
- `x' = x + strength * (decode(h1) - decode(h0))`

This preserves a true no-op at `alpha=1.0` and avoids decode-replacement confounds.

## Feature Selection Protocol

Start with a broad candidate list (20-40 features), then down-select to 3-5 test features.

Selection criteria:

- high activation on coherent prompt subsets
- low obvious polysemantic contamination in top contexts
- concept plausibility for measurable steering

Target categories for initial pass:

- refusal/safety framing
- hedging/uncertainty language
- stepwise/explanatory style
- politeness/assistant tone
- code-formatting tendency

## Prompt/Eval Design

### Prompt sets

- **Discovery set:** broad prompts for candidate curation
- **Steering test set:** minimal pairs + neutral prompts
- **Generalization set:** out-of-template prompts for robustness

### Conditions per feature

- baseline (`alpha=1.0`)
- steer down (`alpha<1`)
- steer up (`alpha>1`)
- random feature control (matched norm)
- wrong-layer or mismatched-hook control (where feasible)

### Alpha sweep

`alpha in {0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0}` (can trim if unstable)

## Metrics

Per feature and condition:

- target behavior score (rubric/classifier/proxy specific to feature concept)
- contrast score on minimal pairs (`target_prompt - matched_control_prompt`)
- output quality proxy (fluency/coherence judge or perplexity proxy)
- task retention rate (did answer remain on-task)
- side-effect drift checks (irrelevant refusal/toxicity style shifts)

Report summary:

- mean effect size vs baseline
- bootstrap 95% CI
- monotonicity trend across alpha
- control-vs-treatment deltas

## Success Criteria (Pre-Registered)

A feature is considered a steering success if:

1. Directional effect appears in expected sign across alpha sweep,
2. Effect exceeds random-feature control with non-overlapping 95% CI (or equivalent significance),
3. Task retention remains above a pre-set threshold (target 90%+),
4. Side-effect drift remains below a pre-set threshold.

## Risks / Failure Modes

- pretrained SAE mismatch with exact hook location
- feature polysemanticity causing noisy or contradictory outputs
- large quality degradation at strong alpha
- effects that appear only on narrow templates (non-transferable)

Mitigations:

- validate hook alignment on a small dry run first
- add stronger controls and OOD prompt checks
- cap alpha for unstable features
- promote only replicated effects to blog claims

## Deliverables

1. Config(s) for Gemma run (single-layer v1)
2. Feature shortlist and rationale table
3. Steering result tables + plots (delta vs alpha)
4. Reproducible command log
5. Working blog draft in site repo `_drafts/`

## Run Plan (Execution Order)

1. Confirm Gemma + SAE compatibility and hook mapping.
2. Run discovery pass and rank candidate features.
3. Build minimal-pair probes for top candidates.
4. Run steering sweeps + controls.
5. Aggregate bootstrap summaries.
6. Draft post with calibrated claims and limitations.
