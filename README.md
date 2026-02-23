# SAE Interpretability on a Small Open-Weight Transformer

A lightweight, local-only, reproducible experiment for feature probing with Sparse Autoencoders (SAEs) on Apple Silicon.

## What this repo does

1. Collects activations from one chosen model layer/stream.
2. Trains an SAE (`d_model -> d_sae -> d_model`) with ReLU + L1 sparsity.
3. Evaluates interpretability + generalization across two contrasting datasets.
4. Produces blog-ready tables/figures and feature summaries.

## Defaults

- Model: `EleutherAI/pythia-70m-deduped`
- Layer: `3`
- Stream: `mlp_output`
- Datasets:
  - A: `wikitext` (`wikitext-2-raw-v1`)
  - B: `codeparrot/github-code-clean`
- Device: MPS if available, CPU fallback

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## One-command run

```bash
./scripts/run_all.sh configs/default.yaml
```

Outputs:

- `outputs/results.json`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`
- `outputs/features/*.json` and `*.md`

## Stage-by-stage CLI

```bash
python -m src.collect_acts --config configs/default.yaml
python -m src.train_sae --config configs/default.yaml
python -m src.interpret --config configs/default.yaml --label A
python -m src.interpret --config configs/default.yaml --label B
python -m src.eval --config configs/default.yaml
python -m src.viz --config configs/default.yaml
```

## Reproducibility notes

- Pinned dependencies in `requirements.txt`
- Fixed seed via config (`seed`)
- Cached HF artifacts in `artifacts/hf_cache`
- Activation artifacts are deterministic files (`.mmap` + metadata)

## Key metrics reported

- Reconstruction: MSE, R²
- Sparsity: average L0 per token, average L1 magnitude, histograms
- Generalization: train-on-A eval-on-B and train-on-B eval-on-A degradation
- Optional proxy: feature selectivity (`A - B` mean activation)

## Blog Post Outline

### 1) Research question + hypothesis
- **Question:** Can SAEs trained on a single transformer layer discover sparse features that are interpretable and transfer across domains?
- **Hypothesis:** Some features are domain-specific (wiki vs code), while others are shared and robust across datasets.

### 2) Method
- **Model:** Pythia-70M (decoder-only transformer)
- **Activation target:** Layer 3, MLP output stream
- **Data:** Small sampled text corpora from general prose + code
- **SAE:** Linear encoder/decoder, ReLU hidden, MSE + L1 objective

### 3) Evaluation
- In-domain reconstruction/sparsity quality
- Cross-domain generalization degradation
- Feature-level interpretation by top activating contexts and heuristic labels

### 4) Key results + failure modes
- Report strongest/weakest transfer paths
- Show features with clear lexical patterns vs noisy mixed features
- Failure modes: dead features, over-sparsification, low-variance reconstruction bias

### 5) Limitations + next steps
- Single layer/stream and small model only
- No causal interventions yet
- Next: top-k SAE, multi-layer comparisons, causal feature patching, richer datasets

## Performance guidance for M1

- Keep defaults modest (e.g., 80k tokens per dataset)
- Use smaller `d_sae` and fewer epochs for faster iteration
- If memory pressure appears, reduce `seq_len`, `batch_size`, or token targets
