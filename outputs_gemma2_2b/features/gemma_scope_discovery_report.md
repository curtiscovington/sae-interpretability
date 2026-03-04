# Gemma Scope Discovery + Steering (Quick Pass)

- Model: `google/gemma-2-2b`
- SAE: `gemma-scope-2b-pt-res-canonical` / `layer_20/width_16k/canonical`
- Layer: `20`

## Selected Features (top differential by category)
- sports: feature `16242`
- code: feature `6631`
- safety: feature `14033`
- uncertainty: feature `4223`

## Steering Summary (Δcontrast vs baseline)

### sports
- alpha=0.00: Δcontrast=0.328125 (std=0.543581, n=8)
- alpha=0.50: Δcontrast=0.179688 (std=0.270246, n=8)
- alpha=1.00: Δcontrast=0.000000 (std=0.000000, n=8)
- alpha=1.50: Δcontrast=-0.265625 (std=0.394932, n=8)
- alpha=2.00: Δcontrast=-0.507812 (std=0.686637, n=8)

### code
- alpha=0.00: Δcontrast=-0.097656 (std=2.147689, n=8)
- alpha=0.50: Δcontrast=0.175781 (std=0.662689, n=8)
- alpha=1.00: Δcontrast=0.000000 (std=0.000000, n=8)
- alpha=1.50: Δcontrast=-0.156250 (std=0.668988, n=8)
- alpha=2.00: Δcontrast=-0.484375 (std=1.093048, n=8)

### safety
- alpha=0.00: Δcontrast=0.070312 (std=0.209798, n=8)
- alpha=0.50: Δcontrast=0.015625 (std=0.098821, n=8)
- alpha=1.00: Δcontrast=0.000000 (std=0.000000, n=8)
- alpha=1.50: Δcontrast=-0.117188 (std=0.178152, n=8)
- alpha=2.00: Δcontrast=-0.203125 (std=0.393516, n=8)

### uncertainty
- alpha=0.00: Δcontrast=-0.148438 (std=0.191731, n=8)
- alpha=0.50: Δcontrast=-0.023438 (std=0.160000, n=8)
- alpha=1.00: Δcontrast=0.000000 (std=0.000000, n=8)
- alpha=1.50: Δcontrast=0.062500 (std=0.074702, n=8)
- alpha=2.00: Δcontrast=0.085938 (std=0.145381, n=8)
