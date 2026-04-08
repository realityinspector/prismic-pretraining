# Prism CUDA Sweep — April 7-8, 2026

## Setup

- **GPU**: NVIDIA A100-SXM4-40GB (Google Colab Pro)
- **Model**: GPT-2 small (124M params)
- **Dataset**: WikiText-2
- **Batch**: 4 × 16 accumulation = effective batch 64
- **Sequence length**: 1024
- **Steps**: 750 per config
- **Eval checkpoints**: steps 100, 250, 500, 750
- **20 configs**: 8 grid (systematic LR/init/warmup) + 12 random (parameter entropy)

## Key Result

**Best config (rand_07): val_ppl 222 at step 750 — 2.74x over orthogonal baseline.**

Config: LR 1.5x (9.38e-5), UV alignment 0.75, warmup 300, spike_skip 50, grad_clip 1.0

No overfitting — val PPL still decreasing at step 750 (234 → 222 from step 500 to 750).

## Full Results (sorted by val_ppl@750)

| Rank | Config | Init | LR | Align | Clip | @250 | @500 | @750 | vs Ortho | Trend |
|------|--------|------|----|-------|------|------|------|------|----------|-------|
| 1 | rand_07 (ss) | UV | 1.5x | 0.75 | 1.0 | 414 | 234 | **222** | **2.74x** | improving |
| 2 | rand_04 | UV | 1.75x | 0.65 | 0.5 | 352 | 244 | **243** | **2.51x** | flat |
| 3 | prism_uv_2x | UV | 2.0x | 0.50 | 1.0 | 378 | 291 | 301 | 2.02x | slight overfit |
| 4 | prism_uv_1.5x | UV | 1.5x | 0.50 | 1.0 | 423 | 308 | 307 | 1.99x | flat |
| 5 | rand_06 | UV | 1.5x | 0.50 | 1.0 | 380 | 307 | 308 | 1.98x | flat |
| 6 | rand_08 | UV | 0.85x | 0.65 | 1.0 | 837 | 385 | 315 | 1.94x | improving |
| 7 | prism_uv_1x | UV | 1.0x | 0.50 | 1.0 | 509 | 349 | 338 | 1.81x | flat |
| 8 | prism_uv_1x_w500 | UV | 1.0x | 0.50 | 1.0 | 835 | 412 | 344 | 1.77x | improving |
| 9 | spectral_2x | spec | 2.0x | 0 | 1.0 | 485 | 363 | 380 | 1.60x | overfit |
| 10 | rand_05 | UV | 1.0x | 0.35 | 0.75 | 1045 | 461 | 389 | 1.57x | improving |
| 11 | rand_10 | UV | 1.15x | 0.25 | 1.0 | 717 | 406 | 393 | 1.55x | flat |
| 12 | rand_09 | spec | 1.5x | 0 | 0.5 | 545 | 392 | 397 | 1.53x | overfit |
| 13 | rand_11 | spec | 1.3x | 0 | 0.75 | 1046 | 467 | 417 | 1.46x | improving |
| 14 | rand_01 | spec | 1.15x | 0 | 1.0 | 640 | 434 | 428 | 1.42x | flat |
| 15 | rand_03 | spec | 1.15x | 0 | 1.0 | 563 | 436 | 433 | 1.41x | flat |
| 16 | prism_uv_0.5x | UV | 0.5x | 0.50 | 1.0 | 745 | 470 | 444 | 1.37x | improving |
| 17 | spectral_1x | spec | 1.0x | 0 | 1.0 | 689 | 460 | 448 | 1.36x | flat |
| 18 | rand_02 | UV | 0.85x | 0.10 | 0.75 | 1512 | 563 | 474 | 1.29x | improving |
| 19 | rand_00 | spec | 0.85x | 0 | 0.75 | 643 | 489 | 480 | 1.27x | flat |
| 20 | ortho_1x | ortho | 1.0x | 0 | 1.0 | 1515 | 682 | **610** | **1.00x** | improving |

## Patterns

### 1. UV alignment strength is the strongest lever

| Alignment | Best val_ppl@750 | vs Ortho |
|-----------|-----------------|----------|
| 0.75 | 222 | 2.74x |
| 0.65 | 243 | 2.51x |
| 0.50 | 301-308 | ~2.0x |
| 0.35 | 389 | 1.57x |
| 0.25 | 393 | 1.55x |
| 0.10 | 474 | 1.29x |
| 0 (spectral only) | 380-448 | 1.36-1.60x |

More alignment = better, but needs stabilization above 0.5.

### 2. Stability interventions unlock stronger alignment

Without stabilization, align > 0.5 + LR > 1x overfits. Two mechanisms work:
- **Spike-skip (threshold 50)**: rand_07 — best overall result
- **Tighter grad clip (0.5)**: rand_04 — second best

Either one prevents the overfitting that kills high-alignment configs.

### 3. Spectral shape alone is worth ~1.4x

Spectral-only configs (no UV alignment) cluster at 380-480 val_ppl, roughly
1.3-1.6x over orthogonal. UV alignment adds another 1.5-2x on top of that.

### 4. LR sweet spot is 1.5x with stabilization

| LR | Best stable result | Notes |
|----|-------------------|-------|
| 0.5x | 444 | Too conservative |
| 1.0x | 338 | Good, safe |
| 1.5x | 222 (with ss) | Best with stabilization |
| 1.75x | 243 (with clip) | Good with clip |
| 2.0x | 301 (slight overfit) | Borderline |

### 5. Warmup matters less than expected

Warmup 100-300 all work. Warmup 500 slows early convergence (hasn't
finished warming up by step 250) but catches up by step 750.

## Winning Config

```python
TrainConfig(
    lr=9.38e-5,           # 1.5x base LR
    warmup_steps=300,
    spike_skip_mult=50.0, # skip steps where gnorm > 50x running median
    grad_clip=1.0,        # standard clip (spike-skip handles outliers)
)

# Prism init with strong UV alignment
make_hybrid_init_fn(
    spectra_coeffs, dirs,
    lam=1.0,
    align_mode='UV',
    align_strength=0.75,  # stronger than default 0.5
)
```

## Open Questions

1. **Does 2.74x hold past 750 steps?** The winning config was still improving
   at step 750. Need a 2000-5000 step run to see if the advantage persists,
   shrinks gradually, or collapses.

2. **Is alignment 0.75 the ceiling?** We didn't test 0.85 or 0.9.
   The 1.0 alignment crashed on MPS (SVD convergence on embeddings)
   but might work with spike-skip on CUDA.

3. **Does this transfer to nanoGPT?** Our training loop uses HuggingFace.
   nanoGPT is the community standard for GPT-2 benchmarks.

4. **Does this scale to GPT-2 medium (355M)?** Self-extraction from
   medium might show different alignment dynamics.

## Cost

20 configs × ~45 min each = ~15 hours on A100. At Colab Pro rates,
roughly $15-20 total compute cost for the sweep.
