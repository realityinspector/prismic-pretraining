# Prism

**Prismic pretraining acceleration.**

3.33x faster convergence on GPT-2 training from scratch by transferring
the spectral fingerprint of a trained model's weight structure into a
fresh random initialization.

A prism decomposes white light into its spectral components. Prism
decomposes pretrained weights into their SVD spectrum and refracts that
structure into a new model — transferring the geometric fingerprint of
learned representations without transferring the weights themselves.

## Techniques

Prism is built on two spectral transfer techniques:

**Spectral Imprint** extracts the singular value distribution from
pretrained weights, compresses it to 8 DCT coefficients per group
(32 numbers total), and reshapes a fresh model's SVD spectrum to match.
This transfers *what the model learned matters* — which subspaces carry
the most energy — without transferring learned values.

**EigenTransfer** partially aligns the fresh model's singular vectors
(50% blend) with the pretrained model's directions. This transfers
*which directions in weight space are task-relevant*, giving the
optimizer a geometric head start beyond what spectral magnitudes alone
provide.

Together, these enable training at 2x the standard learning rate
(orthogonal init diverges at this LR) and produce 3.33x faster
convergence on GPT-2 small (124M params, WikiText-2).

```
W_init = U_fresh · diag(S_imprint) · blend(V_fresh, V_pretrained, 0.5)^T
```

## Results

| Config | PPL@750 | vs Orthogonal |
|--------|---------|---------------|
| Orthogonal baseline | 1,904 | 1.0x |
| Spectral Imprint only | 818 | 2.33x |
| + EigenTransfer + 2x LR | **572** | **3.33x** |

Discovered through 27 systematic experiments. See
[FINDINGS.md](FINDINGS.md) for the full results, mechanism analysis,
and literature grounding.

## Quick Start

```bash
python -m venv prism/.venv && source prism/.venv/bin/activate
pip install torch transformers datasets matplotlib numpy scipy

# Run the best config (Spectral Imprint + EigenTransfer + 2x LR)
python -m prism.stairclimb --run lr_2x_UV

# Run all 27 experiments (~14 hours on M1)
python -m prism.stairclimb --all

# View results
python -m prism.stairclimb --list
```

## Visualization

Open [web/index.html](web/index.html) for an interactive D3.js
visualization of the method and results, or see the
[live deployment](https://prismic-pretraining.up.railway.app).

## How It Works

1. **Extract**: SVD decompose each weight matrix in a pretrained GPT-2.
   Group by type (attention, FFN up, FFN down, embedding). Compress each
   group's average singular value spectrum to 8 DCT coefficients.

2. **Imprint**: For each weight matrix in the fresh model, SVD decompose
   it, replace the singular values with the extracted spectral shape
   (Spectral Imprint), and blend the right singular vectors 50% toward
   the pretrained directions (EigenTransfer). Reconstruct.

3. **Train**: The spectrally shaped landscape tolerates 2x the standard
   learning rate. Train with AdamW at LR 1.25e-4 (vs baseline 6.25e-5).

## Key Files

```
prism/
  spectral_init.py       Spectral Imprint: SVD extraction + DCT compression
  pretrained_extract.py  EigenTransfer: directional alignment
  baselines.py           Standard/Xavier/orthogonal/flat init
  train.py               Training loop
  stairclimb.py          Autoresearch harness (27 hypotheses)
  config.py              Training config
  results/               Experimental results (JSON)
```

## Related Work

Prism connects to several lines of research:

- **Spectral initialization theory** (Saxe et al. 2014): SVD structure at
  initialization determines learning dynamics in deep linear networks.
- **Frequency-domain priors** (Tancik et al. 2020): Fourier feature
  initialization for implicit neural representations. Prism generalizes
  this to extracting spectral priors from trained models.
- **SVD-based transfer** (PiSSA, NeurIPS 2024; DoRA, ICML 2024): Transfer
  SVD components for fine-tuning. Prism applies spectral transfer to
  from-scratch training.
- **Catapult mechanism** (Lewkowycz et al. 2020): Theoretical basis for why
  initialization affects maximum stable learning rate.

## License

Apache 2.0. See [LICENSE](LICENSE).
