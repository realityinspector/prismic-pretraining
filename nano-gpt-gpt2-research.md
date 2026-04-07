# nanoGPT + Prism: Research Brief

Research agent: return a concise table for each numbered item. We need
facts, not opinions. Cite sources. If uncertain, say so.

---

## 1. nanoGPT Architecture & Training Defaults

| Question | Answer |
|----------|--------|
| Repo URL | |
| Main training script filename | |
| Model config for GPT-2 (124M) | |
| Default optimizer + LR + schedule | |
| Default batch size + sequence length | |
| Default warmup steps | |
| Default total training steps/tokens | |
| Default gradient clipping value | |
| Weight initialization method used | |
| How are weights initialized (which function, which file, which line)? | |
| Does it use HuggingFace at all? | |
| Dataset used (default for GPT-2 reproduction) | |
| Reported final val loss for GPT-2 124M | |
| Reported wall time to reproduce GPT-2 124M (what hardware) | |

## 2. Weight Initialization Details

| Question | Answer |
|----------|--------|
| What init does nanoGPT use for attention Q/K/V weights? | |
| What init for attention output projection? | |
| What init for FFN up/down projections? | |
| What init for token embeddings? | |
| What init for position embeddings? | |
| Is there residual scaling (1/sqrt(2N))? If so, where? | |
| Are LayerNorm weights initialized to 1 and biases to 0? | |
| Is there any special init for the final lm_head projection? | |
| Exact code snippet for the init function | |

## 3. SVD Injection Points

| Question | Answer |
|----------|--------|
| Which weight matrices have shape (768, 768)? | |
| Which have shape (768, 3072) or (3072, 768)? | |
| Which have shape (50257, 768) or similar? | |
| Does nanoGPT tie embedding weights (wte = lm_head)? | |
| Are Q/K/V computed as one fused (768, 2304) matrix or separate? | |
| Does nanoGPT use bias in attention/FFN? | |
| Total count of 2D weight matrices eligible for spectral init | |

## 4. Training Loop Integration

| Question | Answer |
|----------|--------|
| Does nanoGPT support custom init functions or hooks? | |
| Where in the code would we insert spectral init (before which line)? | |
| Does it checkpoint during training? Format? | |
| Does it log validation loss? How often? | |
| Does it support wandb or other logging? | |
| Can we add eval checkpoints at specific step counts? | |
| Is the training loop in a single function or class? | |

## 5. Benchmark Comparisons

| Question | Answer |
|----------|--------|
| What val loss does standard nanoGPT achieve at 1K steps? | |
| At 5K steps? | |
| At 10K steps? | |
| At convergence (~600K steps)? | |
| What GPU was used for the canonical benchmark? | |
| How long per step on A100 (tokens/sec)? | |
| What batch size on A100? | |
| Are there known community forks with better results? Which? | |
| Is there a leaderboard or comparison table anywhere? | |

## 6. Prism Integration Plan

| Question | Answer |
|----------|--------|
| Minimum code changes needed to add spectral init to nanoGPT | |
| Can we extract spectra from HF GPT-2 and apply to nanoGPT model? | |
| Are the weight matrix names/shapes compatible between HF and nanoGPT? | |
| Does nanoGPT's GPT-2 have the same 50257 vocab size as HF? | |
| Does nanoGPT's position embedding length match HF (1024)? | |
| Any known forks that modify nanoGPT initialization? | |
| What's the fastest way to validate: modify train.py or wrap model init? | |

## 7. Risk Factors

| Question | Answer |
|----------|--------|
| Does nanoGPT use flash attention? Does that affect SVD init? | |
| Does nanoGPT use torch.compile? Does that affect custom init? | |
| Are there numerical precision differences (bf16, fp16, fp32)? | |
| Does the OpenWebText dataset require special download/preprocessing? | |
| How much disk space does the preprocessed dataset need? | |
| Any license restrictions on nanoGPT (MIT? Apache?)? | |
| Has anyone published spectral/SVD init results on nanoGPT before? | |

---

Return all tables filled in. For code-level questions, include file paths
and line numbers from the latest nanoGPT commit. If the repo has been
updated significantly since the original release, note which version you
are referencing.
