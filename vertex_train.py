#!/usr/bin/env python3
"""Vertex AI training script — runs ortho + Prism, saves results to GCS."""
import sys, os, json, gc, time, warnings, subprocess
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Install deps
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'datasets', 'scipy'])

import torch
import numpy as np

# Clone repo
if not os.path.exists('/tmp/prism'):
    subprocess.run(['git', 'clone', 'https://github.com/realityinspector/prismic-pretraining.git', '/tmp/prism'])
sys.path.insert(0, '/tmp/prism')
os.chdir('/tmp/prism')

from prism.config import TrainConfig, install_signal_handlers
from prism.train import train, _clear_memory
from prism.baselines import make_init_fn
from prism.pretrained_extract import extract_per_layer, make_hybrid_init_fn

install_signal_handlers()
device = 'cuda'
os.makedirs('/tmp/results', exist_ok=True)

gpu_name = torch.cuda.get_device_name(0)
print(f'GPU: {gpu_name}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')

# Pre-cache models
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
GPT2TokenizerFast.from_pretrained('gpt2')
GPT2LMHeadModel.from_pretrained('gpt2')
print('Models cached')

STEPS = 2000
EVALS = [250, 500, 750, 1000, 1500, 2000]

base_config = dict(
    max_steps=STEPS,
    eval_steps=EVALS,
    warmup_steps=200,
    log_every=50,
    seed=42,
    device=device,
    batch_size=4,
    grad_accum_steps=16,
    max_length=1024,
    memory_pressure_threshold=5,
)

def save_result(name, result, config_lr):
    """Save result to disk immediately."""
    data = {
        'name': name,
        'final_ppl': result['final_ppl'],
        'elapsed': result['elapsed'],
        'checkpoints': {str(k): v for k, v in result['checkpoints'].items()},
        'gpu': gpu_name,
        'steps': STEPS,
        'lr': config_lr,
        'batch': 64,
        'seq_len': 1024,
    }
    path = f'/tmp/results/{name}.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved {path}')
    # Also upload to GCS immediately
    gcs_path = os.environ.get('AIP_MODEL_DIR', 'gs://prism-training-results')
    subprocess.run(['gsutil', 'cp', path, f'{gcs_path}/{name}.json'], capture_output=True)
    print(f'Uploaded to GCS: {gcs_path}/{name}.json')

# === RUN 1: Orthogonal baseline ===
print('\n' + '='*60)
print('  RUN 1: Orthogonal baseline')
print('='*60)
t0 = time.time()
result_ortho = train(TrainConfig(**base_config, lr=6.25e-5),
                     init_fn=make_init_fn('orthogonal'),
                     init_name='ortho', verbose=True)
save_result('ortho', result_ortho, 6.25e-5)
print(f'Ortho done in {time.time()-t0:.0f}s — PPL: {result_ortho["final_ppl"]:.1f}')
_clear_memory(device); gc.collect()

# === RUN 2: Prism ===
print('\n' + '='*60)
print('  RUN 2: Prism (Spectral Imprint + EigenTransfer + 2x LR)')
print('='*60)
with open('prism/results/extracted_spectra.json') as f:
    extracted = json.load(f)
print('Extracting pretrained directions...')
dirs = extract_per_layer('gpt2', include_directions=True, device='cpu')
init_fn = make_hybrid_init_fn(
    extracted['spectra_coeffs'], dirs,
    lam=1.0, align_mode='UV', align_strength=0.5
)
gc.collect()

t0 = time.time()
result_prism = train(TrainConfig(**base_config, lr=1.25e-4),
                     init_fn=init_fn,
                     init_name='prism', verbose=True)
save_result('prism', result_prism, 1.25e-4)
print(f'Prism done in {time.time()-t0:.0f}s — PPL: {result_prism["final_ppl"]:.1f}')

# === COMPARISON ===
print('\n' + '='*60)
print('  CUDA VALIDATION RESULTS')
print('='*60)
print(f'GPU: {gpu_name}')
print(f'Config: GPT-2 small (124M), batch 64, seq_len 1024, {STEPS} steps')
print(f'\n{"Step":>6s}  {"Ortho":>8s}  {"Prism":>8s}  {"Ratio":>7s}')
print('-' * 35)
for step in EVALS:
    o = result_ortho['checkpoints'].get(step)
    p = result_prism['checkpoints'].get(step)
    if o and p:
        print(f'{step:>6d}  {o:>8.1f}  {p:>8.1f}  {o/p:>6.2f}x')

r = result_ortho['final_ppl'] / result_prism['final_ppl']
print(f'\nFinal: ortho={result_ortho["final_ppl"]:.1f}  prism={result_prism["final_ppl"]:.1f}  ratio={r:.2f}x')

# Save combined results
combined = {
    'gpu': gpu_name, 'steps': STEPS,
    'ortho': {str(k): v for k, v in result_ortho['checkpoints'].items()},
    'prism': {str(k): v for k, v in result_prism['checkpoints'].items()},
    'ortho_final': result_ortho['final_ppl'],
    'prism_final': result_prism['final_ppl'],
    'ratio': r,
}
with open('/tmp/results/combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
gcs = os.environ.get('AIP_MODEL_DIR', 'gs://prism-training-results')
subprocess.run(['gsutil', 'cp', '/tmp/results/combined.json', f'{gcs}/combined.json'])
print(f'\nAll results saved to GCS.')
