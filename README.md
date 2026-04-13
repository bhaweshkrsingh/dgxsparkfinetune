# Fine-Tuning Language Models on NVIDIA DGX Spark

Complete toolkit for fine-tuning open-source language models on the NVIDIA DGX Spark (GB10 Blackwell).
Primary use-case: LoRA fine-tuning of large models (7B–31B+) on domain-specific medical datasets to
produce specialist doctor LLMs (PediatricianGemma, OncologyGemma, etc.).

The trained models are served via vLLM and consumed by the multi-specialist platform at
`/home/ubuntu/docLlms` — a full-stack multi-agent system with MCP server, Gradio frontend, and
FastAPI backend designed for future Docker deployment.

## Strategy: Specialist Doctor LLMs

Rather than fine-tuning on 2.2M random medical records (~7 months), we fine-tune on 50k
high-quality records per medical specialty (~4 days each). Each run produces one specialist model.

| Model | Dataset | Rows | Status | Est. Finish |
|-------|---------|------|--------|-------------|
| PediatricianGemma | `pediatrics_50k.parquet` | 50,000 | **TRAINING** (step ~6/600) | ~Apr 17 2026 |
| OncologyGemma | to be extracted | 50,000 | Planned | — |
| CardiologyGemma | to be extracted | 50,000 | Planned | — |
| NeurologyGemma | to be extracted | 50,000 | Planned | — |
| ObGynGemma | to be extracted | 50,000 | Planned | — |

Dataset source: `/home/ubuntu/medAI/medical_train.parquet` (2,187,808 rows, cols: question/answer)
Specialty extraction script: keyword-density ranking per specialty — see session notes.

## Quick Start — PediatricianGemma (current run)

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=hf_...

# 2. Run the full pipeline (preprocess → train → quantize)
chmod +x run_medical_finetune.sh
./run_medical_finetune.sh

# Or skip stages already done:
SKIP_INSTALL=1 SKIP_PREPROCESS=1 ./run_medical_finetune.sh
```

## Contents

```
dgxsparkfinetune/
├── finetune_dgx_spark.py         # Main fine-tuning script (all models/specialties)
├── prepare_medical_dataset.py    # Merge 50 medical parquet shards → 1 file
├── quantize_to_nvfp4.py          # Post-training NVFP4 quantization
├── run_medical_finetune.sh       # End-to-end pipeline script
├── monitor_training.sh           # Background monitor → status.log every N seconds
├── requirements.txt
└── output/
    ├── pediatrician_gemma/       # ACTIVE — 50k pediatric Q&A (step ~6/600)
    ├── oncology_gemma/           # Planned
    └── cardiology_gemma/         # Planned

/home/ubuntu/medAI/
├── medical_train.parquet         # Full 2.2M-row dataset
└── pediatrics_50k.parquet        # Extracted 50k pediatric records (top keyword-density score)

/home/ubuntu/docLlms/             # Serving + multi-agent platform (see that repo)
```

## Supported Models

| Model | Parameters | Method | Notes |
|-------|-----------|--------|-------|
| Gemma-4-31B | 31B | LoRA (BF16) | Primary target. QLoRA auto-downgrades to LoRA — see gotchas. |
| Llama 3.1/3.2 | 1B–8B | Full / LoRA / QLoRA | |
| Mistral | 7B–12B | LoRA / QLoRA | |
| Qwen 2.5 | 1.5B–7B | Full / LoRA / QLoRA | |
| Phi-3 | 3.8B–8B | LoRA / QLoRA | |
| Gemma 2 | 2B–9B | LoRA / QLoRA | |

## Fine-Tuning Methods

- **Full** — updates all parameters. Only practical for <3B models on DGX Spark.
- **LoRA** — trains adapter layers only. Recommended for 3B–31B in BF16.
- **QLoRA** — 4-bit NF4 + LoRA. For models where BF16 doesn't fit. Note: Gemma-4 auto-downgrades to LoRA (see gotchas).

## Mini vs Full Training

Always do a mini run first to validate the pipeline end-to-end before committing to a full run.

| Run | Samples | Est. time (DGX Spark GB10) |
|-----|---------|---------------------------|
| mini-medical | 100,000 | ~10 days (234 hrs) |
| full-medical | 2,187,808 | ~213 days (5,110 hrs) |

> **Note on timing:** Original estimates assumed ~466 tok/s. Actual throughput with
> `gradient_checkpointing=True` is ~122 tok/s (3.8× slower — grad-ckpt recomputes activations
> on the backward pass across all 60 layers). Measured from the first two mini-medical steps
> at ~537 s/step (1,566 total steps). The full 2.2M run is ~7 months on a single GB10 —
> consider reducing `max_length` (2048→1024) or `gradient_accumulation_steps` (32→8) to
> trade batch size for throughput before committing to it.

```bash
# Mini run (100k samples — validation)
python finetune_dgx_spark.py \
    --model gemma-4-31b --method qlora \
    --dataset /home/ubuntu/medAI/medical_train.parquet \
    --question-col question --answer-col answer \
    --max-samples 100000 --output-dir output/gemma4_mini_medical

# Full run (2.2M samples)
python finetune_dgx_spark.py \
    --model gemma-4-31b --method qlora \
    --dataset /home/ubuntu/medAI/medical_train.parquet \
    --question-col question --answer-col answer \
    --output-dir output/gemma4_full_medical
```

## Training Observability — See Inside Every Step

> **Why this matters:**
> With Gemma-4-31B at ~610 s/step, `logging_steps=25` means your first loss reading arrives
> after **4+ hours** and the second after **8+ hours**. If the loss is flat or rising you've
> already wasted most of a day. We set `logging_steps=1` and added a `DiagnosticCallback` so
> you know within the **first two steps** whether training is healthy.

### Metrics logged every optimizer step

| Metric | What it tells you | Healthy signal |
|--------|-------------------|----------------|
| `train/loss` | Cross-entropy loss | Decreasing |
| `train/perplexity` | `exp(loss)` — intuitive scale | Decreasing; target 2–20 for SFT |
| `train/loss_delta` | Step-over-step change | Negative from step 2 onward |
| `train/loss_spike` | `1.0` if loss rose >15% vs prev step | Stays at 0 |
| `train/grad_norm` | Overall gradient norm (pre-clip) | High early, stabilises after warmup |
| `train/lora_B_norm_mean` | Mean Frobenius norm of all LoRA B matrices | Starts at **0.0** (B is zero-init), rises as adapters learn — **stuck at 0 = dead training** |
| `train/lora_B_norm_max` | Max Frobenius norm across LoRA B matrices | Same signal, highlights hottest layer |
| `train/lora_grad_norm_max` | Max LoRA param gradient norm (pre-clip) | Tells you *which* layer is driving gradient explosions |
| `train/lora_grad_norm_mean` | Mean LoRA param gradient norm (pre-clip) | Complements overall grad_norm |
| `train/learning_rate` | Current LR (warmup → cosine decay) | Ramps up to peak, then decays |
| `train/mean_token_accuracy` | Token prediction accuracy | Increases as model learns |
| `train/entropy` | Output distribution entropy | Should decrease as model becomes more confident |

### How to monitor

```bash
# Background monitor → status.log every 2 min (reads TensorBoard events, not stdout)
nohup bash monitor_training.sh \
    output/pediatrician_gemma/train.log \
    output/pediatrician_gemma/status.log 120 > /dev/null 2>&1 &

# Tail the status log (loss + step + RAM every 2 min)
tail -f output/pediatrician_gemma/status.log

# TensorBoard (full metrics dashboard)
tensorboard --logdir output/pediatrician_gemma/runs

# Raw training stdout
tail -f output/pediatrician_gemma/train.log
```

### Why loss doesn't appear in train.log

The Trainer is configured with `report_to=["tensorboard"]`. The `{'loss': ...}` console print
is overwritten by tqdm's `\r` carriage returns in the log file. Loss is always written to the
TensorBoard events file under `output/<run>/runs/`. The monitor script reads directly from
there — not from stdout — so `status.log` always shows the real loss.

### Reading the diagnostic signals

**Step 1–2 (within the first ~20 minutes):**
- `loss_delta` should be negative. If it's positive or zero at step 2, the LR or LoRA rank is wrong.
- `lora_B_norm_mean` should be moving away from 0. If it stays exactly 0 after warmup starts, gradients are not reaching the adapters.

**Steps 1–100 (warmup phase):**
- `lora_grad_norm_max` will be very high (thousands). This is normal — LoRA A is randomly initialised and gradients are large before the model finds its footing. They should drop sharply once LR peaks.
- `loss_spike=1.0` occasionally during warmup is acceptable. Sustained spikes after warmup indicate instability — reduce LR.

**After step 100 (post-warmup, cosine decay begins):**
- `loss` and `perplexity` should be on a clear downward trend.
- `lora_B_norm_mean` should be steadily growing.
- `lora_grad_norm_max` should have settled to <100.

**Do not declare training successful until loss is visibly decreasing across at least 3 consecutive logged steps post-warmup.**

## Inference

```bash
python finetune_dgx_spark.py \
    --inference \
    --model-path output/gemma4_mini_medical/final_model/merged_model \
    --prompt "What are the early signs of type-2 diabetes?"
```

---

## DGX Spark Lessons Learned

Hard-won compatibility notes for training on the GB10 Blackwell (SM 12.1, CUDA 13, 128 GB unified memory).

### Required environment variables

Every training session must set these before running:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export PYTORCH_ALLOC_CONF=expandable_segments:True   # NOT the old PYTORCH_CUDA_ALLOC_CONF
source /home/ubuntu/venv/bin/activate
```

### Python venv

Use `/home/ubuntu/venv` (Python 3.12, `torch 2.9.0+cu130`). This is the only venv with working
CUDA support — the nanochat `.venv` (Python 3.10) installs CPU-only PyTorch because the
`cu130` index has no aarch64 Python 3.10 wheel.

### PyTorch 2.9 + SM 12.1 (GB10)

PyTorch 2.9 officially supports SM 8.0–12.0. GB10 is SM 12.1 — technically out-of-spec.
PyTorch prints a warning but runs correctly. CUDAGraphs are the one exception (see below).

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```
This warning is **informational only** — training proceeds correctly.

### CUDAGraphs broken on SM 12.1 (post NVIDIA driver update)

`torch.compile(mode='max-autotune', dynamic=False)` enables CUDAGraph trees by default.
After a NVIDIA driver update (driver 580.142+), CUDAGraph replay corrupts the wte embedding
tensor, causing:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
```

**Fix** — add this line once per optimizer step, **before** the gradient accumulation loop:

```python
torch.compiler.cudagraph_mark_step_begin()
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    ...
```

This is applied in `nanochat/scripts/base_train.py`.

### Gemma-4-31B specific gotchas

**1. QLoRA auto-downgrades to LoRA**
`Gemma4ClippableLinear` wraps projections and is incompatible with bitsandbytes PEFT injection.
The script detects this and auto-switches to BF16 LoRA. With 128 GB unified memory this fits:
~62 GB model + ~3 GB activations (grad-ckpt) + ~2 GB LoRA/Adam ≈ **67 GB peak**.

**2. Use `Gemma4ForCausalLM`, not `AutoModelForCausalLM`**
`AutoModelForCausalLM` resolves to `Gemma4ForConditionalGeneration` (the VLM), which has a
custom loss path that disconnects gradients when TRL's `padding_free` mode is active.
The script explicitly imports and uses `Gemma4ForCausalLM` (text-only decoder).

**3. LoRA targets `.linear` sub-modules**
`Gemma4ClippableLinear` wraps inner `nn.Linear` at `.linear`. PEFT must target
`q_proj.linear`, `k_proj.linear`, etc. — not `q_proj` directly.
`_find_lora_target_modules()` auto-detects this at runtime.

**4. `gradient_checkpointing = True` is MANDATORY**
Without it, all 60 layers' activations live simultaneously during backprop (~25 GB peak).
Combined with the 62 GB BF16 model this caused **hard system reboots** (no error — just reboot).
`torch.compile` is already disabled for Gemma-4, so the old compile+grad-ckpt recompile
cascade no longer applies. Always keep `gradient_checkpointing=True`.

**5. SDPA attention (no flash_attn)**
`flash_attn` wheel cannot be installed on CUDA 13 / SM 12.1 (ABI mismatch).
Use `attn_implementation="sdpa"` — PyTorch's `F.scaled_dot_product_attention` dispatches
to cuDNN's Blackwell-native FlashAttention kernel automatically.

**6. `torch.compile` disabled for Gemma-4 LoRA**
Repeated recompile cascades on SM 12.1 with LoRA autograd. Do not re-enable.

### TRL SFTTrainer warnings (benign)

```
[RANK 0] Padding-free training is enabled, but the attention implementation is not set
to a supported flash attention variant...
```
These appear because `flash_attn` is unavailable. Training proceeds correctly.
Same trade-off nanochat makes with continuous token packing. Safe to ignore.

### Deprecated API: use `warmup_steps` not `warmup_ratio`

`warmup_ratio` is deprecated in transformers and will be removed in v5.2.

```python
# Old (deprecated — causes warning):
SFTConfig(warmup_ratio=0.03, ...)

# Correct:
SFTConfig(warmup_steps=100, ...)
```

`warmup_steps=100` is used in this repo (suitable for both mini and full medical runs).

### nvidia-smi memory quirk

`nvidia-smi --query-gpu=memory.total` returns `[N/A]` for the GB10 — expected, because
it uses unified (CPU+GPU) memory. Use `free -h` to see total system memory instead.

---

## Post-Training: One-Shot Pipeline

When training completes, run the single post-training script in `/home/ubuntu/docLlms`:

```bash
bash /home/ubuntu/docLlms/scripts/post_training.sh
```

This does everything in sequence automatically:
1. Verifies the merged BF16 model exists
2. Quantises to NVFP4 (nvidia-modelopt)
3. Updates `registry.yaml` status: `training` → `ready`
4. Launches vLLM via Docker (`vllm/vllm-openai:gemma4-cu130`)
5. Waits for vLLM health check
6. Restarts Gradio UI (now connects to live model)

Or run quantisation alone:

```bash
python quantize_to_nvfp4.py \
    --model-path  output/pediatrician_gemma/final_model/merged_model \
    --output-path output/pediatrician_gemma_nvfp4 \
    --calibration-data /home/ubuntu/medAI/pediatrics_50k.parquet \
    --num-calibration-samples 512
```

### NVFP4 quantization — version notes

Requires `nvidia-modelopt>=0.21.0` (0.42.0 installed). API changes from older versions:

| Older API | Current (0.42.0+) |
|-----------|-------------------|
| `mtq.FP4_DEFAULT_CFG` | `mtq.NVFP4_DEFAULT_CFG` |
| `from modelopt.torch.export import export_hf` | `from modelopt.torch.export import export_hf_checkpoint as export_hf` |

The script handles both automatically. Use `NVFP4_DEFAULT_CFG` for standard quantisation;
`NVFP4_AWQ_LITE_CFG` for better accuracy at slight speed cost.

### vLLM serving (Docker)

vLLM is served via Docker, not pip — the `vllm/vllm-openai:gemma4-cu130` image (21.9 GB,
already pulled) contains the correct CUDA 13 / SM 12.1 build. Do not `pip install vllm`
into the training venv.

```bash
# Serve PediatricianGemma (BF16 or NVFP4 — auto-detected)
bash /home/ubuntu/docLlms/scripts/serve_model.sh pediatrician

# Test
curl http://localhost:8101/v1/models
```

## License

MIT
