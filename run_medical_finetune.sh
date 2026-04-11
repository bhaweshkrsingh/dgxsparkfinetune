#!/bin/bash
# =============================================================================
# Medical Gemma-4-31B Fine-Tuning Pipeline
# =============================================================================
#
# Full pipeline:
#   1. Install dependencies
#   2. Preprocess training data  (50 parquets → 1 merged parquet)
#   3. QLoRA fine-tune Gemma-4-31B with SFTTrainer + packing
#   4. Quantise merged model to NVFP4 for DGX Spark deployment
#
# Usage:
#   chmod +x run_medical_finetune.sh
#
#   # Full pipeline (all 4 stages):
#   ./run_medical_finetune.sh
#
#   # Skip stages already done:
#   SKIP_INSTALL=1 SKIP_PREPROCESS=1 ./run_medical_finetune.sh
#
#   # Dry-run: print settings, skip training:
#   DRY_RUN=1 ./run_medical_finetune.sh
#
#   # After fine-tuning is complete, quantise to NVFP4:
#   ./run_medical_finetune.sh --quantize-only
#
# Required environment variable:
#   HF_TOKEN   — HuggingFace token for downloading google/gemma-4-31b-it
#                Export before running: export HF_TOKEN=hf_...
# =============================================================================

set -e

# ─── Configurable paths ───────────────────────────────────────────────────────
VENV_DIR="/home/ubuntu/venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_DIR="/home/ubuntu/medAI/trgData"
MERGED_PARQUET="/home/ubuntu/medAI/medical_train.parquet"
OUTPUT_BASE="${SCRIPT_DIR}/output"
FINETUNE_OUTPUT="${OUTPUT_BASE}/gemma4_medical"
NVFP4_OUTPUT="${OUTPUT_BASE}/gemma4_medical_nvfp4"

# ─── Training hyperparameters ─────────────────────────────────────────────────
MODEL="gemma-4-31b"
METHOD="qlora"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN=2048
# batch_size=2, gradient_accumulation=8 → effective batch = 16
# (auto-detected by DGXSparkConfig for the 30B tier)

# ─── Flags ────────────────────────────────────────────────────────────────────
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
DRY_RUN="${DRY_RUN:-0}"
QUANTIZE_ONLY="${1:-}"

# ─── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Medical Gemma-4-31B Fine-Tuning  —  DGX Spark GB10    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Validate HF token ────────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    error "HF_TOKEN is not set.\nExport your HuggingFace token: export HF_TOKEN=hf_..."
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # some tools read this alias
info "HF_TOKEN found."

# ─── Activate venv ────────────────────────────────────────────────────────────
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    info "Activated venv: ${VENV_DIR}"
else
    warn "Venv not found at ${VENV_DIR}. Using system Python."
fi

# =============================================================================
# STAGE 1 — Install dependencies
# =============================================================================
if [ "$SKIP_INSTALL" = "0" ]; then
    info "Installing / verifying dependencies …"
    cd "$SCRIPT_DIR"
    pip install -q --upgrade pip wheel
    pip install -q -r requirements.txt
    info "Dependencies OK."
else
    info "Skipping dependency install (SKIP_INSTALL=1)."
fi

# =============================================================================
# STAGE 2 — Preprocess: merge 50 parquets → 1 Parquet file
# =============================================================================
if [ "$QUANTIZE_ONLY" = "--quantize-only" ]; then
    info "Skipping preprocess and training (--quantize-only)."
elif [ "$SKIP_PREPROCESS" = "0" ] && [ ! -f "$MERGED_PARQUET" ]; then
    info "Preprocessing training data …"
    info "  Input  : ${TRAINING_DATA_DIR} (50 shards)"
    info "  Output : ${MERGED_PARQUET}"
    [ "$DRY_RUN" = "0" ] && python "${SCRIPT_DIR}/prepare_medical_dataset.py" \
        --input-dir "$TRAINING_DATA_DIR" \
        --output    "$MERGED_PARQUET" \
        --min-answer-chars 50
    info "Preprocessing complete."
elif [ -f "$MERGED_PARQUET" ]; then
    info "Merged parquet already exists, skipping preprocess. (delete to redo)"
else
    info "Skipping preprocess (SKIP_PREPROCESS=1)."
fi

# =============================================================================
# STAGE 3 — Fine-tuning
# =============================================================================
if [ "$QUANTIZE_ONLY" != "--quantize-only" ]; then

    info "Fine-tuning configuration:"
    info "  Model            : ${MODEL}  →  google/gemma-4-31b-it (BF16 download)"
    info "  Method           : ${METHOD} (bitsandbytes NF4 + LoRA r=32)"
    info "  Trainer          : SFTTrainer with sequence packing"
    info "  Dataset          : ${MERGED_PARQUET}"
    info "  Epochs           : ${EPOCHS}"
    info "  Max seq length   : ${MAX_SEQ_LEN}"
    info "  Batch size       : 2 per device  (auto-selected for 30B)"
    info "  Grad accumulation: 8             (effective batch = 16)"
    info "  Learning rate    : 1e-4          (auto-selected for 30B QLoRA)"
    info "  Output           : ${FINETUNE_OUTPUT}"
    echo ""
    info "Memory footprint estimate (DGX Spark 128 GB unified):"
    info "  4-bit model weights  : ~16 GB"
    info "  LoRA adapters + opt  : ~10 GB"
    info "  Activations (gc)     : ~20 GB"
    info "  Total                : ~46 GB  ✓ (fits in 128 GB)"
    echo ""
    info "Training time estimate (GB10, 1 epoch, 2.2M samples, packing):"
    info "  ~466 tokens/sec  →  ~14-21 days for full dataset"
    info "  Tip: set MAX_SAMPLES to train on a subset first."
    echo ""

    if [ "$DRY_RUN" = "1" ]; then
        warn "DRY_RUN=1 — skipping actual training."
    else
        mkdir -p "$FINETUNE_OUTPUT"

        python "${SCRIPT_DIR}/finetune_dgx_spark.py" \
            --model           "$MODEL"            \
            --method          "$METHOD"           \
            --dataset         "$MERGED_PARQUET"   \
            --question-col    question            \
            --answer-col      answer              \
            --epochs          "$EPOCHS"           \
            --max-length      "$MAX_SEQ_LEN"      \
            --output-dir      "$FINETUNE_OUTPUT"

        info "Fine-tuning complete."
        info "  LoRA adapters : ${FINETUNE_OUTPUT}/final_model/"
        info "  Merged BF16   : ${FINETUNE_OUTPUT}/final_model/merged_model/"
    fi
fi

# =============================================================================
# STAGE 4 — Quantise to NVFP4
# =============================================================================
MERGED_MODEL="${FINETUNE_OUTPUT}/final_model/merged_model"

if [ "$DRY_RUN" = "1" ]; then
    warn "DRY_RUN=1 — skipping NVFP4 quantisation."
elif [ ! -d "$MERGED_MODEL" ] && [ "$QUANTIZE_ONLY" != "--quantize-only" ]; then
    warn "Merged model not found at ${MERGED_MODEL}. Skipping NVFP4 step."
    warn "Re-run after training completes, or use --quantize-only once model is ready."
else
    info "Quantising merged model to NVFP4 …"
    info "  Input  : ${MERGED_MODEL}"
    info "  Output : ${NVFP4_OUTPUT}"
    info "  Calib  : ${MERGED_PARQUET} (512 samples)"

    # nvidia-modelopt must be installed
    if ! python -c "import modelopt" 2>/dev/null; then
        warn "nvidia-modelopt not found. Installing …"
        pip install -q "nvidia-modelopt[all]>=0.21.0"
    fi

    python "${SCRIPT_DIR}/quantize_to_nvfp4.py" \
        --model-path              "$MERGED_MODEL"   \
        --output-path             "$NVFP4_OUTPUT"   \
        --calibration-data        "$MERGED_PARQUET" \
        --num-calibration-samples 512

    info "NVFP4 model ready: ${NVFP4_OUTPUT}"
fi

# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                     Pipeline Summary                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Merged BF16 model  : ${FINETUNE_OUTPUT}/final_model/merged_model"
echo "  NVFP4 deploy model : ${NVFP4_OUTPUT}"
echo ""
echo "  Deploy with vLLM (copy from DEPLOY.md inside NVFP4 dir):"
echo "    docker run -d --name gemma4-medical \\"
echo "        --gpus all -p 8000:8000 --ipc=host \\"
echo "        -v ${NVFP4_OUTPUT}:/model \\"
echo "        vllm/vllm-openai:gemma4-cu130 \\"
echo "        --model /model --max-model-len 32768 \\"
echo "        --gpu-memory-utilization 0.90 \\"
echo "        --tool-call-parser gemma4 --reasoning-parser gemma4"
echo ""
echo "  Monitor training:"
echo "    tensorboard --logdir ${FINETUNE_OUTPUT}/logs"
echo ""
echo "  Test the fine-tuned model:"
echo "    python finetune_dgx_spark.py --inference \\"
echo "        --model-path ${FINETUNE_OUTPUT}/final_model/merged_model \\"
echo "        --prompt 'What are the early signs of type-2 diabetes?'"
echo ""
