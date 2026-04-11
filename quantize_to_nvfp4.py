#!/usr/bin/env python3
"""
NVFP4 Quantisation Script
===========================
Converts a merged BF16 fine-tuned model to NVIDIA FP4 (NVFP4) format using
NVIDIA ModelOpt.  The output can be served with vLLM on DGX Spark exactly
like the original nvidia/Gemma-4-31B-IT-NVFP4 model, delivering the full
1 PFLOPS FP4 inference performance of the GB10 Blackwell GPU.

Pipeline
--------
  1. [train]   QLoRA fine-tuning  →  LoRA adapters  (finetune_dgx_spark.py)
  2. [train]   merge_and_unload() →  merged BF16 model  (done automatically)
  3. [this]    ModelOpt quantise  →  NVFP4 model
  4. [deploy]  vLLM serves the NVFP4 model  (run_medical_finetune.sh --serve)

Installation
------------
  pip install "nvidia-modelopt[all]>=0.21.0"

Usage
-----
  python quantize_to_nvfp4.py \\
      --model-path  output/gemma4_medical/final_model/merged_model \\
      --output-path output/gemma4_medical_nvfp4 \\
      --calibration-data /home/ubuntu/medAI/medical_train.parquet \\
      --num-calibration-samples 512

Deploy with vLLM (same flags as the original model):
  docker run -d --name gemma4-medical \\
      --gpus all -p 8000:8000 --ipc=host \\
      -v /home/ubuntu/dgxsparkfinetune/output/gemma4_medical_nvfp4:/model \\
      -v /home/bkprity/.cache/vllm:/root/.cache/vllm \\
      vllm/vllm-openai:gemma4-cu130 \\
      --model /model \\
      --max-model-len 32768 \\
      --gpu-memory-utilization 0.90 \\
      --enable-auto-tool-choice \\
      --tool-call-parser gemma4 \\
      --reasoning-parser gemma4
"""

import argparse
import logging
import random
import torch
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data loader
# ---------------------------------------------------------------------------

def load_calibration_texts(data_path: str, num_samples: int, max_length: int = 512) -> List[str]:
    """
    Load a small calibration set from the training Parquet (or JSON).
    ModelOpt uses this to compute per-block FP4 scale factors.
    More samples → more accurate scales, but also more time.
    512 samples is a good default.
    """
    logger.info(f"Loading {num_samples} calibration samples from {data_path} …")
    from datasets import load_dataset

    if data_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=data_path, split="train")
    elif data_path.endswith((".json", ".jsonl")):
        ds = load_dataset("json",    data_files=data_path, split="train")
    else:
        ds = load_dataset(data_path, split="train")

    # Random sample
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    samples = ds.select(indices)

    texts = []
    for row in samples:
        q = row.get("question") or row.get("instruction", "")
        a = row.get("answer")   or row.get("output",      "")
        texts.append(f"{q}\n\n{a}"[:max_length * 4])   # rough char cap before tokenisation

    logger.info(f"Loaded {len(texts)} calibration texts.")
    return texts


# ---------------------------------------------------------------------------
# Main quantisation routine
# ---------------------------------------------------------------------------

def quantize_to_nvfp4(
    model_path: str,
    output_path: str,
    calibration_data_path: str,
    num_calibration_samples: int = 512,
    batch_size: int = 4,
):
    """
    Quantise a BF16 HuggingFace model to NVFP4 using NVIDIA ModelOpt.

    ModelOpt uses activation-aware, per-block scaling (similar to GPTQ/AWQ
    but in FP4) to minimise quantisation error.  The resulting model can be
    loaded by vLLM with '--quantization fp4'.
    """
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf
    except ImportError:
        raise ImportError(
            "nvidia-modelopt is required.\n"
            "Install: pip install 'nvidia-modelopt[all]>=0.21.0'"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader

    logger.info(f"Loading BF16 model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype  = torch.bfloat16,
        device_map   = "auto",
        trust_remote_code = True,
    )
    model.eval()
    logger.info("Model loaded.")

    # Build calibration DataLoader
    texts = load_calibration_texts(calibration_data_path, num_calibration_samples)
    encodings = tokenizer(
        texts,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 512,
        padding        = True,
    )
    cal_dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"]
    )
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size)

    def forward_loop(model):
        """Feed calibration batches through the model for scale estimation."""
        device = next(model.parameters()).device
        for i, (input_ids, attention_mask) in enumerate(cal_loader):
            model(
                input_ids      = input_ids.to(device),
                attention_mask = attention_mask.to(device),
            )
            if (i + 1) % 20 == 0:
                logger.info(f"  Calibration batch {i+1}/{len(cal_loader)}")

    # FP4 quantisation config
    # FP4_DEFAULT_CFG quantises linear weight layers to NVIDIA FP4 with
    # per-block (128-element) scale factors stored in FP8.
    logger.info("Quantising to NVFP4 …")
    quant_cfg = mtq.FP4_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop)
    logger.info("Quantisation complete.")

    # Export to HuggingFace-compatible NVFP4 safetensors
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting NVFP4 model to {output_path} …")
    export_hf(model, tokenizer, output_path)

    # Write a deployment note
    deploy_note = f"""
# Medical Gemma-4-31B  —  NVFP4 Quantised Model
# ================================================
# Base model  : google/gemma-4-31b-it
# Fine-tuned  : medical Q&A ({num_calibration_samples} calibration samples used)
# Format      : NVFP4 (NVIDIA FP4, vLLM-compatible)
# Quantised   : {Path(model_path).resolve()}
#
# Deploy with vLLM:
#   docker run -d --name gemma4-medical \\
#       --gpus all -p 8000:8000 --ipc=host \\
#       -v {output_path}:/model \\
#       -v /home/bkprity/.cache/vllm:/root/.cache/vllm \\
#       vllm/vllm-openai:gemma4-cu130 \\
#       --model /model \\
#       --max-model-len 32768 \\
#       --gpu-memory-utilization 0.90 \\
#       --enable-auto-tool-choice \\
#       --tool-call-parser gemma4 \\
#       --reasoning-parser gemma4
"""
    with open(out / "DEPLOY.md", "w") as f:
        f.write(deploy_note.strip())

    logger.info("=" * 60)
    logger.info("NVFP4 quantisation complete!")
    logger.info(f"  Output : {output_path}")
    logger.info("  Deploy : see DEPLOY.md in the output directory")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantise a fine-tuned BF16 model to NVFP4 using NVIDIA ModelOpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path",  required=True,
                        help="Path to merged BF16 model (output of fine-tuning)")
    parser.add_argument("--output-path", required=True,
                        help="Where to save the NVFP4 model")
    parser.add_argument("--calibration-data", required=True,
                        help="Parquet/JSON file for calibration (use training data)")
    parser.add_argument("--num-calibration-samples", type=int, default=512,
                        help="Number of calibration samples (default: 512)")
    parser.add_argument("--calibration-batch-size",  type=int, default=4,
                        help="Calibration batch size (default: 4)")
    args = parser.parse_args()

    quantize_to_nvfp4(
        model_path              = args.model_path,
        output_path             = args.output_path,
        calibration_data_path   = args.calibration_data,
        num_calibration_samples = args.num_calibration_samples,
        batch_size              = args.calibration_batch_size,
    )


if __name__ == "__main__":
    main()
