#!/usr/bin/env python3
"""
DGX Spark Fine-Tuning Script
=============================
Optimized for NVIDIA DGX Spark's 128GB unified memory architecture.

Supports:
  - Full fine-tuning  (small models <3B)
  - LoRA              (medium models 3-13B)
  - QLoRA             (large models 13B+, primary path for Gemma-4 31B)

Primary use-case in this repo: QLoRA fine-tuning of Gemma-4-31B-IT on
medical Q&A data stored as Parquet, using TRL SFTTrainer with sequence
packing for maximum training throughput on the GB10 Blackwell GPU.

Author: DGX Spark AI Development Series
License: MIT
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
try:
    # Text-only CausalLM class for Gemma-4.
    # AutoModelForCausalLM maps model_type="gemma4" → Gemma4ForConditionalGeneration
    # (the multimodal VLM).  That model requires mm_token_type_ids during training
    # and uses a custom loss path that disconnects gradients when TRL's padding_free
    # mode is active (i.e. packing + no flash_attn).
    # Gemma4ForCausalLM uses Gemma4TextModel: a plain causal decoder with standard
    # attention masks and a standard loss — no multimodal plumbing, no gradient issues.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM
except ImportError:
    _Gemma4ForCausalLM = None
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── GB10 Blackwell performance: enable TF32 + cuDNN benchmark ───────────────
# These are safe global settings; no GPU ops have run yet at import time.
# TF32: uses tensor cores for matmul/conv while keeping BF16 range → free speedup.
# cudnn.benchmark: lets cuDNN pick the fastest algorithm for each kernel shape.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# =============================================================================
# DGX SPARK CONFIGURATION
# =============================================================================

class DGXSparkConfig:
    """
    Hardware-optimised configuration for NVIDIA DGX Spark (GB10 Grace Blackwell).

    Key specs:
      - 128 GB LPDDR5X unified memory (CPU + GPU shared)
      - 900 GB/s NVLink-C2C bandwidth
      - ~250 TFLOPS BF16 / 1 PFLOPS FP4 AI performance
    """

    TOTAL_MEMORY_GB = 128
    RECOMMENDED_MODEL_MEMORY_GB = 80

    # Per-device batch sizes (single GB10 device)
    BATCH_SIZE_RECOMMENDATIONS = {
        "1B":  {"full": 16, "lora": 32, "qlora": 64},
        "3B":  {"full":  8, "lora": 16, "qlora": 32},
        "7B":  {"full":  4, "lora":  8, "qlora": 16},
        "13B": {"full":  2, "lora":  4, "qlora":  8},
        "30B": {"full":  1, "lora":  1, "qlora":  1},   # Gemma-4-31B: batch=1 (safe with torch.compile+grad-ckpt)
        "70B": {"full":  1, "lora":  1, "qlora":  2},
    }

    GRADIENT_ACCUMULATION = {
        "1B": 1, "3B": 2, "7B": 4, "13B": 8, "30B": 32, "70B": 16,
    }

    # Optimal learning rates per size & method
    LEARNING_RATES = {
        "1B":  {"full": 3e-4, "lora": 3e-4, "qlora": 3e-4},
        "3B":  {"full": 2e-4, "lora": 2e-4, "qlora": 2e-4},
        "7B":  {"full": 1e-4, "lora": 2e-4, "qlora": 2e-4},
        "13B": {"full": 1e-4, "lora": 1e-4, "qlora": 1e-4},
        "30B": {"full": 5e-5, "lora": 1e-4, "qlora": 1e-4},
        "70B": {"full": 5e-5, "lora": 5e-5, "qlora": 5e-5},
    }

    @classmethod
    def get_optimal_settings(cls, model_name: str, method: str) -> Dict[str, Any]:
        key = cls._get_size_key(model_name)
        return {
            "batch_size":           cls.BATCH_SIZE_RECOMMENDATIONS.get(key, {}).get(method, 4),
            "gradient_accumulation": cls.GRADIENT_ACCUMULATION.get(key, 4),
            "learning_rate":        cls.LEARNING_RATES.get(key, {}).get(method, 2e-4),
        }

    @staticmethod
    def _get_size_key(model_name: str) -> str:
        s = model_name.lower()
        if any(x in s for x in ["70b", "72b", "65b"]):
            return "70B"
        if any(x in s for x in ["30b", "31b", "32b", "34b"]):
            return "30B"
        if any(x in s for x in ["13b", "14b"]):
            return "13B"
        if any(x in s for x in ["7b", "8b"]):
            return "7B"
        if any(x in s for x in ["3b", "4b"]):
            return "3B"
        return "1B"

    @staticmethod
    def estimate_training_time(
        num_samples: int,
        avg_tokens_per_sample: int = 1200,
        num_epochs: int = 1,
        use_packing: bool = True,
    ) -> Dict[str, float]:
        """
        Rough wall-clock estimate for 31B QLoRA on DGX Spark GB10.

        Assumptions:
          - GB10 effective BF16 throughput: ~100 TFLOPS (40% of 250T peak)
          - 31B model: ~300 GFLOP / token during QLoRA training
          - Packing (SFTTrainer) eliminates padding waste: ~1.4× speedup
        """
        tokens_per_sec_base = 333          # without packing
        packing_speedup     = 1.4 if use_packing else 1.0
        tokens_per_sec      = tokens_per_sec_base * packing_speedup

        total_tokens  = num_samples * avg_tokens_per_sample * num_epochs
        total_seconds = total_tokens / tokens_per_sec
        hours         = total_seconds / 3600
        days          = hours / 24

        return {
            "total_tokens":  total_tokens,
            "tokens_per_sec": int(tokens_per_sec),
            "hours":         round(hours,  1),
            "days":          round(days,   1),
        }


# =============================================================================
# MODEL REGISTRY
# =============================================================================

SUPPORTED_MODELS = {
    # ── Gemma-4 (primary target for this repo) ───────────────────────────────
    # NOTE: fine-tuning requires the standard BF16 HuggingFace weights.
    # The nvidia/Gemma-4-31B-IT-NVFP4 model in your cache is inference-only.
    # HuggingFace will download google/gemma-4-31b-it (~62 GB) automatically
    # when you first run training (set HF_TOKEN in your environment).
    "gemma-4-31b": "google/gemma-4-31b-it",

    # ── Llama ────────────────────────────────────────────────────────────────
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",

    # ── Mistral ──────────────────────────────────────────────────────────────
    "mistral-7b":       "mistralai/Mistral-7B-v0.3",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",

    # ── Qwen 2.5 ─────────────────────────────────────────────────────────────
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b":   "Qwen/Qwen2.5-3B",
    "qwen2.5-7b":   "Qwen/Qwen2.5-7B",

    # ── Phi-3 ────────────────────────────────────────────────────────────────
    "phi-3-mini":  "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",

    # ── Gemma-2 ──────────────────────────────────────────────────────────────
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",

    # ── SmolLM ───────────────────────────────────────────────────────────────
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B",
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

class DatasetPreparator:
    """
    Prepare datasets for fine-tuning.

    Supports:
      - HuggingFace dataset names
      - Local .json / .jsonl files  (Alpaca-style)
      - Local .parquet files        (recommended for large datasets)
      - Local .csv files

    When the loaded tokenizer has a chat_template (e.g. Gemma-4, Llama-3),
    samples are formatted using apply_chat_template for proper turn markers.
    Falls back to the classic ### Instruction / ### Response format otherwise.
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Public: instruction / Q-A datasets
    # ------------------------------------------------------------------

    def prepare_instruction_dataset(
        self,
        dataset_path: str,
        split: str = "train",
        instruction_col: str = "instruction",
        input_col: str = "input",
        output_col: str = "output",
        question_col: Optional[str] = None,   # alias for instruction_col
        answer_col:   Optional[str] = None,   # alias for output_col
        num_samples: Optional[int] = None,
        tokenize: bool = True,
    ) -> Dataset:
        """
        Load and format an instruction / Q-A dataset.

        Args:
            dataset_path : HuggingFace name, or path to .json/.jsonl/.parquet/.csv
            question_col : column name for questions  (overrides instruction_col)
            answer_col   : column name for answers    (overrides output_col)
            tokenize     : If False, returns dataset with a 'text' column only.
                           Set False when using SFTTrainer (handles tokenisation
                           internally and adds sequence packing).
        """
        q_col = question_col or instruction_col
        a_col = answer_col   or output_col

        logger.info(f"Loading dataset: {dataset_path}")
        dataset = self._load_raw_dataset(dataset_path, split)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        logger.info(f"Dataset size: {len(dataset):,} samples")

        def format_sample(example):
            question = example.get(q_col) or example.get(instruction_col, "")
            answer   = example.get(a_col) or example.get(output_col, "")
            extra    = example.get(input_col, "")

            text = self._apply_template(question, answer, extra)
            return {"text": text}

        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

        if tokenize:
            return self._tokenize_dataset(dataset)
        return dataset   # raw 'text' column for SFTTrainer

    def prepare_conversational_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        conversations_col: str = "conversations",
        num_samples: Optional[int] = None,
    ) -> Dataset:
        """Prepare a conversational/ShareGPT-style dataset."""
        logger.info(f"Loading conversational dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        def format_conversation(example):
            parts = []
            for msg in example.get(conversations_col, []):
                role    = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                tag     = "### User" if role in ("user", "human") else "### Assistant"
                parts.append(f"{tag}:\n{content}")
            return {"text": "\n\n".join(parts)}

        dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        return self._tokenize_dataset(dataset)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw_dataset(self, dataset_path: str, split: str) -> Dataset:
        """Detect format and load dataset."""
        if dataset_path.endswith(".parquet"):
            return load_dataset("parquet", data_files=dataset_path, split="train")
        if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
            return load_dataset("json",    data_files=dataset_path, split="train")
        if dataset_path.endswith(".csv"):
            return load_dataset("csv",     data_files=dataset_path, split="train")
        # HuggingFace hub name
        return load_dataset(dataset_path, split=split)

    def _apply_template(self, question: str, answer: str, extra: str = "") -> str:
        """Format a Q/A pair using the tokenizer's chat template if available."""
        if getattr(self.tokenizer, "chat_template", None):
            user_content = f"{question}\n\n{extra}".strip() if extra else question
            # Gemma-4 and newer Gemma models use "model" role for the assistant.
            # Other models (Llama-3, Mistral, etc.) use "assistant".
            assistant_role = (
                "model"
                if "gemma" in (self.tokenizer.name_or_path or "").lower()
                else "assistant"
            )
            messages = [
                {"role": "user",          "content": user_content},
                {"role": assistant_role,  "content": answer},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        # Fallback: classic Alpaca-style markers
        if extra:
            return (
                f"### Instruction:\n{question}\n\n"
                f"### Input:\n{extra}\n\n"
                f"### Response:\n{answer}"
            )
        return f"### Instruction:\n{question}\n\n### Response:\n{answer}"

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        def tokenize_fn(examples):
            tok = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )
            tok["labels"] = tok["input_ids"].copy()
            return tok

        return dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenising",
        )


# =============================================================================
# FINE-TUNING STRATEGIES
# =============================================================================

class FineTuner:
    """
    Fine-tuning orchestrator for DGX Spark.

    Primary path   : QLoRA + SFTTrainer with packing  (recommended for ≥13B)
    Secondary path : LoRA / Full + standard Trainer   (smaller models)
    """

    def __init__(
        self,
        model_name: str,
        method: str = "qlora",
        output_dir: str = "./output",
        use_flash_attention: bool = True,
    ):
        self.model_name        = self._resolve_model_name(model_name)
        self.method            = method.lower()
        self.output_dir        = Path(output_dir)
        self.use_flash_attention = use_flash_attention

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.model     = None
        self.trainer   = None

        logger.info("Initialising FineTuner")
        logger.info(f"  Model  : {self.model_name}")
        logger.info(f"  Method : {self.method}")
        logger.info(f"  Output : {self.output_dir}")

    def _resolve_model_name(self, name: str) -> str:
        return SUPPORTED_MODELS.get(name.lower(), name)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model_and_tokenizer(self):
        logger.info("Loading tokeniser …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect Gemma-4 once so all _load_* methods can branch on self._is_gemma4.
        self._is_gemma4 = False
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            if getattr(_cfg, "model_type", "") == "gemma4":
                self._is_gemma4 = True
        except Exception:
            pass

        # Gemma-4 wraps its linear layers in Gemma4ClippableLinear, which is
        # not a subclass of torch.nn.Linear and is therefore incompatible with
        # bitsandbytes + PEFT adapter injection (QLoRA).
        # With 128 GB unified memory, BF16 LoRA fits comfortably (~86 GB), so
        # we automatically downgrade qlora → lora for Gemma-4.
        if self.method == "qlora" and self._is_gemma4:
            logger.warning(
                "Gemma-4 detected: Gemma4ClippableLinear is incompatible with "
                "bitsandbytes PEFT injection (QLoRA). "
                "Switching to BF16 LoRA — 128 GB unified memory is sufficient "
                "(model ~62 GB + LoRA + optimizer + activations ≈ 86 GB)."
            )
            self.method = "lora"

        logger.info(f"Loading model [{self.method}] …")
        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}

        # Always use PyTorch SDPA for Gemma-4 (and as default for all models).
        # SDPA dispatches to cuDNN's Blackwell-native FlashAttention kernel on
        # CUDA 13 / GB10 (SM 12.1) — same backend that nanochat uses via
        # F.scaled_dot_product_attention, which achieved 1,600 tok/sec on this GPU.
        # flash_attn wheel can't be imported (CUDA 13 ABI mismatch), but SDPA
        # does not require the flash_attn package — it's built into PyTorch.
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Attention: PyTorch SDPA (cuDNN Blackwell-native FlashAttention)")

        if self._is_gemma4:
            logger.info(
                "Gemma-4 model detected — will load Gemma4ForCausalLM (text-only class) "
                "instead of AutoModelForCausalLM which defaults to Gemma4ForConditionalGeneration "
                "(the VLM). The text-only class has a standard causal-LM forward pass with "
                "no multimodal branches, so gradients flow correctly through LoRA layers."
            )

        dispatch = {
            "full":  self._load_full_precision_model,
            "lora":  self._load_lora_model,
            "qlora": self._load_qlora_model,
        }
        if self.method not in dispatch:
            raise ValueError(f"Unknown method '{self.method}'. Choose: full, lora, qlora")

        dispatch[self.method](model_kwargs)
        self._log_memory_usage()

    def _model_cls(self):
        """Return the right from_pretrained class: Gemma4ForCausalLM or AutoModelForCausalLM."""
        if self._is_gemma4 and _Gemma4ForCausalLM is not None:
            return _Gemma4ForCausalLM
        return AutoModelForCausalLM

    def _load_full_precision_model(self, model_kwargs: Dict):
        model_kwargs["dtype"] = torch.bfloat16
        self.model = self._model_cls().from_pretrained(self.model_name, **model_kwargs)
        # gradient_checkpointing disabled at SFTConfig level; no need to enable here
        logger.info("Loaded model for full fine-tuning (BF16)")

    @staticmethod
    def _find_lora_target_modules(model) -> List[str]:
        """
        Auto-detect LoRA target modules at runtime.

        Gemma-4 wraps attention and MLP projections in Gemma4ClippableLinear,
        which PEFT cannot inject into directly.  This function detects that
        pattern and returns paths that target the inner nn.Linear instead
        (e.g. "q_proj.linear" instead of "q_proj").
        """
        standard = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

        clippable = set()
        for name, mod in model.named_modules():
            if type(mod).__name__ == "Gemma4ClippableLinear":
                leaf = name.rsplit(".", 1)[-1]
                if leaf in standard:
                    clippable.add(leaf)

        if clippable:
            targets = [f"{t}.linear" for t in standard if t in clippable]
            logger.info(
                f"Gemma4ClippableLinear detected on {sorted(clippable)}. "
                f"Targeting inner .linear sub-modules: {targets}"
            )
            return targets

        return standard

    def _load_lora_model(self, model_kwargs: Dict):
        model_kwargs["dtype"] = torch.bfloat16
        base = self._model_cls().from_pretrained(self.model_name, **model_kwargs)

        target_modules = self._find_lora_target_modules(base)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()
        logger.info("Loaded model with LoRA adapters")

    def _load_qlora_model(self, model_kwargs: Dict):
        """
        QLoRA for large models (13B+).

        Weights loaded in NF4 4-bit (bitsandbytes), compute in BF16.
        LoRA rank 32 is chosen to balance capacity and memory for a 31B model.
        Double quantisation reduces the quantisation constant memory footprint.
        """
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_cfg

        base = self._model_cls().from_pretrained(self.model_name, **model_kwargs)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)

        # r=32 for 31B: provides enough capacity for domain adaptation without
        # over-fitting or excessive memory. alpha=64 keeps effective scale = 2.
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()
        logger.info("Loaded quantised model with QLoRA adapters (NF4 + BF16 compute)")

    def _log_memory_usage(self):
        if torch.cuda.is_available():
            alloc   = torch.cuda.memory_allocated()  / 1e9
            reserved = torch.cuda.memory_reserved()  / 1e9
            logger.info(f"GPU memory — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    # ------------------------------------------------------------------
    # Training  —  SFTTrainer path (recommended: packing, chat template)
    # ------------------------------------------------------------------

    def train_sft(
        self,
        sft_dataset: Dataset,               # dataset with 'text' column, NOT tokenised
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 1,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        max_seq_length: int = 2048,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Fine-tune using TRL SFTTrainer with sequence packing.

        Packing concatenates multiple short examples into a single sequence
        up to max_seq_length, eliminating padding waste and improving GPU
        utilisation by 30-50% for datasets with variable-length samples.
        """
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError(
                "TRL is required for SFT training. "
                "Run: pip install 'trl>=0.12.0'"
            )

        settings = DGXSparkConfig.get_optimal_settings(self.model_name, self.method)
        batch_size                = batch_size                or settings["batch_size"]
        gradient_accumulation_steps = gradient_accumulation_steps or settings["gradient_accumulation"]
        learning_rate             = learning_rate             or settings["learning_rate"]

        effective_batch = batch_size * gradient_accumulation_steps
        logger.info("SFT Training configuration:")
        logger.info(f"  Epochs              : {num_epochs}")
        logger.info(f"  Per-device batch    : {batch_size}")
        logger.info(f"  Gradient accum      : {gradient_accumulation_steps}")
        logger.info(f"  Effective batch     : {effective_batch}")
        logger.info(f"  Learning rate       : {learning_rate}")
        logger.info(f"  Max sequence length : {max_seq_length}")
        logger.info(f"  Packing             : enabled")

        sft_cfg = SFTConfig(
            output_dir                  = str(self.output_dir),
            num_train_epochs            = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            learning_rate               = learning_rate,
            weight_decay                = 0.01,
            warmup_steps                = warmup_steps,
            lr_scheduler_type           = "cosine",
            logging_steps               = logging_steps,
            save_steps                  = save_steps,
            save_total_limit            = 3,
            eval_strategy               = "steps" if eval_dataset else "no",
            eval_steps                  = save_steps if eval_dataset else None,
            bf16                        = True,
            tf32                        = True,
            # gradient_checkpointing re-enabled: torch.compile is disabled for Gemma-4 LoRA
            # (SM 12.1 autograd incompatibility), so the compile+grad-ckpt recompile cascade
            # no longer applies. Without grad-ckpt all 60 layers' activations live simultaneously
            # (~25 GB peak) which combined with the 62 GB BF16 model caused hard system reboots.
            # With grad-ckpt: ~62 GB model + ~3 GB activations + ~2 GB LoRA/Adam ≈ 67 GB peak.
            gradient_checkpointing      = True,
            max_grad_norm               = max_grad_norm,
            report_to                   = ["tensorboard"],
            logging_dir                 = str(self.output_dir / "logs"),
            dataloader_num_workers      = 4,
            dataloader_pin_memory       = True,
            remove_unused_columns       = False,
            optim                       = "adamw_torch_fused",
            # SFT-specific ────────────────────────────────────────────
            max_length                  = max_seq_length,   # TRL 1.0: renamed from max_seq_length
            packing                     = True,
            dataset_text_field          = "text",
        )

        # No mm_token_type_ids patch needed: we load Gemma4ForCausalLM (text-only)
        # instead of Gemma4ForConditionalGeneration (VLM).  Gemma4ForCausalLM uses
        # Gemma4TextModel which has a standard causal-LM forward — no multimodal
        # token-type routing, no custom loss path, no gradient disconnection.
        #
        # Previous approach: monkey-patch forward() to inject mm_token_type_ids=zeros.
        # Problem: Gemma4ForConditionalGeneration's loss code has a non-standard path
        # that disconnects gradients when TRL padding_free mode is active (packing +
        # no flash_attn).  Root cause: TRL auto-enables padding_free=True for the
        # "bfd" packing strategy, which passes position_ids instead of attention_mask.
        # The VLM's loss computation then returns a constant tensor with no grad_fn.
        # Fix: use the text-only model class — same weights, clean forward pass.
        logger.info("Using Gemma4ForCausalLM (text-only) — no mm_token_type_ids patch needed")

        # LoRA adapters are already applied in _load_lora_model / _load_qlora_model
        # via get_peft_model(). Passing peft_config again to SFTTrainer would
        # double-apply LoRA (rejected by TRL 1.0+). Always pass None here.
        self.trainer = SFTTrainer(
            model              = self.model,
            processing_class   = self.tokenizer,   # TRL 1.0: renamed from tokenizer
            args               = sft_cfg,
            train_dataset      = sft_dataset,
            eval_dataset       = eval_dataset,
            peft_config        = None,
            callbacks          = [MemoryMonitorCallback(), DiagnosticCallback(self.model)],
        )

        logger.info("Starting SFT training …")
        result = self.trainer.train()
        self.save_model()

        metrics = result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Training  —  standard Trainer path (backward-compatible)
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        warmup_steps: int = 100,
        save_steps: int = 100,
        logging_steps: int = 10,
        max_grad_norm: float = 1.0,
    ):
        """Standard Trainer path. Accepts a pre-tokenised dataset."""
        settings = DGXSparkConfig.get_optimal_settings(self.model_name, self.method)
        batch_size                  = batch_size or settings["batch_size"]
        gradient_accumulation_steps = gradient_accumulation_steps or settings["gradient_accumulation"]

        logger.info(f"Training — epochs: {num_epochs}, batch: {batch_size}, "
                    f"accum: {gradient_accumulation_steps}, lr: {learning_rate}")

        training_args = TrainingArguments(
            output_dir                  = str(self.output_dir),
            num_train_epochs            = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            learning_rate               = learning_rate,
            weight_decay                = 0.01,
            warmup_steps                = warmup_steps,
            lr_scheduler_type           = "cosine",
            logging_steps               = logging_steps,
            save_steps                  = save_steps,
            save_total_limit            = 3,
            eval_strategy               = "steps" if eval_dataset else "no",
            eval_steps                  = save_steps if eval_dataset else None,
            bf16                        = True,
            tf32                        = True,
            gradient_checkpointing      = True,
            max_grad_norm               = max_grad_norm,
            report_to                   = ["tensorboard"],
            logging_dir                 = str(self.output_dir / "logs"),
            dataloader_num_workers      = 4,
            dataloader_pin_memory       = True,
            remove_unused_columns       = False,
            optim                       = "adamw_torch_fused",
        )

        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        self.trainer = Trainer(
            model         = self.model,
            args          = training_args,
            train_dataset = train_dataset,
            eval_dataset  = eval_dataset,
            data_collator = collator,
            callbacks     = [MemoryMonitorCallback()],
        )

        logger.info("Starting training …")
        result = self.trainer.train()
        self.save_model()

        metrics = result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None):
        save_path = Path(path) if path else self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path} …")

        if self.method in ("lora", "qlora"):
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            merged_path = save_path.parent / "merged_model"
            logger.info(f"Merging and saving full model to {merged_path} …")
            merged = self.model.merge_and_unload()
            merged.save_pretrained(merged_path, safe_serialization=True)
            self.tokenizer.save_pretrained(merged_path)
            logger.info(f"Merged BF16 model saved to {merged_path}")
            logger.info(
                "Next step: run quantize_to_nvfp4.py to convert the merged model "
                "to NVFP4 for maximum DGX Spark inference performance."
            )
        else:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        meta = {
            "model_name": self.model_name,
            "method":     self.method,
            "timestamp":  datetime.now().isoformat(),
        }
        with open(save_path / "training_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model saved successfully.")


class MemoryMonitorCallback(TrainerCallback):
    """Log GPU memory every 50 steps."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"Step {state.global_step}: GPU mem = {alloc:.2f} GB")


class DiagnosticCallback(TrainerCallback):
    """
    Enhanced training diagnostics — logged to TensorBoard every step.

    Metrics added:
      perplexity        — exp(loss); more readable than raw CE loss
      loss_delta        — step-over-step change; negative = converging
      loss_spike        — 1.0 if loss rose >15% vs previous step (instability flag)
      lora_B_norm_mean  — mean Frobenius norm of all LoRA B matrices
      lora_B_norm_max   — max  Frobenius norm of all LoRA B matrices
                          B is zero-initialised, so these start at 0 and grow
                          as the adapters learn. Staying near 0 = dead training.
      lora_grad_norm_max  — max  LoRA param gradient norm, captured pre-clip
      lora_grad_norm_mean — mean LoRA param gradient norm, captured pre-clip
                            Complements the overall grad_norm; helps pinpoint
                            which adapter layers are driving gradient explosions.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._prev_loss: Optional[float] = None
        self._lora_grad_max: float = 0.0
        self._lora_grad_mean: float = 0.0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Capture LoRA gradient norms before the optimizer step clips them."""
        norms = [
            p.grad.detach().float().norm().item()
            for n, p in self.model.named_parameters()
            if p.grad is not None and ("lora_A" in n or "lora_B" in n)
        ]
        if norms:
            self._lora_grad_max  = max(norms)
            self._lora_grad_mean = sum(norms) / len(norms)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import math

        loss = logs.get("loss")
        if loss is not None:
            # Perplexity — cap at exp(20) ≈ 485M to avoid inf on early chaotic steps
            logs["perplexity"] = round(math.exp(min(loss, 20.0)), 2)

            # Step-over-step delta and spike flag
            if self._prev_loss is not None:
                logs["loss_delta"] = round(loss - self._prev_loss, 6)
                logs["loss_spike"] = 1.0 if loss > self._prev_loss * 1.15 else 0.0
            self._prev_loss = loss

        # LoRA B matrix norms — zero at init, grow as adapters learn
        # Using only B (not A) because B starts at 0; cheap single norm per param.
        try:
            b_norms = [
                p.detach().float().norm().item()
                for n, p in self.model.named_parameters()
                if "lora_B" in n and p.requires_grad
            ]
            if b_norms:
                logs["lora_B_norm_mean"] = round(sum(b_norms) / len(b_norms), 6)
                logs["lora_B_norm_max"]  = round(max(b_norms), 6)
        except Exception:
            pass

        # Pre-clip LoRA gradient norms (captured in on_pre_optimizer_step)
        if self._lora_grad_max > 0:
            logs["lora_grad_norm_max"]  = round(self._lora_grad_max, 4)
            logs["lora_grad_norm_mean"] = round(self._lora_grad_mean, 6)


# =============================================================================
# INFERENCE
# =============================================================================

class InferenceEngine:
    """Run inference with a fine-tuned (or base) model."""

    def __init__(self, model_path: str, use_flash_attention: bool = True):
        logger.info(f"Loading model from {model_path} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs = {
            "dtype":          torch.bfloat16,
            "device_map":     "auto",
            "trust_remote_code": True,
        }
        if use_flash_attention:
            kwargs["attn_implementation"] = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        logger.info("Model loaded for inference.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens  = max_new_tokens,
                temperature     = temperature,
                top_p           = top_p,
                do_sample       = do_sample,
                pad_token_id    = self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

    def chat(self, question: str, context: str = "") -> str:
        if getattr(self.tokenizer, "chat_template", None):
            content = f"{question}\n\n{context}".strip() if context else question
            assistant_role = (
                "model" if "gemma" in (self.tokenizer.name_or_path or "").lower()
                else "assistant"
            )
            messages = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif context:
            prompt = f"### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        return self.generate(prompt)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models on NVIDIA DGX Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Medical fine-tuning example (Gemma-4-31B, QLoRA, SFTTrainer):
  export HF_TOKEN=<your_token>
  python finetune_dgx_spark.py \\
      --model gemma-4-31b \\
      --method qlora \\
      --dataset /home/ubuntu/medAI/medical_train.parquet \\
      --question-col question \\
      --answer-col answer \\
      --epochs 1 \\
      --output-dir output/gemma4_medical

Classic LoRA example:
  python finetune_dgx_spark.py --model qwen2.5-3b --method lora --dataset data.json

Inference:
  python finetune_dgx_spark.py \\
      --inference \\
      --model-path output/gemma4_medical/merged_model \\
      --prompt "What are the symptoms of type-2 diabetes?"
        """,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--model",  required=True,
                        help="Model shorthand (e.g. gemma-4-31b) or HuggingFace path")
    parser.add_argument("--method", default="qlora",
                        choices=["full", "lora", "qlora"],
                        help="Fine-tuning method (default: qlora)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument("--dataset",       required=True,
                        help="Path to .parquet / .json / .jsonl / .csv, or HF dataset name")
    parser.add_argument("--question-col",  default=None,
                        help="Column name for questions (default: 'instruction')")
    parser.add_argument("--answer-col",    default=None,
                        help="Column name for answers   (default: 'output')")
    parser.add_argument("--dataset-format", default="instruction",
                        choices=["instruction", "conversation"],
                        help="Dataset schema (default: instruction)")
    parser.add_argument("--max-samples",   type=int, default=None,
                        help="Cap the number of training samples")
    parser.add_argument("--max-length",    type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=1,
                        help="Training epochs (default: 1 — sufficient for 2.2M samples)")
    parser.add_argument("--batch-size",    type=int,   default=None,
                        help="Per-device batch size (auto if omitted)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (auto-selected per model size if omitted)")
    parser.add_argument("--output-dir",    default="./output",
                        help="Output directory")
    parser.add_argument("--no-sft-trainer", action="store_true",
                        help="Use standard Trainer instead of SFTTrainer (disables packing)")
    parser.add_argument("--no-flash-attention", action="store_true",
                        help="Disable Flash Attention 2")

    # ── Inference ────────────────────────────────────────────────────────────
    parser.add_argument("--inference",  action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--model-path", type=str,
                        help="Path to fine-tuned model for inference")
    parser.add_argument("--prompt",     type=str,
                        help="Prompt for inference mode")

    args = parser.parse_args()

    # ── Inference mode ───────────────────────────────────────────────────────
    if args.inference:
        if not args.model_path:
            parser.error("--model-path required for --inference")
        engine = InferenceEngine(args.model_path,
                                 use_flash_attention=not args.no_flash_attention)
        if args.prompt:
            print(f"\nResponse:\n{engine.chat(args.prompt)}")
        else:
            print("\nInteractive mode (type 'quit' to exit)")
            while True:
                q = input("\nYou: ").strip()
                if q.lower() == "quit":
                    break
                print(f"\nAssistant: {engine.chat(q)}")
        return

    # ── Training mode ────────────────────────────────────────────────────────
    print("=" * 60)
    print("DGX Spark Fine-Tuning")
    print("=" * 60)

    use_sft = not args.no_sft_trainer

    finetuner = FineTuner(
        model_name        = args.model,
        method            = args.method,
        output_dir        = args.output_dir,
        use_flash_attention = not args.no_flash_attention,
    )
    finetuner.load_model_and_tokenizer()

    preparator = DatasetPreparator(finetuner.tokenizer, max_length=args.max_length)

    if args.dataset_format == "instruction":
        dataset = preparator.prepare_instruction_dataset(
            args.dataset,
            question_col = args.question_col,
            answer_col   = args.answer_col,
            num_samples  = args.max_samples,
            tokenize     = not use_sft,   # SFTTrainer does its own tokenisation
        )
    elif args.dataset_format == "conversation":
        dataset = preparator.prepare_conversational_dataset(
            args.dataset, num_samples=args.max_samples
        )
    else:
        raise ValueError(f"Unknown dataset-format: {args.dataset_format}")

    # Print training time estimate for large jobs
    if len(dataset) > 10_000:
        est = DGXSparkConfig.estimate_training_time(
            num_samples           = len(dataset),
            avg_tokens_per_sample = 1200,
            num_epochs            = args.epochs,
            use_packing           = use_sft,
        )
        print(f"\nTraining time estimate (DGX Spark GB10):")
        print(f"  Samples          : {len(dataset):,}")
        print(f"  Epochs           : {args.epochs}")
        print(f"  Total tokens     : {est['total_tokens']:,}")
        print(f"  Throughput est.  : {est['tokens_per_sec']:,} tokens/sec")
        print(f"  Estimated time   : {est['hours']:.1f} h  (~{est['days']:.1f} days)")
        print()

    if use_sft:
        metrics = finetuner.train_sft(
            sft_dataset    = dataset,
            num_epochs     = args.epochs,
            learning_rate  = args.learning_rate,
            batch_size     = args.batch_size,
            max_seq_length = args.max_length,
        )
    else:
        metrics = finetuner.train(
            train_dataset  = dataset,
            num_epochs     = args.epochs,
            learning_rate  = args.learning_rate or 2e-4,
            batch_size     = args.batch_size,
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to  : {args.output_dir}")
    print(f"Training loss   : {metrics.get('train_loss', 'N/A')}")
    print()
    print("Next step — convert merged model to NVFP4 for DGX Spark deployment:")
    print(f"  python quantize_to_nvfp4.py \\")
    print(f"      --model-path {args.output_dir}/final_model/merged_model \\")
    print(f"      --output-path {args.output_dir}/gemma4_medical_nvfp4 \\")
    print(f"      --calibration-data {args.dataset}")


if __name__ == "__main__":
    main()
