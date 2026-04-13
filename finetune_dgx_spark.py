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

# ── Transformers 4.56+ tokenizer monkey-patch ────────────────────────────────
# Bug in transformers 4.56.x–4.57.x: _set_model_specific_special_tokens has
# signature `list[str]` (matching Gemma-4's tokenizer_config extra_special_tokens)
# but the body calls .keys()/.items(), which are dict methods.  This crashes when
# loading the Gemma-4-31B tokenizer whose config has:
#   "extra_special_tokens": ["<|video|>"]   ← list, not dict
# Fix: guard on dict; skip list inputs (the <|video|> video token is irrelevant
# for text-only medical fine-tuning and can safely be left unregistered).
def _patch_set_model_specific_special_tokens():
    try:
        from transformers import PreTrainedTokenizerBase
        from transformers.tokenization_utils_base import AddedToken
        original = getattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None)
        if original is None:
            return  # older transformers without this method — nothing to patch

        def _safe_set(self, special_tokens):
            if not isinstance(special_tokens, dict):
                return  # list input (e.g. ["<|video|>"]) — skip silently
            self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
            for key, value in special_tokens.items():
                if isinstance(value, (str, AddedToken)):
                    self._special_tokens_map[key] = value

        PreTrainedTokenizerBase._set_model_specific_special_tokens = _safe_set
    except Exception:
        pass  # never crash the training over a patch

_patch_set_model_specific_special_tokens()


# ── Gemma-4 chat-template {% generation %} patch ─────────────────────────────
# TRL 1.0 assistant_only_loss uses tokenizer.apply_chat_template(
#   ..., return_assistant_tokens_mask=True) which requires {% generation %} /
# {% endgeneration %} markers in the Jinja2 template to identify which tokens
# are the model response.  Gemma-4's shipped template (as of Apr 2026) lacks
# these markers.  We patch them in at runtime, right before the two places
# where the model-role content is rendered:
#   1. string content:  {{- strip_thinking(message['content']) -}}
#   2. sequence items:  {{- strip_thinking(item['text']) -}}
def _patch_gemma4_chat_template_for_generation_markers(tokenizer) -> None:
    """Add {% generation %} markers to Gemma-4 chat template if missing."""
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl is None or "{% generation %}" in tmpl:
        return  # already patched or not a template tokenizer

    # Pattern 1 — string message content for model role
    old1 = "{{- strip_thinking(message['content']) -}}"
    new1 = "{% generation %}{{- strip_thinking(message['content']) -}}{% endgeneration %}"

    # Pattern 2 — sequence item text content for model role
    old2 = "{{- strip_thinking(item['text']) -}}"
    new2 = "{% generation %}{{- strip_thinking(item['text']) -}}{% endgeneration %}"

    if old1 not in tmpl and old2 not in tmpl:
        logger.warning(
            "Gemma-4 chat template {% generation %} patch: expected patterns not found. "
            "Template may have changed upstream. assistant_only_loss may not work correctly."
        )
        return

    patched = tmpl.replace(old1, new1).replace(old2, new2)
    tokenizer.chat_template = patched
    logger.info("Patched Gemma-4 chat template with {% generation %} markers for assistant_only_loss.")

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
# PyTorch 2.9: allow_tf32 replaced by fp32_precision="tf32"; keep legacy fallback.
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"   # PyTorch 2.9+
except AttributeError:
    torch.backends.cuda.matmul.allow_tf32 = True          # PyTorch ≤ 2.8
torch.backends.cudnn.allow_tf32  = True
torch.backends.cudnn.benchmark   = True


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

        # For SFTTrainer (tokenize=False) produce a 'messages' column in
        # conversational format so TRL can apply the chat template itself and
        # correctly mask prompt tokens when assistant_only_loss=True.
        # For the standard Trainer (tokenize=True) keep the pre-formatted 'text'.
        assistant_role = (
            "model"
            if "gemma" in (self.tokenizer.name_or_path or "").lower()
            else "assistant"
        )

        if not tokenize:
            def format_sample(example):
                question = example.get(q_col) or example.get(instruction_col, "")
                answer   = example.get(a_col) or example.get(output_col, "")
                extra    = example.get(input_col, "")
                user_content = f"{question}\n\n{extra}".strip() if extra else question
                return {"messages": [
                    {"role": "user",          "content": user_content},
                    {"role": assistant_role,  "content": answer},
                ]}
        else:
            def format_sample(example):
                question = example.get(q_col) or example.get(instruction_col, "")
                answer   = example.get(a_col) or example.get(output_col, "")
                extra    = example.get(input_col, "")
                text = self._apply_template(question, answer, extra)
                return {"text": text}

        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

        if tokenize:
            return self._tokenize_dataset(dataset)
        return dataset   # 'messages' column for SFTTrainer with assistant_only_loss

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

        # Patch Gemma-4 chat template with {% generation %} markers so that
        # TRL's assistant_only_loss can correctly mask prompt tokens.
        _patch_gemma4_chat_template_for_generation_markers(self.tokenizer)

        # Detect Gemma-4 once so all _load_* methods can branch on self._is_gemma4.
        # The original VLM checkpoint has model_type="gemma4"; the remapped text-only
        # checkpoint we produce has model_type="gemma4_text". Either indicates Gemma-4.
        self._is_gemma4 = False
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            if getattr(_cfg, "model_type", "").startswith("gemma4"):
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
                "Gemma-4 detected — will remap VLM checkpoint shards to Gemma4ForCausalLM "
                "namespace (model.language_model.* → model.*) and load text-only model."
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
        return AutoModelForCausalLM

    def _load_full_precision_model(self, model_kwargs: Dict):
        model_kwargs["dtype"] = torch.bfloat16
        self.model = self._model_cls().from_pretrained(self.model_name, **model_kwargs)
        # gradient_checkpointing disabled at SFTConfig level; no need to enable here
        logger.info("Loaded model for full fine-tuning (BF16)")

    @staticmethod
    def _find_lora_target_modules(model):
        standard = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        logger.info(f"LoRA target modules: {standard}")
        return standard

    @staticmethod
    def _prepare_gemma4_causal_checkpoint(src_path: str, dst_path: str) -> str:
        """Remap VLM safetensors shards to Gemma4ForCausalLM key namespace.

        google/gemma-4-31b-it is a VLM checkpoint: weights live under
        model.language_model.* and model.vision_tower.* etc.  Gemma4ForCausalLM
        (model_type="gemma4_text") expects model.* at the top level.

        This method reads each safetensors shard one at a time (peak ~0.5 GB),
        extracts only the LM keys (model.language_model.* → model.*), saves new
        shards, and writes a matching model.safetensors.index.json + config.json
        with model_type="gemma4_text".  Result is cached at dst_path.

        Returns dst_path (str) — ready for Gemma4ForCausalLM.from_pretrained().
        """
        import shutil
        from pathlib import Path as _Path
        from transformers import AutoConfig

        src = _Path(src_path)
        dst = _Path(dst_path)

        # Cache check — skip if already prepared (sentinel: config.json present)
        if (dst / "config.json").exists():
            logger.info(f"Gemma4 causal checkpoint already at {dst} — reusing cache.")
            return str(dst)

        dst.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing Gemma4ForCausalLM checkpoint: {src} → {dst}")

        # --- safetensors imports ---
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors package required. Run: pip install safetensors")

        # --- Read shard index ---
        index_path = src / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]   # original_key → shard_filename

        all_shards = sorted(set(weight_map.values()))
        new_weight_map: Dict[str, str] = {}

        for shard_name in all_shards:
            shard_path = src / shard_name
            logger.info(f"  Shard {shard_name}: remapping keys …")
            remapped: Dict[str, Any] = {}

            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith("model.language_model."):
                        new_key = key.replace("model.language_model.", "model.", 1)
                        remapped[new_key] = sf.get_tensor(key)
                    elif key.startswith("lm_head."):
                        remapped[key] = sf.get_tensor(key)
                    # vision/audio encoder keys — dropped

            logger.info(f"    → {len(remapped)} LM keys saved")
            dst_shard = dst / shard_name
            save_file(remapped, str(dst_shard))

            for new_key in remapped:
                new_weight_map[new_key] = shard_name

        # --- Write new shard index ---
        new_index = {"metadata": {"format": "pt"}, "weight_map": new_weight_map}
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)
        logger.info(f"  Written model.safetensors.index.json ({len(new_weight_map)} keys)")

        # --- Write config.json: text_config fields at top level, model_type=gemma4_text ---
        orig_cfg = AutoConfig.from_pretrained(str(src))
        text_cfg = orig_cfg.get_text_config()
        text_cfg_dict = text_cfg.to_dict()
        text_cfg_dict["model_type"] = "gemma4_text"
        with open(dst / "config.json", "w") as f:
            json.dump(text_cfg_dict, f, indent=2)
        logger.info("  Written config.json (model_type=gemma4_text)")

        # --- Copy tokenizer and generation config files ---
        for fname in [
            "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
            "generation_config.json", "special_tokens_map.json",
        ]:
            src_f = src / fname
            if src_f.exists():
                shutil.copy2(str(src_f), str(dst / fname))
                logger.info(f"  Copied {fname}")

        logger.info(f"Gemma4ForCausalLM checkpoint ready at {dst}")
        return str(dst)

    def _load_lora_model(self, model_kwargs: Dict):
        model_kwargs["dtype"] = torch.bfloat16

        if self._is_gemma4:
            # VLM checkpoint (model.language_model.*) cannot be loaded directly into
            # Gemma4ForCausalLM (expects model.*). Remap shards to a local cache dir
            # first, then load the text-only model with correct pretrained weights.
            # Peak memory: one shard at a time (~0.5 GB) — far less than loading two
            # full 62 GB models simultaneously.
            try:
                from huggingface_hub import snapshot_download
                src_local = snapshot_download(self.model_name, local_files_only=True)
            except Exception:
                src_local = self.model_name   # already a local path

            dst_local = "/home/ubuntu/.cache/gemma4_causal_text"
            causal_path = self._prepare_gemma4_causal_checkpoint(src_local, dst_local)

            # Force all layers onto GPU 0 (not device_map="auto") to avoid the
            # meta-device / PEFT backward-pass conflict:
            #   "MmBackward0 returned an invalid gradient — expected meta but got cuda:0"
            # device_map="auto" can leave some tensors on meta as disk-offload placeholders;
            # PEFT LoRA then fails during backward.  62GB BF16 fits on 128GB unified memory.
            gemma4_kwargs = {**model_kwargs, "device_map": {"": 0}}
            logger.info(f"Loading Gemma4ForCausalLM from remapped checkpoint: {causal_path}")
            base = AutoModelForCausalLM.from_pretrained(causal_path, **gemma4_kwargs)
        else:
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
        logger.info(f"  Packing             : disabled (packing=False avoids SDPA cross-contamination)")

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
            # packing=False: TRL 1.0's BFD packing auto-enables padding_free=True, which
            # flattens sequences into a 1D stream and relies on FlashAttention 2/3 to prevent
            # cross-sequence attention contamination.  DGX Spark GB10 (SM 12.1 / CUDA 13) has
            # no prebuilt flash_attn wheel → forced to use SDPA, which does NOT support
            # padding-free / variable-length packed sequences. Result: tokens from example B
            # attend to tokens from example A → loss > ln(vocab_size) ≈ 12.45 (worse than
            # random).  Disabling packing removes this dependency entirely. SDPA with standard
            # attention masks handles each example independently and correctly.
            packing                     = False,
            # Compute loss on assistant (response) tokens only.
            # Gemma4ForCausalLM is a plain causal decoder — standard CE loss, no VLM
            # plumbing. The {% generation %} markers patched into the chat template let
            # TRL's apply_chat_template(return_assistant_tokens_mask=True) correctly
            # identify response tokens and mask prompt tokens from the loss.
            assistant_only_loss         = True,
            # NEFTune: add uniform noise to input embeddings during forward pass.
            # Empirically improves fine-tuning quality for instruction-following tasks.
            neftune_noise_alpha         = 5,
            # Track tokens seen — used by DiagnosticCallback for tokens/sec metric.
            include_num_input_tokens_seen = "all",
        )

        logger.info("Starting SFTTrainer setup …")

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
        self._t_last: Optional[float] = None
        self._tokens_last: int = 0

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
        import math, time

        # ── tokens/second throughput ─────────────────────────────────────────
        now = time.monotonic()
        tokens_seen = getattr(state, "num_input_tokens_seen", None)
        if tokens_seen is not None and self._t_last is not None:
            dt = now - self._t_last
            if dt > 0.0:
                logs["tokens_per_second"] = round((tokens_seen - self._tokens_last) / dt)
        if tokens_seen is not None:
            self._tokens_last = tokens_seen
        self._t_last = now

        loss = logs.get("loss")
        if loss is not None:
            # Perplexity — cap at exp(20) ≈ 485M to avoid inf on early chaotic steps
            logs["perplexity"] = round(math.exp(min(loss, 20.0)), 2)

            # Step-1 sanity check: a properly pretrained 31B model on response-only
            # tokens should start at loss≈2–4. Loss>10 means tokens are not masked
            # correctly (loss computed on question tokens) or training is broken.
            VOCAB_RANDOM_BASELINE = math.log(256_000)  # ln(Gemma-4 vocab) ≈ 12.45
            if state.global_step <= 5:
                if loss > VOCAB_RANDOM_BASELINE:
                    logger.warning(
                        f"⚠  Step {state.global_step}: loss={loss:.4f} is WORSE than random "
                        f"(random baseline={VOCAB_RANDOM_BASELINE:.2f}). "
                        "train_on_responses_only may not be masking prompt tokens correctly."
                    )
                elif loss > 6.0:
                    logger.warning(
                        f"⚠  Step {state.global_step}: loss={loss:.4f} is high — expected 2–4 "
                        "for a pretrained 31B model on response tokens. Monitor closely."
                    )
                else:
                    logger.info(
                        f"✓  Step {state.global_step}: loss={loss:.4f} — looks healthy "
                        "(expected 2–4 for pretrained 31B on response-only tokens)."
                    )

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
