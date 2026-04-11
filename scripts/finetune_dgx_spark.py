#!/usr/bin/env python3
"""Thin re-export so scripts/finetune_dgx_spark.py stays in sync with the root copy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_dgx_spark import main   # noqa: F401

if __name__ == "__main__":
    main()
