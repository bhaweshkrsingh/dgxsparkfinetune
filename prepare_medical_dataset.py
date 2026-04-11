#!/usr/bin/env python3
"""
Medical Dataset Preprocessor
==============================
Merges 50 parquet shards from /home/ubuntu/medAI/trgData into a single
Parquet file with only the 'question' and 'answer' columns.

Why Parquet (not JSON/CSV)?
  - Columnar: reads only requested columns; skips 'responses' entirely on disk
  - Snappy-compressed: ~4-5x smaller than equivalent JSON
  - Arrow-native: HuggingFace datasets loads it with zero-copy memory mapping
  - Fast: no JSON parsing overhead during training data loading

Usage:
  python prepare_medical_dataset.py
  python prepare_medical_dataset.py --input-dir /path/to/parquets --output /path/to/out.parquet
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = "/home/ubuntu/medAI/trgData"
DEFAULT_OUTPUT    = "/home/ubuntu/medAI/medical_train.parquet"

KEEP_COLS = ["question", "answer"]


def merge_parquets(input_dir: str, output_path: str, min_answer_chars: int = 50) -> None:
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print(f"[ERROR] No parquet files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} parquet shards in {input_dir}")
    print(f"Keeping columns: {KEEP_COLS}")
    print(f"Filtering out rows where answer < {min_answer_chars} characters\n")

    tables = []
    total_raw = 0
    total_kept = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        tbl = pq.read_table(fpath, columns=KEEP_COLS)
        total_raw += tbl.num_rows

        # Drop rows where question or answer is null / too short
        import pyarrow.compute as pc
        mask_a = pc.greater(pc.utf8_length(tbl["answer"]),   min_answer_chars)
        mask_q = pc.greater(pc.utf8_length(tbl["question"]), 10)
        mask   = pc.and_(mask_a, mask_q)
        tbl = tbl.filter(mask)
        total_kept += tbl.num_rows
        tables.append(tbl)

        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(files)}] {os.path.basename(fpath)}: "
              f"{tbl.num_rows:>7,} rows kept  ({elapsed:.1f}s)")

    print(f"\nConcatenating {len(tables)} tables...")
    merged = pa.concat_tables(tables)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {merged.num_rows:,} rows to {output_path} ...")
    pq.write_table(
        merged,
        output_path,
        compression="snappy",        # fast read, good compression
        row_group_size=50_000,       # ~50k rows/group; good for streaming reads
    )

    size_mb = out.stat().st_size / (1024 ** 2)
    elapsed_total = time.time() - t0

    print("\n" + "=" * 60)
    print("Preprocessing complete")
    print("=" * 60)
    print(f"  Input rows   : {total_raw:>10,}")
    print(f"  Output rows  : {total_kept:>10,}  (dropped {total_raw - total_kept:,} short/null rows)")
    print(f"  Output file  : {output_path}")
    print(f"  File size    : {size_mb:.1f} MB")
    print(f"  Elapsed      : {elapsed_total:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Merge medical parquet shards")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help="Directory with train-*.parquet files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output merged parquet path")
    parser.add_argument("--min-answer-chars", type=int, default=50,
                        help="Drop rows where answer is shorter than this (removes stubs)")
    args = parser.parse_args()

    merge_parquets(args.input_dir, args.output, args.min_answer_chars)


if __name__ == "__main__":
    main()
