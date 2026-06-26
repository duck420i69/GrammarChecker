#!/usr/bin/env python
"""
Script to build N-gram model from UVW-2026 dataset and store in SQLite.
Usage: python build_ngram_model.py [--db ngrams.db] [--split train] [--sample 10000]
"""

import argparse
import os
from dotenv import load_dotenv

from core.ngram_storage import NGramStorage

def main():
    parser = argparse.ArgumentParser(
        description="Build N-gram model from Vietnamese corpus and store in SQLite"
    )
    parser.add_argument(
        "--db", 
        default="ngrams.db",
        help="Path to SQLite database file (default: ngrams.db)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available (uses ./ngrams/<db>.ckpt)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Number of documents between automatic commits/checkpoints (default: 1000)"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of documents to sample (default: all)"
    )
    parser.add_argument(
        "--dataset",
        default="undertheseanlp/UVW-2026",
        help="Hugging Face dataset name (default: undertheseanlp/UVW-2026)"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face authentication token (default: from HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    load_dotenv()

    # Get HF token from argument or environment variable or  .env file
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        print("Using Hugging Face authentication token")
    else:
        print("Warning: No HF_TOKEN provided - using unauthenticated requests")
    
    print(f"Building N-gram model from {args.dataset} dataset...")
    print(f"Database: {args.db}")
    print(f"Split: {args.split}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    
    # Determine checkpoint path
    checkpoint_path = os.path.join("./ngrams", args.db + ".ckpt")

    start_index = 0
    if args.resume:
        # If resuming, read checkpoint if available
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as fh:
                    start_index = int(fh.read().strip() or 0)
                    print(f"Resuming from checkpoint index {start_index}")
            except Exception as e:
                print(f"Warning: failed to read checkpoint file, starting from 0: {e}")
        else:
            print(f"No checkpoint file found at {checkpoint_path}, starting from 0")

    # Keep existing DB when resuming
    storage = NGramStorage(args.db, keep_existing=args.resume)
    storage.build_from_dataset(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample,
        hf_token=hf_token,
        start_index=start_index,
        checkpoint_path=checkpoint_path if args.resume or args.checkpoint_interval > 0 else None,
        checkpoint_interval=args.checkpoint_interval
    )
    
    stats = storage.get_stats()
    print("\n=== N-gram Statistics ===")
    print(f"Unique unigrams: {stats['unique_unigrams']:,}")
    print(f"Unique bigrams: {stats['unique_bigrams']:,}")
    print(f"Unique trigrams: {stats['unique_trigrams']:,}")
    print(f"Total unigram occurrences: {stats['total_unigrams']:,}")
    print(f"Database file: {stats['db_file']}")
    
    storage.close()
    print("\nN-gram model built successfully!")


if __name__ == "__main__":
    main()