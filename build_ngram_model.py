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
    
    storage = NGramStorage(args.db)
    storage.build_from_dataset(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample,
        hf_token=hf_token
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