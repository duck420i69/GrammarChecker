#!/usr/bin/env python
"""
Example usage of the Vietnamese Grammar Checker with SQLite N-gram storage.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ngram_storage import NGramStorage
from core.grammar_checker import VietnameseGrammarChecker


def example_1_train_and_check_with_memory():
    """Example 1: Train N-gram model in memory and check grammar."""
    print("=" * 50)
    print("Example 1: In-Memory N-gram Model")
    print("=" * 50)
    
    # Initialize checker without database
    checker = VietnameseGrammarChecker()
    
    # Train on sample corpus
    corpus = [
        "Tôi thích ăn cơm.",
        "Anh ấy đi học.",
        "Chúng ta sẽ gặp nhau vào ngày mai.",
        "Cô giáo dạy tiếng Anh.",
        "Em học bài ở nhà."
    ]
    
    print(f"Training on {len(corpus)} sentences...")
    checker.train_ngram_model(corpus)
    
    # Test sentences
    test_sentences = [
        "Tôi thích ăn cơm.",  # Correct
        "cơm ăn thích tôi",    # Incorrect word order
    ]
    
    for sentence in test_sentences:
        tokens = tokenize_vietnamese(sentence)
        token_probs = checker.get_ngram_probability(tokens)
        avg_prob = sum(p for _, _, p in token_probs) / len(token_probs) if token_probs else 0.0
        print(f"Sentence: '{sentence}'")
        print(f"  Tokens: {tokens}")
        print(f"  Avg per-token N-gram probability: {avg_prob:.4f}")
        print()


def example_2_build_and_check_with_sqlite():
    """Example 2: Build SQLite N-gram database and check grammar."""
    print("=" * 50)
    print("Example 2: SQLite-Backed N-gram Model")
    print("=" * 50)
    
    # Build N-gram database from sample (this takes time)
    # Uncomment the following lines to build a new database:
    # print("Building N-gram database from UVW-2026 dataset...")
    # storage = NGramStorage("examples_ngrams.db")
    # storage.build_from_dataset(split="train", sample_size=100)
    # stats = storage.get_stats()
    # print(f"Database built with stats: {stats}")
    # storage.close()
    
    # Initialize checker with existing database
    checker = VietnameseGrammarChecker(ngram_db="test_small.db")
    print("Loaded N-gram database: test_small.db")
    
    # Test sentences
    test_sentences = [
        "Đây là một câu tiếng Việt.",
        "Tôi yêu tiếng Việt.",
    ]
    
    for sentence in test_sentences:
        tokens = tokenize_vietnamese(sentence)
        token_probs = checker.get_ngram_probability(tokens)
        avg_prob = sum(p for _, _, p in token_probs) / len(token_probs) if token_probs else 0.0
        errors = checker.detect_errors(sentence)
        ranges = errors[0][1] if errors else []
        print(f"Sentence: '{sentence}'")
        print(f"  Tokens: {tokens[:5]}... (showing first 5)")
        print(f"  Avg per-token N-gram probability: {avg_prob:.6f}")
        print(f"  Wrong-token ranges: {ranges}")
        if ranges:
            for start, end in ranges:
                print(f"    -> '{sentence[start:end]}' at ({start},{end})")
        print()


def example_3_database_stats():
    """Example 3: Get statistics from N-gram database."""
    print("=" * 50)
    print("Example 3: Database Statistics")
    print("=" * 50)
    
    try:
        storage = NGramStorage("test_small.db")
        stats = storage.get_stats()
        print(f"Unique unigrams: {stats['unique_unigrams']:,}")
        print(f"Unique bigrams: {stats['unique_bigrams']:,}")
        print(f"Unique trigrams: {stats['unique_trigrams']:,}")
        print(f"Total unigram occurrences: {stats['total_unigrams']:,}")
        print(f"Database file: {stats['db_file']}")
        storage.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Vietnamese Grammar Checker Examples\n")
    
    # Run examples
    try:
        example_1_train_and_check_with_memory()
    except Exception as e:
        print(f"Example 1 error: {e}\n")
    
    try:
        example_2_build_and_check_with_sqlite()
    except Exception as e:
        print(f"Example 2 error: {e}\n")
    
    try:
        example_3_database_stats()
    except Exception as e:
        print(f"Example 3 error: {e}\n")

