import sys
import argparse
from pathlib import Path

from core.grammar_checker import VietnameseGrammarChecker
from core.word_integration import process_document

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Grammar Checker for Word")
    parser.add_argument("input_file", help="Path to input .docx file")
    parser.add_argument("--output", help="Path to output .docx file", default=None)
    parser.add_argument("--ngram-db", help="Path to SQLite N-gram database", default=None)
    args = parser.parse_args()

    if not args.output:
        args.output = args.input_file.replace(".docx", "_corrected.docx")

    # Initialize checker with optional SQLite N-gram database
    checker = VietnameseGrammarChecker(ngram_db=args.ngram_db)

    # Only train in-memory model if not using SQLite database
    if not args.ngram_db:
        sample_corpus = [
            "Tôi thích ăn cơm.",
            "Anh ấy đi học.",
            "Chúng ta sẽ gặp nhau vào ngày mai."
        ]
        checker.train_ngram_model(sample_corpus)
    else:
        print(f"Using N-gram database: {args.ngram_db}")

    # Process the document
    errors = process_document(args.input_file, args.output, checker)

    print(f"Processed {args.input_file}, found {len(errors)} potential errors. Output saved to {args.output}")

if __name__ == "__main__":
    main()
