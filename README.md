# Vietnamese Grammar Checker with SQLite N-gram Storage

A Python-based Vietnamese grammar checker for Microsoft Word that uses N-gram models stored in SQLite for efficient error detection.

## Installation

1. Clone the repository and create a virtual environment:
```bash
cd GrammarChecker
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -e .
```

## Building N-gram Models

### Option 1: Using the UVW-2026 Dataset

The project can build N-gram models from the Vietnamese Wikipedia corpus (UVW-2026) available on Hugging Face.

#### Build a small model for testing (10k documents):
```bash
python build_ngram_model.py --sample 10000 --db ngrams_10k.db
```

#### Build a full model (entire training set, ~890k documents):
```bash
python build_ngram_model.py --split train --db ngrams_full.db
```

#### Build from validation set:
```bash
python build_ngram_model.py --split validation --db ngrams_val.db
```

#### Available options:
```bash
python build_ngram_model.py --help
```

Options:
- `--db`: Path to SQLite database file (default: ngrams.db)
- `--split`: Dataset split (train/validation/test, default: train)
- `--sample`: Number of documents to sample (default: all)
- `--dataset`: Hugging Face dataset name (default: undertheseanlp/UVW-2026)
- `--hf-token`: Hugging Face authentication token (default: from HF_TOKEN env var)

### Option 2: Training In-Memory N-grams

For smaller datasets or testing:

```python
from core.grammar_checker import VietnameseGrammarChecker

checker = VietnameseGrammarChecker()
corpus = [
    "Tôi thích ăn cơm.",
    "Anh ấy đi học.",
    "Chúng ta sẽ gặp nhau vào ngày mai."
]
checker.train_ngram_model(corpus)
```

## Usage

### With SQLite N-gram Database:
```bash
python core.py input.docx --ngram-db ngrams_full.db
```

### With In-Memory N-grams:
```bash
python core.py input.docx
```

### With custom output file:
```bash
python core.py input.docx --output corrected.docx --ngram-db ngrams_full.db
```

### Build N-gram model and benchmark:
```bash
python -m cProfile -o "results.prof" build_ngram_model.py --sample 100 --split train --db ngrams_100.db```
```

## Architecture

### NGramStorage Class
Manages N-gram storage and retrieval from SQLite:

**Methods:**
- `build_from_dataset(dataset_name, split, sample_size)`: Build N-grams from Hugging Face dataset
- `add_unigram(token)`: Add/increment a unigram
- `add_bigram(token1, token2)`: Add/increment a bigram
- `add_trigram(token1, token2, token3)`: Add/increment a trigram
- `get_bigram_frequency(token1, token2)`: Get bigram frequency
- `get_trigram_frequency(token1, token2, token3)`: Get trigram frequency
- `get_stats()`: Get database statistics

### Database Schema

The SQLite database contains three main tables:

**unigrams**:
- id: INTEGER PRIMARY KEY
- token: TEXT (unique)
- frequency: INTEGER

**bigrams**:
- id: INTEGER PRIMARY KEY
- token1_id: FOREIGN KEY to unigrams
- token2_id: FOREIGN KEY to unigrams
- frequency: INTEGER

**trigrams**:
- id: INTEGER PRIMARY KEY
- token1_id: FOREIGN KEY to unigrams
- token2_id: FOREIGN KEY to unigrams
- token3_id: FOREIGN KEY to unigrams
- frequency: INTEGER

## Performance

Database size examples (estimated):
- 10k documents: ~50-100 MB
- 100k documents: ~500 MB - 1 GB
- 890k documents (full training set): ~4-5 GB

Lookup time: < 1ms for bigrams/trigrams

## Testing

Run unit tests:
```bash
python -m unittest test_grammar.py
```

## Key Features

- **Vietnamese-aware tokenization** using underthesea library
- **Efficient N-gram storage** in SQLite with frequency counts
- **Flexible model building** from Hugging Face datasets
- **Support for both in-memory and SQLite-backed models**
- **Grammar error detection** based on N-gram probabilities
- **Microsoft Word integration** via COM automation (Windows)

## Notes

- First run of building the full UVW-2026 model will take several hours
- Database file size depends on the number of documents processed
- For best results, use a model trained on a large corpus (10k+ documents minimum)
- The checker uses smoothing (add-0.01 Laplace smoothing) for unseen N-grams

## Future Enhancements

- Support for other Vietnamese NLP tasks
- Integration as a Word add-in
- Batch processing of multiple documents
- Support for other N-gram sizes
- GPU-accelerated processing for larger datasets

