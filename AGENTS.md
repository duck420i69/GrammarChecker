# GrammarChecker AI Agent Guidelines

## General Rules
Respond terse like smart caveman. All technical substance stay. Only fluff die.

## Persistence

ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure. Off only: "stop caveman" / "normal mode".

Default: **full**. Switch: `/caveman lite|full|ultra`.

## Rules

Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms (big not extensive, fix not "implement a solution for"). No tool-call narration, no decorative tables/emoji, no dumping long raw error logs unless asked — quote shortest decisive line. Standard well-known tech acronyms OK (DB/API/HTTP); never invent new abbreviations reader can't decode. Technical terms exact. Code blocks unchanged. Errors quoted exact.

Preserve user's dominant language. User write Portuguese → reply Portuguese caveman. User write Spanish → reply Spanish caveman. Compress the style, not the language. No forced English openings or status phrases. ALWAYS keep technical terms, code, API names, CLI commands, commit-type keywords (feat/fix/...), and exact error strings verbatim — unless user explicitly ask for translation.

No self-reference. Never name or announce the style. No "caveman mode on", "me caveman think", no third-person caveman tags. Output caveman-only — never normal answer plus "Caveman:" recap. Exception: user explicitly ask what the mode is.

Pattern: `[thing] [action] [reason]. [next step].`

Not: "Sure! I'd be happy to help you with that. The issue you're experiencing is likely caused by..."
Yes: "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"

## Intensity

| Level | What change |
|-------|------------|
| **lite** | No filler/hedging. Keep articles + full sentences. Professional but tight |
| **full** | Drop articles, fragments OK, short synonyms. Classic caveman. No tool-call narration, no decorative tables/emoji, no long raw error-log dumps unless asked. Standard acronyms OK; no invented abbreviations |
| **ultra** | Abbreviate prose words (DB/auth/config/req/res/fn/impl) — prose words only, never real code symbols/function names. Strip conjunctions, arrows for causality (X → Y), one word when one word enough. Code symbols, function names, API names, error strings: never abbreviate |

Example — "Why React component re-render?"
- lite: "Your component re-renders because you create a new object reference each render. Wrap it in `useMemo`."
- full: "New object ref each render. Inline object prop = new ref = re-render. Wrap in `useMemo`."
- ultra: "Inline obj prop → new ref → re-render. `useMemo`."

Example — "Explain database connection pooling."
- lite: "Connection pooling reuses open connections instead of creating new ones per request. Avoids repeated handshake overhead."
- full: "Pool reuse open DB connections. No new connection per request. Skip handshake overhead."
- ultra: "Pool = reuse DB conn. Skip handshake → fast under load."

## Auto-Clarity

Drop caveman when:
- Security warnings
- Irreversible action confirmations
- Multi-step sequences where fragment order or omitted conjunctions risk misread
- Compression itself creates technical ambiguity (e.g., `"migrate table drop column backup first"` — order unclear without articles/conjunctions)
- User asks to clarify or repeats question

Resume caveman after clear part done.

Example — destructive op:
> **Warning:** This will permanently delete all rows in the `users` table and cannot be undone.
> ```sql
> DROP TABLE users;
> ```
> Caveman resume. Verify backup exist first.

## Boundaries

Code/commits/PRs: write normal. "stop caveman" or "normal mode": revert. Level persist until changed or session end.

## Project Overview
This is a Python-based Vietnamese grammar checker for Microsoft Word. It uses N-gram models (bigrams and trigrams) stored in SQLite for CPU-only error detection and correction, with no GPU requirements. The project handles Vietnamese tokenization, tone marks, and diacritics correctly.

## Quick Start for AI Agents

**Key Entry Points:**
- `main/main.py` - Grammar checking for .docx files with optional SQLite backend
- `main/build_ngram_model.py` - Build N-gram models from UVW-2026 Vietnamese Wikipedia dataset
- `grammar_checker.py` - Core grammar detection logic (supports in-memory or SQLite backends)
- `ngram_storage.py` - SQLite database management for N-gram storage

**Core Workflow:**
1. Build N-gram model from dataset → SQLite database (optional, or use in-memory)
2. Initialize `VietnameseGrammarChecker` with optional `ngram_db` parameter
3. Tokenize Vietnamese text with `underthesea` library
4. Detect errors using N-gram probability thresholds
5. Process Word documents via COM integration

## Project Structure

```
GrammarChecker/
├── main/                     # Runnable application scripts
│   ├── main.py              # Entry point: python main/main.py input.docx [--ngram-db db]
│   ├── build_ngram_model.py # CLI: python main/build_ngram_model.py --sample 10000 --db ngrams.db
│   ├── examples.py          # Usage examples for both in-memory and SQLite modes
│   └── remove_rare.py       # Maintenance: removes rare N-grams to save disk space
│
├── tests/                    # Test suite
│   ├── test_grammar.py      # Unit tests (unittest framework)
│   ├── quick_test.py        # Integration tests for all components
│   └── clean_text_test.py   # Text preprocessing tests
│
├── Core Modules (imported by other files)
│   ├── grammar_checker.py   # VietnameseGrammarChecker class (dual-mode: in-memory/SQLite)
│   ├── ngram_storage.py     # NGramStorage class for SQLite backend
│   └── word_integration.py  # process_document() function for Word COM integration
│
├── Configuration
│   ├── pyproject.toml       # Dependencies, version constraints (Python >=3.10)
│   ├── AGENTS.md           # This file
│   ├── REORGANIZATION_SUMMARY.md  # Explains folder structure
│   └── README.md           # User documentation
│
└── Data
    ├── ngrams/             # SQLite database directory
    │   └── ngrams.db      # Example database
    └── cache/             # Hugging Face dataset cache
```

## Architecture & Data Flow

### Dual-Mode Design
The checker supports two backends determined at initialization:
1. **In-Memory Mode** (testing/demo): `VietnameseGrammarChecker()` - trains on sample corpus
2. **SQLite Mode** (production): `VietnameseGrammarChecker(ngram_db='path/to/db')` - loads from SQLite

### N-gram Model Structure
SQLite schema (normalized with foreign keys):
```sql
-- Token storage with frequency counts
unigrams(id PRIMARY KEY, token TEXT UNIQUE, frequency INTEGER)

-- Token pair frequencies with composite unique constraint
bigrams(id, token1_id FK, token2_id FK, frequency, UNIQUE(token1_id, token2_id))

-- Token triplet frequencies
trigrams(id, token1_id FK, token2_id FK, token3_id FK, frequency, 
         UNIQUE(token1_id, token2_id, token3_id))
```

### Key Data Flow
1. **Dataset → N-grams**: `build_ngram_model.py` loads UVW-2026 from Hugging Face → tokenizes with underthesea → builds N-grams → stores in SQLite
2. **Text → Tokens**: Vietnamese text → `underthesea.word_tokenize()` → token list
3. **Tokens → Probabilities**: Token sequence → lookup bigram/trigram frequencies → calculate probability with smoothing
4. **Probability → Errors**: Low probability sequences flagged as potential errors
5. **Document → Corrections**: Word document → extract text → detect errors → save corrected document

## Critical Components & Patterns

### 1. VietnameseGrammarChecker (grammar_checker.py)
**Dual-mode class** - switches behavior based on `ngram_db` parameter:

```python
# In-memory mode (training)
checker = VietnameseGrammarChecker()
checker.train_ngram_model(corpus)

# SQLite mode (production)
checker = VietnameseGrammarChecker(ngram_db="ngrams.db")
```

**Key methods:**
- `tokenize_vietnamese(text)` - Uses underthesea for proper Vietnamese segmentation
- `train_ngram_model(corpus)` - Trains in-memory model from list of sentences
- `get_ngram_probability(tokens)` - Returns list of (token, position, probability) tuples — per-token probability (with smoothing)
- `detect_errors(text)` - Returns list of (sentence, [(start,end), ...]) tuples where ranges mark wrong token character offsets in the sentence

**Important:** Handles Vietnamese-specific features:
- Tone marks (á, à, ả, ã, ạ, ă, ắ, etc.)
- Diacritics in Unicode
- Compound words (handled by underthesea)

### 2. NGramStorage (ngram_storage.py)
**SQLite backend manager** - handles all database operations:

```python
storage = NGramStorage("ngrams.db")
storage.build_from_dataset(split="train", sample_size=10000, hf_token=token)
freq = storage.get_bigram_frequency("Tôi", "thích")
stats = storage.get_stats()
storage.close()

# Or use context manager
with NGramStorage("ngrams.db") as storage:
    # ... operations ...
```

**Database initialization:**
- Auto-creates tables on first connection
- Creates indexes on all foreign keys
- Uses PRAGMA settings for performance (journal_mode=OFF, cache_size=1GB)

**Key methods:**
- `build_from_dataset(dataset_name, split, sample_size, hf_token)` - Main entry for building models
- `get_token_id(token, create=True)` - Token lookup with optional creation (uses in-memory cache)
- `add_bigram/add_trigram()` - Insert or increment N-gram frequency
- `get_bigram/trigram_frequency()` - Query N-gram frequency
- `get_stats()` - Return database statistics

**Dataset preprocessing functions:**
- `preprocess_document(doc)` - Extracts and cleans text from dataset documents
- `clean_text(text)` - Removes HTML, handles diacritics, normalizes spaces
- `filter_dataset(dataset)` - Filters by quality_score >= 5

### 3. Word Integration (word_integration.py)
**COM-based Word processing:**

```python
errors = process_document(input_file, output_file, checker)
```

Uses `pywin32` for COM automation:
- Opens .docx via Word COM object
- Extracts full document text
- Detects errors using grammar checker
- Saves corrected document (currently placeholder)

**Note:** Only runs on Windows with Microsoft Word installed.

## Development Workflows

### Running the Grammar Checker
```bash
# From project root
python core/core.py document.docx --output corrected.docx --ngram-db ngrams.db
python core/core.py document.docx  # Uses in-memory model if no --ngram-db
```

### Building N-gram Models
```bash
# Build from UVW-2026 dataset (10k documents for testing)
python core/build_ngram_model.py --sample 10000 --db ngrams_10k.db

# Build full model from training set
python core/build_ngram_model.py --split train --db ngrams_full.db --hf-token YOUR_TOKEN

# Check what's in a database
python core/remove_rare.py  # Also shows most frequent N-grams
```

### Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e .

# Set HF_TOKEN for authenticated Hugging Face access (optional but recommended)
# On Windows:
$env:HF_TOKEN="your_token_here"
# Or in .env file: HF_TOKEN=your_token_here
```

### Running Tests
```bash
# From project root
python tests/test_grammar.py     # Unit tests
python tests/quick_test.py       # Integration tests
python tests/clean_text_test.py  # Preprocessing tests

# Or use unittest discovery
python -m unittest discover -s tests -p "test_*.py"
```

### Testing Locally
```bash
# Quick in-memory test
python -c "
from grammar_checker import VietnameseGrammarChecker
checker = VietnameseGrammarChecker()
checker.train_ngram_model(['Tôi thích ăn cơm.', 'Anh ấy đi học.'])
print(checker.detect_errors('cơm ăn thích tôi'))  # Bad word order
"
```

## Dependencies & Requirements

**Python Version:** >=3.10 (specified in pyproject.toml)

**Core Dependencies:**
- `underthesea` - Vietnamese NLP (word_tokenize, sent_tokenize)
- `nltk` - N-gram generation (`nltk.ngrams`)
- `datasets` - Hugging Face dataset loading (for UVW-2026)
- `pywin32` - Word COM integration (Windows only)
- `python-dotenv` - Environment variable loading (for HF_TOKEN)

**Key External Resources:**
- **Dataset:** `undertheseanlp/UVW-2026` from Hugging Face (Vietnamese Wikipedia, 894k docs)
- **Hugging Face API:** Requires optional HF_TOKEN for authenticated access

## Conventions & Patterns

### Naming & Paths
- Module names: snake_case (`grammar_checker.py`, `ngram_storage.py`)
- Database files: suffix with size (`ngrams_10k.db`, `ngrams_full.db`) for clarity
- Database location: `ngrams/` subdirectory (created automatically)
- Virtual environment: `.venv` (standard location, in .gitignore)

### Main Execution Guard
All runnable files use:
```python
if __name__ == "__main__":
    main()  # or equivalent
```

### Import Pattern (for files in main/ and tests/)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Now can import root-level modules
from core.grammar_checker import VietnameseGrammarChecker
```

### Context Managers
NGramStorage supports `with` statement for auto-closing:
```python
with NGramStorage("ngrams.db") as storage:
    # ... operations ...
    # Auto-closes on exit
```

### Database Context
Frequent `conn.commit()` calls during bulk operations (every 1000 documents in build_from_dataset).

## Common Tasks & Solutions

### Task: Add a new grammar rule
**Where:** `grammar_checker.py` - `detect_errors()` method  
**Pattern:** Check N-gram probability, flag if below threshold  
**Example:** Rule for common Vietnamese word order mistakes

### Task: Support a new dataset
**Where:** `ngram_storage.py` - `build_from_dataset()` method  
**Pattern:** Load from Hugging Face → preprocess → build N-grams → store  
**Key functions:** `filter_dataset()`, `preprocess_document()`, `clean_text()`

### Task: Optimize database performance
**Where:** `ngram_storage.py` - `init_db()` PRAGMA settings and indexes  
**Current optimizations:**
- `journal_mode=OFF` for speed
- `cache_size=1GB` for RAM-based caching
- Indexes on foreign keys and composite keys
- Token ID caching in memory (`self.id_map`)

### Task: Add custom Word document processing
**Where:** `word_integration.py` - `process_document()` function  
**Current:** Reads full text, detects errors, saves document  
**Enhancement:** Add in-place corrections or custom formatting

## Debugging Tips

**Issue: Imports fail when running from main/ or tests/**
- Solution: The `sys.path.insert()` pattern is already added to all files
- Verify: Check first 10 lines of file include Path manipulation

**Issue: Database is very large**
- Solution: Run `python main/remove_rare.py` to remove frequency=1 N-grams
- Also checks: Use `--sample` parameter when building to test first

**Issue: HF_TOKEN not recognized**
- Solution: Set via environment variable: `set HF_TOKEN=your_token` (Windows)
- Or: Create `.env` file in project root with `HF_TOKEN=your_token`
- Note: `load_dotenv()` is called in `build_ngram_model.py`

**Issue: Word integration fails on non-Windows**
- Status: Expected - `pywin32` and Word COM only available on Windows
- Tests handle this gracefully (skip Word integration)

## Edge Cases & Limitations

1. **Vietnamese Handling:** Project correctly handles tone marks and diacritics, but depends on underthesea for proper word segmentation
2. **Large Models:** Full UVW-2026 creates ~500MB+ database; use `--sample` for testing
3. **Windows-Only:** Word COM integration requires Windows + Microsoft Word installed
4. **No GPU:** By design; all operations are CPU-based using SQLite and N-gram lookups
5. **Smoothing:** Uses Laplace smoothing (adding 0.01) to handle unseen N-grams

## Future Extension Points

1. **Statistical Methods:** Add HMM, Viterbi algorithm, or other models alongside N-grams
2. **UI Integration:** Build Word add-in instead of command-line tool
3. **Batch Processing:** Support processing multiple documents at once
4. **Performance:** Consider replacing SQLite with faster KV store (RocksDB, LMDB)
5. **Multi-language:** Adapt for other languages using similar underthesea approach
6. **Advanced Rules:** Add rule-based checks (POS tagging, dependency parsing)

## Version & Compatibility

- **Project Version:** 0.1.0 (in pyproject.toml)
- **Python:** >=3.10
- **Tested:** Python 3.10, 3.11, 3.12 (assumed)
- **OS:** Windows (for Word integration); Linux/Mac (core checking functions)

## References

- **Hugging Face Dataset:** https://huggingface.co/datasets/undertheseanlp/UVW-2026
- **underthesea:** https://github.com/undertheseanlp/underthesea
- **NLTK N-grams:** https://www.nltk.org/api/nltk.util.html#nltk.util.ngrams
- **SQLite:** https://www.sqlite.org/cli.html

