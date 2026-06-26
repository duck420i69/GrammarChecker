from __future__ import annotations

import collections
import sqlite3
from itertools import chain
from multiprocessing import Pool
from sqlite3 import Connection

from nltk import ngrams as nltk_ngrams
import underthesea
from tqdm import tqdm
from datasets import load_dataset, Dataset

import re
import html
import os


CHUNK_SIZE = 16  # Number of documents to process in each batch for multiprocessing


def filter_dataset(dataset: Dataset):
    # Filter out based on score
    def score_filter(document):
        for category in ["Military", "Politics", "quân sự", "chính trị"]:
            if document.get("main_category", "").lower() == category.lower():
                return True
        return document.get("quality_score", 0) >= 7

    return dataset.filter(score_filter, cache_file_name="./cache/filtered_dataset8.arrow")


def format_date(text):
    text = re.sub(r'\b\d{1,2}[-/. ]\d{1,2}[-/. ]\d{2,4}\b', 'date_token', text)
    return text

def format_number(text):
    text = re.sub(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', 'num_token', text)
    text = re.sub(r'\b\d+\b', 'num_token', text)
    return text


def clean_text(text):
    text = html.unescape(text) # Converts &nbsp; to actual space
    text = re.sub(r'<[^>]*>', ' ', text) # Removes any remaining HTML tags
    text = re.sub(r'\b\w+_\w+\b', '', text)
    # remove words like this "length_mi" that have underscore
    text = format_date(text)
    text = format_number(text)
    # remove separators and punctuation except for Vietnamese characters and spaces
    text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]', ' ', text)
    # Replace multiple spaces with a single space and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_text_symbol(text):
    # Split text into chunks based on punctuation and separators
    chunks = re.split(r'[.,;:!?()\[\]{}"\'\n\r]+', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def preprocess_document(document):
    text = document.get("title", "") + " " + document.get("content", "")
    chunks = split_text_symbol(text)
    clean_chunks = [clean_text(chunk) for chunk in chunks if clean_text(chunk)]
    return clean_chunks


class NGramStorage:
    """Manage N-gram storage and retrieval from SQLite database."""
    
    def __init__(self, db_path="ngrams.db", keep_existing=False):
        # db_path passed as filename (e.g., 'ngrams.db')
        self.db_path = "./ngrams/" + db_path
        self.keep_existing = keep_existing
        self.conn: Connection | None = None
        self.init_db()

        self.unigram_counter = collections.Counter()
        self.bigram_counter = collections.Counter()
        self.trigram_counter = collections.Counter()

        self.id_map = {} # Cache for token to ID mapping
    
    def init_db(self):
        """Initialize SQLite database with tables for N-grams.

        If keep_existing is False and a database file exists, remove it to
        start fresh. If keep_existing is True and file exists, open it and
        reuse tables (CREATE TABLE IF NOT EXISTS used for safety).
        """
        # create directory if missing
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        if os.path.exists(self.db_path) and not self.keep_existing:
            os.remove(self.db_path)

        self.conn = sqlite3.connect(self.db_path)
        if self.conn is None:
            raise RuntimeError("Can't connect to DB. Exit...")
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA journal_mode = OFF")
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA cache_size = 1000000")  # Use 1GB of RAM as cache

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS args (
                id INTEGER PRIMARY KEY,
                save_db TEXT NOT NULL,
                training_set TEXT NOT NULL,
                sample_size INTEGER,
                start_index INTEGER,
                checkpoint_interval INTEGER
            )
        """)
        
        # Create table for unigrams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unigrams (
                id INTEGER PRIMARY KEY,
                token TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1
            )
        """)
        
        # Create table for bigrams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bigrams (
                id INTEGER PRIMARY KEY,
                token1_id INTEGER NOT NULL,
                token2_id INTEGER NOT NULL,
                frequency INTEGER DEFAULT 1,
                FOREIGN KEY (token1_id) REFERENCES unigrams(id),
                FOREIGN KEY (token2_id) REFERENCES unigrams(id),
                UNIQUE(token1_id, token2_id)
            )
        """)
        
        # Create table for trigrams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trigrams (
                id INTEGER PRIMARY KEY,
                token1_id INTEGER NOT NULL,
                token2_id INTEGER NOT NULL,
                token3_id INTEGER NOT NULL,
                frequency INTEGER DEFAULT 1,
                FOREIGN KEY (token1_id) REFERENCES unigrams(id),
                FOREIGN KEY (token2_id) REFERENCES unigrams(id),
                FOREIGN KEY (token3_id) REFERENCES unigrams(id),
                UNIQUE(token1_id, token2_id, token3_id)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_unigrams_token ON unigrams(token)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bigrams ON bigrams(token1_id, token2_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trigrams ON trigrams(token1_id, token2_id, token3_id)")
        
        self.conn.commit()
    
    def get_token_id(self, token, create=True):
        """Get or create a token ID."""
        # use cache to avoid redundant database queries
        if token in self.id_map:
            return self.id_map[token]

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM unigrams WHERE token = ?", (token,))
        result = cursor.fetchone()
        
        if result:
            self.id_map[token] = result[0]
            return result[0]
        elif create:
            cursor.execute("INSERT INTO unigrams (token, frequency) VALUES (?, 0)", (token,))
            self.id_map[token] = cursor.lastrowid
            return cursor.lastrowid
        else:
            return None

    def commit_database(self):
        """Commit changes to the database."""
        if self.conn:
            cursor = self.conn.cursor()

            # Commit unigrams
            cursor.executemany("UPDATE unigrams SET frequency = frequency + ? WHERE id = ?",
                               [(freq, self.get_token_id(token, create=True))
                                for token, freq in self.unigram_counter.items()])
            cursor.executemany("""
            INSERT INTO bigrams (token1_id, token2_id, frequency)
            VALUES (?, ?, ?)
            ON CONFLICT(token1_id, token2_id) DO UPDATE SET frequency = frequency + excluded.frequency
            """, [(self.get_token_id(token1, create=False),
                   self.get_token_id(token2, create=False),
                   freq) for (token1, token2), freq in self.bigram_counter.items()])
            cursor.executemany("""
            INSERT INTO trigrams (token1_id, token2_id, token3_id, frequency)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(token1_id, token2_id, token3_id) DO UPDATE SET frequency = frequency + excluded.frequency
            """, [(self.get_token_id(token1, create=False),
                   self.get_token_id(token2, create=False),
                   self.get_token_id(token3, create=False),
                   freq) for (token1, token2, token3), freq in self.trigram_counter.items()])

            self.conn.commit()

            self.unigram_counter.clear()
            self.bigram_counter.clear()
            self.trigram_counter.clear()

    
    def build_from_dataset(self, dataset_name="undertheseanlp/UVW-2026", split="train",
                           sample_size=None, hf_token=None, start_index=0,
                           checkpoint_path=None, checkpoint_interval=1000):
        """Build N-grams from Hugging Face dataset with optional checkpoint/resume.
            :param sample_size: number of samples to use for training
            :param split: which set use for training ("train", "val", "test")
            :param dataset_name: Huggingface dataset's name
            :param hf_token: Token to access Huggingface (build slower without it)
            :param start_index: index of first document to process (for resume).
            :param checkpoint_path: path to write checkpoint file (stores next index to process).
            :param checkpoint_interval: how many documents between commits/checkpoints.
        """
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, split=split, token=hf_token, streaming=False)
        dataset = filter_dataset(dataset)
        
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        total = len(dataset)
        print(f"Building N-grams from {total} documents... (starting at {start_index})")

        # Sequential processing with periodic commits and checkpoints to allow pause/resume
        processed = 0
        for idx, example in enumerate(tqdm(dataset, total=total, desc="Processing documents")):
            if idx < start_index:
                continue

            uni, bi, tri = self.build_process(example)
            self.unigram_counter.update(uni)
            self.bigram_counter.update(bi)
            self.trigram_counter.update(tri)
            processed += 1

            # Periodic commit + checkpoint
            if processed % checkpoint_interval == 0:
                print(f"Checkpoint at document index {idx+1}. Committing to DB and writing checkpoint...")
                self.commit_database()
                if checkpoint_path:
                    try:
                        with open(checkpoint_path, 'w', encoding='utf-8') as fh:
                            fh.write(str(idx+1))
                    except Exception as e:
                        print(f"Warning: failed to write checkpoint {checkpoint_path}: {e}")

        # Final commit
        self.commit_database()

        # Remove rare n-grams and optimize
        self.conn.execute("DELETE FROM trigrams WHERE frequency = 1")  # Remove rare trigrams to save space
        self.conn.execute("DELETE FROM bigrams WHERE frequency = 1")   # Remove rare bigrams to save space
        self.conn.commit()
        self.conn.execute("VACUUM")
        self.conn.commit()
        
        # Remove checkpoint file when finished
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

        print(f"N-gram model saved to {self.db_path}")

    @staticmethod
    def build_process(example):
        """Process a single document and return N-gram counters."""
        try:
            chunks = preprocess_document(example)
            sentences = chain.from_iterable(map(underthesea.sent_tokenize, chunks))

            unigram_counter = collections.Counter()
            bigram_counter = collections.Counter()
            trigram_counter = collections.Counter()

            for chunk in sentences:
                if not chunk:
                    continue

                tokens = underthesea.word_tokenize(chunk)
                unigram_counter.update(tokens)
                bigram_counter.update(nltk_ngrams(tokens, 2))
                trigram_counter.update(nltk_ngrams(tokens, 3))

            return unigram_counter, bigram_counter, trigram_counter

        except Exception as e:
            print(f"Error processing document: {e}")
            return collections.Counter(), collections.Counter(), collections.Counter()

    def get_bigram_frequency(self, token1, token2):
        """Get frequency of a bigram."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT frequency FROM bigrams 
            WHERE token1_id = (SELECT id FROM unigrams WHERE token = ?) 
            AND token2_id = (SELECT id FROM unigrams WHERE token = ?)
        """, (token1, token2))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    def get_trigram_frequency(self, token1, token2, token3):
        """Get frequency of a trigram."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT frequency FROM trigrams 
            WHERE token1_id = (SELECT id FROM unigrams WHERE token = ?) 
            AND token2_id = (SELECT id FROM unigrams WHERE token = ?)
            AND token3_id = (SELECT id FROM unigrams WHERE token = ?)
        """, (token1, token2, token3))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    def get_stats(self):
        """Get statistics about the N-gram database."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM unigrams")
        unigram_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM bigrams")
        bigram_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trigrams")
        trigram_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(frequency) FROM unigrams")
        total_unigrams = cursor.fetchone()[0] or 0
        
        return {
            "unique_unigrams": unigram_count,
            "unique_bigrams": bigram_count,
            "unique_trigrams": trigram_count,
            "total_unigrams": total_unigrams,
            "db_file": self.db_path
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()