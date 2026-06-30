import os
from pathlib import Path
import collections
import pytest

import ngram_storage as ngs


def test_init_creates_tables_and_stats(tmp_path):
    # Arrange
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("test_ngrams.db", keep_existing=False)
    try:
        # Act
        stats = storage.get_stats()
        # Assert
        assert isinstance(stats, dict)
        assert stats["unique_unigrams"] == 0
        assert stats["unique_bigrams"] == 0
        assert stats["unique_trigrams"] == 0
        assert stats["total_unigrams"] == 0
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_get_token_id_creates_and_reuses_id(tmp_path):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("ids.db", keep_existing=False)
    try:
        # Act
        id1 = storage.get_token_id("tok", create=True)
        id2 = storage.get_token_id("tok", create=False)
        # Assert
        assert isinstance(id1, int) and id1 > 0
        assert id1 == id2
        other = storage.get_token_id("other", create=True)
        assert other != id1
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_get_token_id_returns_none_when_missing(tmp_path):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("id_none.db", keep_existing=False)
    try:
        assert storage.get_token_id("missing", create=False) is None
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_commit_database_inserts_ngrams_and_getters(tmp_path):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("commit.db", keep_existing=False)
    try:
        # Arrange
        storage.unigram_counter.update({"A": 2, "B": 1, "X": 1})
        storage.bigram_counter.update({("A", "B"): 3})
        storage.trigram_counter.update({("X", "A", "B"): 1})
        # Act
        storage.commit_database()
        # Assert
        assert storage.get_bigram_frequency("A", "B") == 3
        assert storage.get_trigram_frequency("X", "A", "B") == 1
        stats = storage.get_stats()
        assert stats["unique_unigrams"] >= 3
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_commit_database_with_empty_counters_does_not_raise(tmp_path):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("empty_commit.db", keep_existing=False)
    try:
        storage.commit_database()
        stats = storage.get_stats()
        assert stats["unique_unigrams"] == 0
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_commit_database_increments_existing_counts(tmp_path):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("increment.db", keep_existing=False)
    try:
        storage.unigram_counter.update({"A": 1})
        storage.commit_database()
        assert storage.get_unigram_frequency("A") == 1
        # Ensure unigram for B exists before inserting bigram
        storage.unigram_counter.update({"A": 2, "B": 0})
        storage.bigram_counter.update({("A", "B"): 1})
        storage.commit_database()

        assert storage.get_unigram_frequency("A") == 3
        assert storage.get_bigram_frequency("A", "B") == 1
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_build_from_dataset_writes_and_removes_checkpoint(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    storage = ngs.NGramStorage("build_ds.db", keep_existing=False)
    try:
        sample = [{"title": "t1", "content": "A B C"}, {"title": "t2", "content": "D E F"}]
        monkeypatch.setattr(ngs, "load_dataset", lambda *a, **k: sample)
        monkeypatch.setattr(ngs, "filter_dataset", lambda d: d)
        checkpoint = tmp_path / "cp.txt"
        storage.build_from_dataset(dataset_name="fake", split="train", sample_size=None, start_index=0,
                                   checkpoint_path=str(checkpoint), checkpoint_interval=1)
        assert not checkpoint.exists()
        stats = storage.get_stats()
        assert stats["unique_unigrams"] > 0
    finally:
        storage.close()
        Path(storage.db_path).unlink()


def test_build_process_returns_counters_for_document():
    example = {"title": "Title", "content": "Tôi thích ăn cơm. Em đi học."}
    uni, bi, tri = ngs.NGramStorage.build_process(example)
    assert isinstance(uni, collections.Counter)
    assert uni.get("Tôi", 0) >= 1


def test_build_process_handles_tokenizer_exception(monkeypatch):
    def fail_tokenize(_):
        raise RuntimeError("tokenizer failure")
    monkeypatch.setattr(ngs, "underthesea", type("U", (), {"word_tokenize": fail_tokenize, "sent_tokenize": lambda t: [t]}))
    example = {"title": "t", "content": "c"}
    uni, bi, tri = ngs.NGramStorage.build_process(example)
    assert sum(uni.values()) == 0


def test_build_process_handles_large_document_speed():
    long_tokens = "word " * 2000
    example = {"title": "t", "content": long_tokens}
    uni, bi, tri = ngs.NGramStorage.build_process(example)
    assert sum(uni.values()) >= 2000
    assert sum(bi.values()) >= 1999
    assert sum(tri.values()) >= 1998


def test_storage_init_keep_existing_behavior(tmp_path):
    os.chdir(tmp_path)
    dbname = "keep_existing.db"
    s1 = ngs.NGramStorage(dbname, keep_existing=False)
    try:
        s1.unigram_counter.update({"T": 1})
        s1.commit_database()
        s1.close()
        s2 = ngs.NGramStorage(dbname, keep_existing=True)
        assert Path(s2.db_path).exists()
    finally:
        try:
            s2.close()
        except Exception:
            pass
        Path(s1.db_path).unlink()
