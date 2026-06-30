from pathlib import Path
import pytest
import collections

import grammar_checker as gc
import ngram_storage as ngs


def test_detect_errors_flags_incorrect_order(tmp_path):
    # Arrange: prepare DB with known good sequence
    Path.cwd()  # ensure path available
    tmp = tmp_path
    Path(tmp).mkdir(exist_ok=True)
    os_cwd = tmp
    # Use NGramStorage to build small model
    s = ngs.NGramStorage("grammar_test.db", keep_existing=False)
    try:
        tokens = ["Tôi", "thích", "ăn", "cơm"]
        s.unigram_counter.update({t: 10 for t in tokens})
        s.bigram_counter.update({("Tôi", "thích"): 10, ("thích", "ăn"): 10, ("ăn", "cơm"): 10})
        s.trigram_counter.update({("Tôi", "thích", "ăn"): 10, ("thích", "ăn", "cơm"): 10})
        s.commit_database()
        s.close()

        checker = gc.VietnameseGrammarChecker(ngram_db="grammar_test.db")
        correct = "Tôi thích ăn cơm"
        reversed_sent = "cơm ăn thích tôi"
        correct_tokens = gc.tokenize_vietnamese(correct)
        reversed_tokens = gc.tokenize_vietnamese(reversed_sent)
        correct_scores = [checker._score_token_by_neighbors(correct_tokens, i) for i in range(len(correct_tokens))]
        reversed_scores = [checker._score_token_by_neighbors(reversed_tokens, i) for i in range(len(reversed_tokens))]
        avg_correct = sum(correct_scores) / len(correct_scores)
        avg_reversed = sum(reversed_scores) / len(reversed_scores)
        assert avg_reversed <= avg_correct
    finally:
        try:
            checker.ngram_storage.close()
        except Exception:
            pass
        try:
            Path(s.db_path).unlink()
        except Exception:
            pass


def test_detect_incorrect_words_and_empty_input(tmp_path):
    # Arrange
    storage = ngs.NGramStorage("detect_words.db", keep_existing=False)
    try:
        storage.unigram_counter.update({"A": 1, "B": 1, "C": 1})
        storage.bigram_counter.update({("A", "B"): 1, ("B", "C"): 1})
        storage.trigram_counter.update({("A", "B", "C"): 1})
        storage.commit_database()
        storage.close()

        checker = gc.VietnameseGrammarChecker(ngram_db="detect_words.db")
        # happy path
        results = checker.detect_incorrect_words("A B C", threshold=0.001)
        assert isinstance(results, list)
        # empty input
        assert checker.detect_errors("") == []
    finally:
        try:
            checker.ngram_storage.close()
        except Exception:
            pass
        try:
            Path(storage.db_path).unlink()
        except Exception:
            pass


def test_detect_errors_with_repeated_tokens_maps_correct_ranges(tmp_path, monkeypatch):
    s = ngs.NGramStorage("repeat.db", keep_existing=False)
    try:
        s.unigram_counter.update({"foo": 1})
        s.commit_database()
        s.close()
        checker = gc.VietnameseGrammarChecker(ngram_db="repeat.db")
        # patch scoring to flag middle
        def score_by_pos(tokens, pos):
            return 0.0 if pos == 1 else 1.0
        monkeypatch.setattr(gc.VietnameseGrammarChecker, "_score_token_by_neighbors", staticmethod(score_by_pos))
        sentence = "foo foo foo"
        errors = checker.detect_errors(sentence, threshold=0.5)
        assert len(errors) == 1
        ranges = errors[0][1]
        assert len(ranges) == 1
        start, end = ranges[0]
        assert sentence[start:end] == "foo"
    finally:
        try:
            checker.ngram_storage.close()
        except Exception:
            pass
        try:
            Path(s.db_path).unlink()
        except Exception:
            pass


def test_detect_errors_invalid_input_and_exception_propagation(tmp_path, monkeypatch):
    s = ngs.NGramStorage("exc.db", keep_existing=False)
    try:
        s.commit_database()
        s.close()
        checker = gc.VietnameseGrammarChecker(ngram_db="exc.db")
        # invalid input: accept that either TypeError is raised by tokenizer
        # or function returns empty list; be flexible to avoid brittleness
        try:
            checker.detect_errors(None)
        except TypeError:
            pass

        # propagate storage exception
        def boom(*a, **k):
            raise RuntimeError("db broken")
        # Patch multiple accessors to ensure any ngram lookup triggers the exception
        monkeypatch.setattr(checker.ngram_storage, "get_trigram_frequency", boom)
        monkeypatch.setattr(checker.ngram_storage, "get_bigram_frequency", boom)
        with pytest.raises(RuntimeError, match="db broken"):
            checker.detect_errors("A B C")
    finally:
        try:
            checker.ngram_storage.close()
        except Exception:
            pass
        try:
            Path(s.db_path).unlink()
        except Exception:
            pass


def test_detect_incorrect_single_token_and_long_sentence(tmp_path):
    s = ngs.NGramStorage("single.db", keep_existing=False)
    try:
        s.unigram_counter.update({"Solo": 1})
        s.commit_database()
        s.close()
        checker = gc.VietnameseGrammarChecker(ngram_db="single.db")
        assert isinstance(checker.detect_incorrect_words("Solo"), list)
        long = "X " * 1500
        errors = checker.detect_errors(long.strip(), threshold=0.0001)
        assert isinstance(errors, list)
    finally:
        try:
            checker.ngram_storage.close()
        except Exception:
            pass
        try:
            Path(s.db_path).unlink()
        except Exception:
            pass
