import os
import sys
from pathlib import Path
import pytest

# Make core modules importable
CORE_DIR = Path(__file__).resolve().parent.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


@pytest.fixture(autouse=True)
def patch_tokenizers_and_isolate(monkeypatch, tmp_path):
    """Autouse fixture: deterministic tokenizers and change CWD to tmp_path.

    Runs before each test to ensure deterministic behavior and filesystem isolation.
    """
    def word_tokenize(text):
        if text is None:
            raise TypeError("text must be str")
        return [t for t in str(text).split() if t]

    def sent_tokenize(text):
        if text is None:
            raise TypeError("text must be str")
        parts = []
        for part in str(text).replace("?", ".").replace("!", ".").split("."):
            p = part.strip()
            if p:
                parts.append(p)
        return parts or []

    # Patch underthesea in modules under test (done lazily; modules may be imported later)
    try:
        import core.ngram_storage as ngs
        monkeypatch.setattr(ngs, "underthesea", type("U", (), {"word_tokenize": word_tokenize, "sent_tokenize": sent_tokenize}))
    except Exception:
        pass
    try:
        import core.grammar_checker as gc
        monkeypatch.setattr(gc, "underthesea", type("U", (), {"word_tokenize": word_tokenize, "sent_tokenize": sent_tokenize}))
    except Exception:
        pass

    # change current working directory to tmp_path to isolate ./ngrams/ directory
    monkeypatch.chdir(tmp_path)
    return tmp_path
