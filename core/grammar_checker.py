import underthesea
from nltk import ngrams
from collections import defaultdict, Counter
import re
from ngram_storage import NGramStorage

class VietnameseGrammarChecker:
    def __init__(self, ngram_db=None):
        self.ngram_model = defaultdict(Counter)
        self.vocab = set()
        self.ngram_storage = None
        if ngram_db:
            self.ngram_storage = NGramStorage(ngram_db)

    def tokenize_vietnamese(self, text):
        """Tokenize Vietnamese text, preserving diacritics and compound words."""
        # Use underthesea for word segmentation
        tokens = underthesea.word_tokenize(text)
        # Normalize diacritics if needed, but underthesea handles them well
        return tokens

    def train_ngram_model(self, corpus, n=3):
        """Train N-gram model on Vietnamese corpus."""
        for sentence in corpus:
            tokens = self.tokenize_vietnamese(sentence)
            self.vocab.update(tokens)
            for gram in ngrams(tokens, n):
                prefix = gram[:-1]
                suffix = gram[-1]
                self.ngram_model[prefix][suffix] += 1

    def get_ngram_probability(self, tokens, n=3):
        """Calculate probability of a sequence using N-gram model."""
        if len(tokens) < n:
            return 0.0
        prob = 1.0
        
        # If using SQLite-backed model
        if self.ngram_storage:
            for i in range(n-1, len(tokens)):
                if n == 2:
                    freq = self.ngram_storage.get_bigram_frequency(tokens[i-1], tokens[i])
                elif n == 3:
                    freq = self.ngram_storage.get_trigram_frequency(tokens[i-2], tokens[i-1], tokens[i])
                else:
                    freq = 1  # Fallback for other n-gram sizes
                
                prob *= (freq + 0.01) / max(freq + 1, 1)  # Smoothing
            return prob
        
        # Otherwise use in-memory model
        for i in range(n-1, len(tokens)):
            prefix = tuple(tokens[i-n+1:i])
            suffix = tokens[i]
            if prefix in self.ngram_model and suffix in self.ngram_model[prefix]:
                prob *= self.ngram_model[prefix][suffix] / sum(self.ngram_model[prefix].values())
            else:
                prob *= 0.01  # Smoothing for unseen n-grams
        return prob

    def detect_errors(self, text):
        """Detect potential grammar errors based on low N-gram probability."""
        sentences = underthesea.sent_tokenize(text)
        errors = []
        for sentence in sentences:
            tokens = self.tokenize_vietnamese(sentence)
            prob = self.get_ngram_probability(tokens)
            if prob < 0.1:  # Threshold for error detection
                errors.append((sentence, prob))
        return errors

    def suggest_corrections(self, error_sentence):
        """Simple correction suggestion (placeholder for more advanced logic)."""
        # For now, just return the sentence as is; could implement edit distance or rule-based corrections
        return error_sentence
