import math
import underthesea
from ngram_storage import NGramStorage


def tokenize_vietnamese(text):
    """Tokenize Vietnamese text, preserving diacritics and compound words.

    Args:
        text: Input Vietnamese text string.

    Returns:
        List of tokenized words/compounds.
    """
    return underthesea.word_tokenize(text)


class VietnameseGrammarChecker:
    """Vietnamese grammar checker using N-gram models with forward and backward scoring.

    Uses NGramStorage (SQLite-backed) exclusively for all n-gram lookups.
    Computes grammar probability by combining forward and backward n-gram
    probabilities for each position in the token sequence, using Laplace smoothing.
    """

    def __init__(self, ngram_db, alpha=0.01):
        """
        Initialize the grammar checker with an N-gram database.

        Args:
            ngram_db: Path to the SQLite N-gram database (relative to ./ngrams/).
            alpha: Laplace smoothing parameter (default: 0.01).
        """
        if ngram_db is None:
            raise ValueError(
                "ngram_db must be provided. "
                "Use a pre-built SQLite N-gram database instead."
            )
        self.ngram_storage = NGramStorage(ngram_db)
        self.alpha = alpha
        stats = self.ngram_storage.get_stats()
        self.vocab_size = stats.get('unique_unigrams', 0)

    def get_ngram_probability(self, tokens, n=3):
        """Return per-token N-gram probabilities using bidirectional scoring.

        For each token position, computes the average bidirectional N-gram
        score of all windows that include that token. Uses the same window
        scoring as get_token_anomaly_scores/_window_score.

        Args:
            tokens: List of token strings.
            n: N-gram size (2 for bigrams, 3 for trigrams). Default is 3.

        Returns:
            List of (token, position, probability) tuples — per-token probability
            score between 0.0 and 1.0. Tokens at edges may have fewer contributing
            windows and receive the score of the nearest full window.
        """
        # Reuse existing token scoring routine which already computes
        # average bidirectional window scores per token.
        return self.get_token_anomaly_scores(tokens, n=n)

    def _window_score(self, t1, t2, t3):
        """Compute the bidirectional score for a single trigram window (t1, t2, t3)."""
        forward_freq = self.ngram_storage.get_trigram_frequency(t1, t2, t3)
        forward_prefix_freq = self.ngram_storage.get_bigram_frequency(t1, t2)

        backward_freq = self.ngram_storage.get_trigram_frequency(t3, t2, t1)
        backward_prefix_freq = self.ngram_storage.get_bigram_frequency(t2, t3)

        forward_prob = (forward_freq + self.alpha) / (
            forward_prefix_freq + self.alpha * self.vocab_size
        )
        backward_prob = (backward_freq + self.alpha) / (
            backward_prefix_freq + self.alpha * self.vocab_size
        )

        return math.sqrt(forward_prob * backward_prob)

    def _bigram_window_score(self, t1, t2):
        """Compute the bidirectional score for a single bigram window (t1, t2)."""
        forward_freq = self.ngram_storage.get_bigram_frequency(t1, t2)
        backward_freq = self.ngram_storage.get_bigram_frequency(t2, t1)

        forward_prob = (forward_freq + self.alpha) / (
            forward_freq + self.alpha * self.vocab_size
        )
        backward_prob = (backward_freq + self.alpha) / (
            backward_freq + self.alpha * self.vocab_size
        )

        return math.sqrt(forward_prob * backward_prob)

    def get_token_anomaly_scores(self, tokens, n=3):
        """Score each token in the sequence by how anomalous it appears.

        For each token position, averages the bidirectional n-gram scores of
        all windows that include that token. A low score means the token is
        likely incorrect or out of context.

        Args:
            tokens: List of token strings.
            n: N-gram size (2 for bigrams, 3 for trigrams). Default is 3.

        Returns:
            List of (token, position, score) tuples — one per token.
            Tokens at the edges of the sequence may have fewer contributing
            windows and are assigned the score of the closest full window.
        """
        if len(tokens) < 2 or self.vocab_size == 0:
            return [(token, i, 0.0) for i, token in enumerate(tokens)]

        scores = [0.0] * len(tokens)
        weights = [0] * len(tokens)

        if n == 3:
            for i in range(2, len(tokens)):
                t1, t2, t3 = tokens[i - 2], tokens[i - 1], tokens[i]
                score = self._window_score(t1, t2, t3)
                # This window contributes to all three positions in it
                scores[i - 2] += score
                scores[i - 1] += score
                scores[i] += score
                weights[i - 2] += 1
                weights[i - 1] += 1
                weights[i] += 1

        elif n == 2:
            for i in range(1, len(tokens)):
                t1, t2 = tokens[i - 1], tokens[i]
                score = self._bigram_window_score(t1, t2)
                scores[i - 1] += score
                scores[i] += score
                weights[i - 1] += 1
                weights[i] += 1

        # Normalize by weight and handle edge tokens with zero weight
        result = []
        for i, token in enumerate(tokens):
            if weights[i] > 0:
                avg_score = scores[i] / weights[i]
            else:
                avg_score = 0.0
            result.append((token, i, avg_score))

        return result

    def detect_incorrect_words(self, text, threshold=0.1, n=3):
        """Detect specific incorrect words in a sentence based on anomaly scores.

        Splits text into sentences, then for each sentence evaluates every
        token's anomaly score. Tokens with a score below the threshold are
        returned as likely incorrect.

        Args:
            text: Input Vietnamese text string.
            threshold: Score threshold below which a token is flagged
                       as potentially incorrect (default: 0.1).
            n: N-gram size (2 for bigrams, 3 for trigrams). Default is 3.

        Returns:
            List of (sentence, [(token, position_in_sentence, score), ...]) tuples.
            Each entry represents a sentence that has at least one suspicious word.
        """
        sentences = underthesea.sent_tokenize(text)
        results = []

        for sentence in sentences:
            tokens = tokenize_vietnamese(sentence)
            token_scores = self.get_token_anomaly_scores(tokens, n=n)
            flagged = [
                (token, pos, score)
                for token, pos, score in token_scores
                if score < threshold
            ]
            if flagged:
                results.append((sentence, flagged))

        return results

    def detect_errors(self, text, threshold=0.1, n=3):
        """Detect potential grammar errors and return character ranges of wrong tokens.

        Splits text into sentences, tokenizes each, computes per-token anomaly scores
        (bidirectional N-gram average). Tokens with score below `threshold` are
        considered suspicious. For each suspicious token, returns its character
        start/end positions within the sentence.

        Args:
            text: Input Vietnamese text string.
            threshold: Score threshold below which a token is flagged
                       as potentially incorrect (default: 0.1).
            n: N-gram size used for scoring (default: 3).

        Returns:
            List of (sentence, [(start, end), ...]) tuples. `start` and `end`
            are character offsets within the sentence for each flagged token.
        """
        sentences = underthesea.sent_tokenize(text)
        errors = []

        for sentence in sentences:
            tokens = tokenize_vietnamese(sentence)
            if not tokens:
                continue

            # Get per-token anomaly scores: list of (token, pos, score)
            token_scores = self.get_token_anomaly_scores(tokens, n=n)

            # Find token indices flagged as suspicious
            flagged_indices = [pos for _, pos, score in token_scores if score < threshold]
            if not flagged_indices:
                continue

            # Map token indices to character ranges in the sentence
            ranges = []
            search_pos = 0
            for token, pos, score in token_scores:
                if pos not in flagged_indices:
                    # advance search_pos to consume token even if not flagged
                    idx = sentence.find(token, search_pos)
                    if idx != -1:
                        search_pos = idx + len(token)
                    continue

                # find token occurrence starting from search_pos
                start = sentence.find(token, search_pos)
                if start == -1:
                    # fallback: try from beginning
                    start = sentence.find(token)
                if start == -1:
                    # cannot map token to sentence; skip
                    continue
                end = start + len(token)
                ranges.append((start, end))
                search_pos = end

            if ranges:
                errors.append((sentence, ranges))

        return errors

    def suggest_corrections(self, error_sentence):
        """Simple correction suggestion (placeholder for more advanced logic).

        Args:
            error_sentence: Sentence flagged as potentially erroneous.

        Returns:
            The original sentence (placeholder — no corrections applied yet).
        """
        return error_sentence