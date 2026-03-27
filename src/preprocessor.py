# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline for the NLP Summarization Pipeline.

Provides a complete NLP preprocessing pipeline for Indonesian text,
including case folding, cleaning, tokenization, stopword removal, and stemming.
"""

import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import nltk

# Add parent directory to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config

logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource_name)
            nltk.download(resource_name, quiet=True)


# Ensure NLTK resources are available at import time
_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class TextPreprocessor:
    """
    Complete NLP preprocessing pipeline for Indonesian text.

    Each preprocessing step is implemented as a separate callable method,
    and a full pipeline method runs all steps in sequence.

    Attributes:
        custom_stopwords: Additional stopwords to include in removal.
        indonesian_stopwords: Combined set of Indonesian stopwords.
        stemmer: Indonesian stemmer instance from PySastrawi.
    """

    def __init__(self, custom_stopwords: Optional[List[str]] = None) -> None:
        """
        Initialize the TextPreprocessor.

        Args:
            custom_stopwords: Optional list of additional stopwords to remove.
        """
        # Load Indonesian stopwords
        try:
            self.indonesian_stopwords = set(stopwords.words("indonesian"))
        except OSError:
            logger.warning("Indonesian stopwords not found in NLTK. Using empty set.")
            self.indonesian_stopwords = set()

        # Add custom stopwords if provided
        if custom_stopwords:
            self.indonesian_stopwords.update(custom_stopwords)
        if config.CUSTOM_STOPWORDS:
            self.indonesian_stopwords.update(config.CUSTOM_STOPWORDS)

        logger.info(
            "Stopword list loaded with %d words", len(self.indonesian_stopwords)
        )

        # Initialize Indonesian stemmer
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("PySastrawi stemmer initialized successfully")
        except ImportError:
            logger.warning(
                "PySastrawi not installed. Stemming will be skipped. "
                "Install with: pip install PySastrawi"
            )
            self.stemmer = None

    def case_folding(self, text: str) -> str:
        """
        Step 1: Convert all text to lowercase.

        Args:
            text: Input text string.

        Returns:
            Lowercased text string.
        """
        if not text:
            return ""
        return text.lower()

    def clean_text(self, text: str) -> str:
        """
        Step 2: Clean text by removing URLs, emails, HTML tags,
        special characters, and extra whitespace.

        Preserves sentence-ending punctuation (., ?, !) for later
        sentence tokenization.

        Args:
            text: Input text string.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove numbers with special chars (e.g., reference numbers)
        text = re.sub(r"\[\d+\]", " ", text)

        # Remove special characters but preserve sentence-ending punctuation
        # Keep letters, spaces, and sentence-ending punctuation
        text = re.sub(r"[^a-zA-Z\s.?!]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Step 3: Split text into a list of sentences.

        Uses NLTK sent_tokenize with Indonesian language support.

        Args:
            text: Input text string.

        Returns:
            List of sentence strings.
        """
        if not text:
            return []

        try:
            sentences = sent_tokenize(text, language="indonesian")
        except LookupError:
            # Fallback to default tokenizer if Indonesian model not available
            logger.warning(
                "Indonesian sentence tokenizer not available. Using default."
            )
            sentences = sent_tokenize(text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def word_tokenize(self, text: str) -> List[str]:
        """
        Step 4: Tokenize text into words.

        Args:
            text: Input text string (a single sentence or full text).

        Returns:
            List of word tokens.
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text, language="indonesian")
        except LookupError:
            logger.warning("Indonesian word tokenizer not available. Using default.")
            tokens = word_tokenize(text)

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Step 5: Remove Indonesian stopwords from a list of tokens.

        Args:
            tokens: List of word tokens.

        Returns:
            List of tokens with stopwords removed.
        """
        if not tokens:
            return []

        filtered = [
            token
            for token in tokens
            if token.lower() not in self.indonesian_stopwords
            and len(token) > 1  # Remove single-character tokens
        ]
        return filtered

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Step 6: Apply Indonesian stemming to a list of tokens.

        Uses PySastrawi StemmerFactory for Indonesian language stemming.

        Args:
            tokens: List of word tokens.

        Returns:
            List of stemmed tokens.
        """
        if not tokens:
            return []

        if self.stemmer is None:
            logger.warning("Stemmer not available. Returning tokens unchanged.")
            return tokens

        stemmed = [self.stemmer.stem(token) for token in tokens]
        return stemmed

    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """
        Step 7: Run the full preprocessing pipeline on input text.

        Executes all preprocessing steps in order:
        1. Case folding
        2. Cleaning
        3. Sentence tokenization (on cleaned text)
        4. Word tokenization per sentence
        5. Stopword removal
        6. Stemming

        Args:
            text: Raw input text string.

        Returns:
            Tuple of:
                - fully_preprocessed_text: Single string of preprocessed text
                  (stemmed, stopwords removed).
                - sentences: List of cleaned sentences (before word-level
                  processing, useful for extractive summarization).
        """
        if not text or not text.strip():
            return "", []

        # Step 1: Case folding
        text_lower = self.case_folding(text)

        # Step 2: Cleaning
        text_clean = self.clean_text(text_lower)

        # Step 3: Sentence tokenization (on clean text)
        sentences = self.sentence_tokenize(text_clean)

        # Steps 4-6 per sentence: Word tokenize, remove stopwords, stem
        processed_sentences = []
        for sentence in sentences:
            tokens = self.word_tokenize(sentence)
            tokens = self.remove_stopwords(tokens)
            tokens = self.stem_tokens(tokens)
            if tokens:
                processed_sentences.append(" ".join(tokens))

        # Combine all processed tokens into a single text
        fully_preprocessed_text = " ".join(processed_sentences)

        return fully_preprocessed_text, sentences

    def preprocess_for_extractive(self, text: str) -> List[str]:
        """
        Preprocess text specifically for extractive summarization.

        Returns cleaned sentences suitable for TF-IDF and TextRank.
        Only applies case folding and cleaning, preserving sentence structure.

        Args:
            text: Raw input text string.

        Returns:
            List of cleaned sentence strings.
        """
        if not text or not text.strip():
            return []

        text_lower = self.case_folding(text)
        text_clean = self.clean_text(text_lower)
        sentences = self.sentence_tokenize(text_clean)

        return sentences


if __name__ == "__main__":
    """Test the TextPreprocessor when run directly."""
    from src.utils import setup_logging

    setup_logging("DEBUG")

    preprocessor = TextPreprocessor()

    sample_text = (
        "Penelitian ini bertujuan untuk menganalisis pengaruh media sosial "
        "terhadap prestasi akademik mahasiswa di Universitas Indonesia. "
        "Metode yang digunakan adalah survei kuantitatif dengan sampel 200 "
        "mahasiswa dari berbagai fakultas. Hasil penelitian menunjukkan bahwa "
        "terdapat korelasi negatif antara durasi penggunaan media sosial "
        "dengan indeks prestasi kumulatif (IPK) mahasiswa. "
        "Untuk informasi lebih lanjut, kunjungi https://example.com atau "
        "hubungi email@test.com. <b>Referensi:</b> [1] [2] [3]"
    )

    print("=" * 60)
    print("ORIGINAL TEXT:")
    print(sample_text)

    print("\n" + "=" * 60)
    print("STEP 1 — Case Folding:")
    print(preprocessor.case_folding(sample_text))

    print("\n" + "=" * 60)
    print("STEP 2 — Cleaning:")
    cleaned = preprocessor.clean_text(preprocessor.case_folding(sample_text))
    print(cleaned)

    print("\n" + "=" * 60)
    print("STEP 3 — Sentence Tokenization:")
    sentences = preprocessor.sentence_tokenize(cleaned)
    for i, sent in enumerate(sentences):
        print(f"  [{i}] {sent}")

    print("\n" + "=" * 60)
    print("STEP 4 — Word Tokenization (first sentence):")
    if sentences:
        tokens = preprocessor.word_tokenize(sentences[0])
        print(f"  {tokens}")

    print("\n" + "=" * 60)
    print("STEP 5 — Stopword Removal:")
    if sentences:
        tokens = preprocessor.word_tokenize(sentences[0])
        filtered = preprocessor.remove_stopwords(tokens)
        print(f"  Before: {tokens}")
        print(f"  After:  {filtered}")

    print("\n" + "=" * 60)
    print("STEP 6 — Stemming:")
    if sentences:
        tokens = preprocessor.word_tokenize(sentences[0])
        filtered = preprocessor.remove_stopwords(tokens)
        stemmed = preprocessor.stem_tokens(filtered)
        print(f"  Before: {filtered}")
        print(f"  After:  {stemmed}")

    print("\n" + "=" * 60)
    print("STEP 7 — Full Pipeline:")
    preprocessed_text, clean_sentences = preprocessor.preprocess(sample_text)
    print(f"  Preprocessed text: {preprocessed_text}")
    print(f"  Clean sentences ({len(clean_sentences)}):")
    for i, sent in enumerate(clean_sentences):
        print(f"    [{i}] {sent}")

    print("\nPreprocessor module test completed successfully.")
