# -*- coding: utf-8 -*-
"""
Extractive Summarization module using TextRank + TF-IDF.

Implements a graph-based extractive summarizer that uses TF-IDF
sentence representations and PageRank scoring to select the most
important sentences from a document.
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Add parent directory to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class ExtractiveSummarizer:
    """
    TextRank-based extractive summarizer with TF-IDF sentence scoring.

    Uses TF-IDF vectorization to represent sentences, builds a cosine
    similarity graph, and applies the PageRank algorithm to rank sentences
    by importance. CPU-only — no GPU required.

    Attributes:
        num_sentences: Number of sentences to extract for the summary.
        max_features: Maximum number of TF-IDF features.
        preprocessor: TextPreprocessor instance for text preprocessing.
    """

    def __init__(
        self,
        num_sentences: Optional[int] = None,
        max_features: Optional[int] = None,
    ) -> None:
        """
        Initialize the ExtractiveSummarizer.

        Args:
            num_sentences: Number of sentences to extract. Defaults to config value.
            max_features: Max TF-IDF features. Defaults to config value.
        """
        self.num_sentences = num_sentences or config.NUM_EXTRACTIVE_SENTENCES
        self.max_features = max_features or config.TFIDF_MAX_FEATURES
        self.preprocessor = TextPreprocessor()

        logger.info(
            "ExtractiveSummarizer initialized — num_sentences=%d, max_features=%d",
            self.num_sentences,
            self.max_features,
        )

    def _get_sentences(self, text: str) -> List[str]:
        """
        Extract clean sentences from raw text using the preprocessor.

        Args:
            text: Raw input text string.

        Returns:
            List of cleaned sentence strings.
        """
        sentences = self.preprocessor.preprocess_for_extractive(text)
        return sentences

    def _build_tfidf_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build a TF-IDF matrix from a list of sentences.

        Args:
            sentences: List of sentence strings.

        Returns:
            TF-IDF matrix as a numpy array of shape (n_sentences, n_features).

        Raises:
            ValueError: If no valid TF-IDF features can be extracted.
        """
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=None,  # Stopwords already handled in preprocessing
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            logger.debug("TF-IDF matrix shape: %s", tfidf_matrix.shape)
            return tfidf_matrix
        except ValueError as e:
            logger.error("Failed to build TF-IDF matrix: %s", e)
            raise ValueError(
                f"Could not build TF-IDF matrix from {len(sentences)} sentences: {e}"
            )

    def _build_similarity_graph(self, tfidf_matrix: np.ndarray) -> nx.Graph:
        """
        Build a cosine similarity graph from the TF-IDF matrix.

        Each node represents a sentence. Edge weights are cosine similarities
        between sentence pairs.

        Args:
            tfidf_matrix: TF-IDF matrix of shape (n_sentences, n_features).

        Returns:
            NetworkX graph with similarity-weighted edges.
        """
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Create graph from similarity matrix
        graph = nx.from_numpy_array(similarity_matrix)

        # Remove self-loops
        graph.remove_edges_from(nx.selfloop_edges(graph))

        logger.debug(
            "Similarity graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def _rank_sentences(self, graph: nx.Graph) -> List[float]:
        """
        Apply PageRank algorithm to score sentences by importance.

        Args:
            graph: NetworkX graph with similarity-weighted edges.

        Returns:
            List of PageRank scores, one per sentence (indexed by node).
        """
        try:
            scores = nx.pagerank(graph, weight="weight")
            logger.debug("PageRank scores: %s", scores)
            return scores
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge. Using uniform scores.")
            n = graph.number_of_nodes()
            return {i: 1.0 / n for i in range(n)}

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """
        Generate an extractive summary of the input text.

        Args:
            text: Raw input text string.
            num_sentences: Number of sentences to extract. Overrides instance default.

        Returns:
            Summary string containing the top-ranked sentences in
            their original document order.
        """
        n_sentences = num_sentences or self.num_sentences

        # Handle empty input
        if not text or not text.strip():
            logger.warning("Empty input text. Returning empty summary.")
            return ""

        # Step 1: Get clean sentences
        sentences = self._get_sentences(text)

        if not sentences:
            logger.warning("No sentences extracted from text. Returning empty summary.")
            return ""

        # Handle edge case: fewer sentences than requested
        if len(sentences) <= n_sentences:
            logger.info(
                "Text has %d sentences (requested %d). Returning all sentences.",
                len(sentences),
                n_sentences,
            )
            return " ".join(sentences)

        # Step 2: Build TF-IDF matrix
        try:
            tfidf_matrix = self._build_tfidf_matrix(sentences)
        except ValueError:
            logger.warning("TF-IDF failed. Returning first %d sentences.", n_sentences)
            return " ".join(sentences[:n_sentences])

        # Step 3: Build similarity graph
        graph = self._build_similarity_graph(tfidf_matrix)

        # Step 4: Rank sentences using PageRank
        scores = self._rank_sentences(graph)

        # Step 5: Select top-N sentences
        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        top_indices = sorted(ranked_indices[:n_sentences])  # Original order

        # Step 6: Construct summary from selected sentences
        summary_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(summary_sentences)

        logger.info(
            "Extractive summary generated: %d sentences selected from %d total",
            len(summary_sentences),
            len(sentences),
        )
        return summary

    def batch_summarize(
        self, texts: List[str], num_sentences: Optional[int] = None
    ) -> List[str]:
        """
        Generate extractive summaries for a batch of texts.

        Args:
            texts: List of raw input text strings.
            num_sentences: Number of sentences to extract per text.

        Returns:
            List of summary strings.
        """
        from tqdm import tqdm

        summaries = []
        for text in tqdm(texts, desc="Extractive Summarization", unit="doc"):
            summary = self.summarize(text, num_sentences=num_sentences)
            summaries.append(summary)

        logger.info(
            "Batch extractive summarization completed: %d documents", len(summaries)
        )
        return summaries


if __name__ == "__main__":
    """Test the ExtractiveSummarizer when run directly."""
    from src.utils import setup_logging

    setup_logging("DEBUG")

    summarizer = ExtractiveSummarizer(num_sentences=2)

    sample_text = (
        "Penelitian ini bertujuan untuk menganalisis pengaruh media sosial "
        "terhadap prestasi akademik mahasiswa di Universitas Indonesia. "
        "Metode yang digunakan adalah survei kuantitatif dengan sampel 200 "
        "mahasiswa dari berbagai fakultas di lingkungan kampus. "
        "Hasil penelitian menunjukkan bahwa terdapat korelasi negatif antara "
        "durasi penggunaan media sosial dengan indeks prestasi kumulatif mahasiswa. "
        "Mahasiswa yang menggunakan media sosial lebih dari 4 jam per hari "
        "memiliki IPK rata-rata lebih rendah dibandingkan mahasiswa yang "
        "menggunakan kurang dari 2 jam per hari. "
        "Penelitian ini juga menemukan bahwa jenis platform media sosial "
        "yang digunakan mempengaruhi dampak terhadap prestasi akademik. "
        "Kesimpulan dari penelitian ini adalah perlu adanya manajemen waktu "
        "yang baik dalam penggunaan media sosial oleh mahasiswa."
    )

    print("=" * 60)
    print("ORIGINAL TEXT:")
    print(sample_text)
    print("\n" + "=" * 60)
    print("EXTRACTIVE SUMMARY (2 sentences):")
    summary = summarizer.summarize(sample_text)
    print(summary)

    # Test edge case: short text
    print("\n" + "=" * 60)
    print("EDGE CASE — Short text:")
    short_text = "Ini adalah kalimat pendek."
    print(f"Input: {short_text}")
    print(f"Summary: {summarizer.summarize(short_text)}")

    # Test edge case: empty text
    print("\n" + "=" * 60)
    print("EDGE CASE — Empty text:")
    print(f"Summary: '{summarizer.summarize('')}'")

    print("\nExtractiveSummarizer module test completed successfully.")
