# -*- coding: utf-8 -*-
"""
Evaluation module for the NLP Summarization Pipeline.

Computes ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) for comparing
generated summaries against ground truth references. Produces
comparison tables and saves evaluation reports.
"""

import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.utils import save_json

logger = logging.getLogger(__name__)


class Evaluator:
    """
    ROUGE-based evaluation module for text summarization.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores with precision,
    recall, and F-measure. Supports per-document and aggregate evaluation.

    Attributes:
        rouge_metrics: List of ROUGE metric types to compute.
        results_dir: Directory path for saving evaluation results.
        summaries_dir: Directory path for saving summaries.
    """

    def __init__(
        self,
        rouge_metrics: Optional[List[str]] = None,
        results_dir: Optional[str] = None,
        summaries_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            rouge_metrics: ROUGE metrics to compute. Defaults to config value.
            results_dir: Output directory for results. Defaults to config value.
            summaries_dir: Output directory for summaries. Defaults to config value.
        """
        self.rouge_metrics = rouge_metrics or config.ROUGE_METRICS
        self.results_dir = results_dir or config.RESULTS_DIR
        self.summaries_dir = summaries_dir or config.SUMMARIES_DIR

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        # Initialize ROUGE scorer
        try:
            from rouge_score import rouge_scorer

            self.scorer = rouge_scorer.RougeScorer(
                self.rouge_metrics, use_stemmer=False
            )
            logger.info("ROUGE scorer initialized with metrics: %s", self.rouge_metrics)
        except ImportError:
            logger.error(
                "rouge-score library not installed. "
                "Install with: pip install rouge-score"
            )
            raise

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate ROUGE scores for a list of predictions and references.

        Args:
            predictions: List of generated summary strings.
            references: List of ground truth summary strings.

        Returns:
            Dictionary mapping metric names to dictionaries with
            'precision', 'recall', and 'fmeasure' values (averaged).

        Raises:
            ValueError: If predictions and references have different lengths.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(references)} references"
            )

        if not predictions:
            logger.warning("Empty prediction list. Returning zero scores.")
            return {
                metric: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
                for metric in self.rouge_metrics
            }

        # Accumulate scores
        all_scores = {
            metric: {"precision": [], "recall": [], "fmeasure": []}
            for metric in self.rouge_metrics
        }

        for pred, ref in zip(predictions, references):
            # Handle empty strings
            pred = pred if pred and pred.strip() else " "
            ref = ref if ref and ref.strip() else " "

            scores = self.scorer.score(ref, pred)

            for metric in self.rouge_metrics:
                all_scores[metric]["precision"].append(scores[metric].precision)
                all_scores[metric]["recall"].append(scores[metric].recall)
                all_scores[metric]["fmeasure"].append(scores[metric].fmeasure)

        # Compute averages
        avg_scores = {}
        for metric in self.rouge_metrics:
            avg_scores[metric] = {
                "precision": sum(all_scores[metric]["precision"]) / len(predictions),
                "recall": sum(all_scores[metric]["recall"]) / len(predictions),
                "fmeasure": sum(all_scores[metric]["fmeasure"]) / len(predictions),
            }

        logger.info("ROUGE scores computed for %d document pairs.", len(predictions))
        return avg_scores

    def compute_per_document(
        self, predictions: List[str], references: List[str]
    ) -> pd.DataFrame:
        """
        Compute ROUGE scores for each document individually.

        Args:
            predictions: List of generated summary strings.
            references: List of ground truth summary strings.

        Returns:
            DataFrame with per-document ROUGE scores.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(references)} references"
            )

        records = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred = pred if pred and pred.strip() else " "
            ref = ref if ref and ref.strip() else " "

            scores = self.scorer.score(ref, pred)

            record = {"doc_id": i}
            for metric in self.rouge_metrics:
                record[f"{metric}_precision"] = scores[metric].precision
                record[f"{metric}_recall"] = scores[metric].recall
                record[f"{metric}_fmeasure"] = scores[metric].fmeasure

            records.append(record)

        df = pd.DataFrame(records)
        logger.info("Per-document ROUGE scores computed for %d documents.", len(df))
        return df

    def generate_comparison_table(
        self,
        extractive_scores: Dict[str, Dict[str, float]],
        abstractive_scores: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Generate a comparison table between Extractive and Abstractive results.

        Args:
            extractive_scores: Aggregate ROUGE scores for extractive method.
            abstractive_scores: Aggregate ROUGE scores for abstractive method.

        Returns:
            DataFrame containing the comparison table.
        """
        rows = []
        for metric in self.rouge_metrics:
            rows.append(
                {
                    "Metric": metric.upper(),
                    "Extractive_Precision": extractive_scores[metric]["precision"],
                    "Extractive_Recall": extractive_scores[metric]["recall"],
                    "Extractive_F1": extractive_scores[metric]["fmeasure"],
                    "Abstractive_Precision": abstractive_scores[metric]["precision"],
                    "Abstractive_Recall": abstractive_scores[metric]["recall"],
                    "Abstractive_F1": abstractive_scores[metric]["fmeasure"],
                }
            )

        df = pd.DataFrame(rows)
        return df

    def print_comparison_report(
        self,
        extractive_scores: Dict[str, Dict[str, float]],
        abstractive_scores: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Print a formatted comparison report to the console.

        Args:
            extractive_scores: Aggregate ROUGE scores for extractive method.
            abstractive_scores: Aggregate ROUGE scores for abstractive method.
        """
        print("\n" + "=" * 48)
        print("  TEXT SUMMARIZATION EVALUATION REPORT")
        print("=" * 48)
        print(f"{'Method':<16}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}")
        print("-" * 48)

        # Get F-measure scores for display
        ext_r1 = extractive_scores.get("rouge1", {}).get("fmeasure", 0.0)
        ext_r2 = extractive_scores.get("rouge2", {}).get("fmeasure", 0.0)
        ext_rl = extractive_scores.get("rougeL", {}).get("fmeasure", 0.0)

        abs_r1 = abstractive_scores.get("rouge1", {}).get("fmeasure", 0.0)
        abs_r2 = abstractive_scores.get("rouge2", {}).get("fmeasure", 0.0)
        abs_rl = abstractive_scores.get("rougeL", {}).get("fmeasure", 0.0)

        print(f"{'Extractive':<16}{ext_r1:>10.4f}{ext_r2:>10.4f}{ext_rl:>10.4f}")
        print(f"{'Abstractive':<16}{abs_r1:>10.4f}{abs_r2:>10.4f}{abs_rl:>10.4f}")
        print("-" * 48)

        # Determine best method by average ROUGE F-measure
        ext_avg = (ext_r1 + ext_r2 + ext_rl) / 3
        abs_avg = (abs_r1 + abs_r2 + abs_rl) / 3
        best = "Extractive" if ext_avg >= abs_avg else "Abstractive"

        print(f"{'Best Method:':<16}{best:>30}")
        print("=" * 48 + "\n")

    def save_evaluation_report(
        self,
        extractive_preds: List[str],
        abstractive_preds: List[str],
        references: List[str],
        extractive_scores: Dict[str, Dict[str, float]],
        abstractive_scores: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Save all evaluation results to files.

        Saves:
        - Per-document evaluation report as CSV
        - Extractive summaries as CSV
        - Abstractive summaries as CSV
        - Comparison summary as JSON

        Args:
            extractive_preds: List of extractive summaries.
            abstractive_preds: List of abstractive summaries.
            references: List of ground truth summaries.
            extractive_scores: Aggregate ROUGE scores for extractive method.
            abstractive_scores: Aggregate ROUGE scores for abstractive method.
        """
        # Save extractive summaries
        ext_summary_path = os.path.join(self.summaries_dir, "extractive_summaries.csv")
        pd.DataFrame(
            {
                "reference": references,
                "extractive_summary": extractive_preds,
            }
        ).to_csv(ext_summary_path, index=False)
        logger.info("Extractive summaries saved to: %s", ext_summary_path)

        # Save abstractive summaries
        abs_summary_path = os.path.join(self.summaries_dir, "abstractive_summaries.csv")
        pd.DataFrame(
            {
                "reference": references,
                "abstractive_summary": abstractive_preds,
            }
        ).to_csv(abs_summary_path, index=False)
        logger.info("Abstractive summaries saved to: %s", abs_summary_path)

        # Save per-document evaluation report
        ext_per_doc = self.compute_per_document(extractive_preds, references)
        ext_per_doc.columns = [
            f"extractive_{col}" if col != "doc_id" else col
            for col in ext_per_doc.columns
        ]
        abs_per_doc = self.compute_per_document(abstractive_preds, references)
        abs_per_doc.columns = [
            f"abstractive_{col}" if col != "doc_id" else col
            for col in abs_per_doc.columns
        ]

        eval_report = pd.merge(ext_per_doc, abs_per_doc, on="doc_id")
        eval_report_path = os.path.join(self.results_dir, "evaluation_report.csv")
        eval_report.to_csv(eval_report_path, index=False)
        logger.info("Evaluation report saved to: %s", eval_report_path)

        # Save comparison summary as JSON
        comparison = {
            "extractive": extractive_scores,
            "abstractive": abstractive_scores,
            "best_method": (
                "Extractive"
                if sum(extractive_scores[m]["fmeasure"] for m in self.rouge_metrics)
                >= sum(abstractive_scores[m]["fmeasure"] for m in self.rouge_metrics)
                else "Abstractive"
            ),
            "num_documents": len(references),
        }
        comparison_path = os.path.join(self.results_dir, "comparison_summary.json")
        save_json(comparison, comparison_path)
        logger.info("Comparison summary saved to: %s", comparison_path)

    def evaluate_single_method(
        self, predictions: List[str], references: List[str], method_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single summarization method and log results.

        Args:
            predictions: List of generated summaries.
            references: List of ground truth summaries.
            method_name: Name of the method (for logging).

        Returns:
            Dictionary of aggregate ROUGE scores.
        """
        scores = self.compute_rouge(predictions, references)

        logger.info("=" * 40)
        logger.info("  %s Evaluation Results", method_name)
        logger.info("=" * 40)
        for metric in self.rouge_metrics:
            logger.info(
                "  %s — P: %.4f, R: %.4f, F1: %.4f",
                metric.upper(),
                scores[metric]["precision"],
                scores[metric]["recall"],
                scores[metric]["fmeasure"],
            )
        logger.info("=" * 40)

        return scores


if __name__ == "__main__":
    """Test the Evaluator when run directly."""
    from src.utils import setup_logging

    setup_logging("DEBUG")

    evaluator = Evaluator()

    # Sample predictions and references
    predictions = [
        "Penelitian menunjukkan korelasi negatif antara media sosial dan prestasi akademik.",
        "Implementasi machine learning untuk klasifikasi sentimen ulasan produk Indonesia.",
        "Kajian efektivitas pembelajaran daring terhadap motivasi belajar mahasiswa.",
    ]
    references = [
        "Penelitian ini menunjukkan korelasi negatif antara penggunaan media sosial dan prestasi akademik mahasiswa.",
        "Implementasi machine learning untuk klasifikasi sentimen ulasan produk Indonesia menggunakan tiga model.",
        "Kajian efektivitas pembelajaran daring terhadap motivasi belajar mahasiswa Teknik Informatika.",
    ]

    # Test aggregate ROUGE
    print("=" * 60)
    print("AGGREGATE ROUGE SCORES:")
    scores = evaluator.compute_rouge(predictions, references)
    for metric, values in scores.items():
        print(
            f"  {metric}: P={values['precision']:.4f}, R={values['recall']:.4f}, F1={values['fmeasure']:.4f}"
        )

    # Test per-document ROUGE
    print("\n" + "=" * 60)
    print("PER-DOCUMENT ROUGE SCORES:")
    per_doc = evaluator.compute_per_document(predictions, references)
    print(per_doc.to_string(index=False))

    # Test comparison report
    print("\n" + "=" * 60)
    print("COMPARISON REPORT:")
    # Simulate abstractive scores (slightly different)
    abs_scores = evaluator.compute_rouge(
        [p[: len(p) // 2] for p in predictions],  # Simulated worse summaries
        references,
    )
    evaluator.print_comparison_report(scores, abs_scores)

    print("Evaluator module test completed successfully.")
