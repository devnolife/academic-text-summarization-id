# -*- coding: utf-8 -*-
"""
Main entry point for the NLP Summarization Pipeline.

Orchestrates the full end-to-end pipeline: data loading, preprocessing,
extractive and abstractive summarization, evaluation, and report generation.

Usage:
    python main.py                          # Run full pipeline with both methods
    python main.py --mode train             # Train abstractive model only
    python main.py --mode evaluate          # Evaluate only (requires trained model)
    python main.py --mode full              # Full pipeline (default)
    python main.py --model extractive       # Run extractive method only
    python main.py --model abstractive      # Run abstractive method only
    python main.py --model both             # Run both methods (default)
    python main.py --data path/to/data.csv  # Override dataset path
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import config
from src.utils import setup_logging, set_seed, save_json, log_message
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.extractive_model import ExtractiveSummarizer
from src.abstractive_model import AbstractiveSummarizer
from src.evaluator import Evaluator

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="NLP Text Summarization Pipeline — "
        "Extractive vs Abstractive on Indonesian Academic Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="Pipeline mode: 'train' (fine-tune abstractive model), "
        "'evaluate' (evaluate only), 'full' (train + evaluate). "
        "Default: full",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["extractive", "abstractive", "both"],
        default="both",
        help="Summarization method to use: 'extractive', 'abstractive', or 'both'. "
        "Default: both",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset file (CSV or JSON). Overrides config.DATASET_PATH.",
    )

    return parser.parse_args()


def run_extractive(
    test_texts: List[str],
) -> List[str]:
    """
    Run extractive summarization on test documents.

    Args:
        test_texts: List of raw test document texts.

    Returns:
        List of extractive summary strings.
    """
    logger.info("=" * 50)
    logger.info("  EXTRACTIVE SUMMARIZATION")
    logger.info("=" * 50)

    summarizer = ExtractiveSummarizer()
    summaries = summarizer.batch_summarize(test_texts)

    logger.info(
        "Extractive summarization completed: %d summaries generated.", len(summaries)
    )
    return summaries


def run_abstractive(
    train_texts: List[str],
    train_summaries: List[str],
    val_texts: List[str],
    val_summaries: List[str],
    test_texts: List[str],
    mode: str = "full",
) -> List[str]:
    """
    Run abstractive summarization with optional fine-tuning.

    Args:
        train_texts: Training document texts.
        train_summaries: Training summary targets.
        val_texts: Validation document texts.
        val_summaries: Validation summary targets.
        test_texts: Test document texts to summarize.
        mode: Pipeline mode ('train', 'evaluate', 'full').

    Returns:
        List of abstractive summary strings.
    """
    logger.info("=" * 50)
    logger.info("  ABSTRACTIVE SUMMARIZATION")
    logger.info("=" * 50)

    summarizer = AbstractiveSummarizer()

    if mode in ("train", "full"):
        # Load checkpoint if exists, otherwise fine-tune
        summarizer.load_or_train(
            train_texts=train_texts,
            train_summaries=train_summaries,
            val_texts=val_texts,
            val_summaries=val_summaries,
        )
    elif mode == "evaluate":
        # Evaluate mode — must load existing checkpoint
        checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model")
        loaded = summarizer._load_checkpoint(checkpoint_path)
        if not loaded:
            raise RuntimeError(
                f"No checkpoint found at {checkpoint_path}. "
                "Run with --mode train or --mode full first."
            )

    # Generate summaries on test set
    summaries = summarizer.batch_summarize(test_texts)

    logger.info(
        "Abstractive summarization completed: %d summaries generated.", len(summaries)
    )
    return summaries


def run_evaluation(
    extractive_preds: Optional[List[str]],
    abstractive_preds: Optional[List[str]],
    references: List[str],
) -> None:
    """
    Run ROUGE evaluation and generate comparison reports.

    Args:
        extractive_preds: Extractive summaries (None if not computed).
        abstractive_preds: Abstractive summaries (None if not computed).
        references: Ground truth summaries.
    """
    logger.info("=" * 50)
    logger.info("  EVALUATION")
    logger.info("=" * 50)

    evaluator = Evaluator()

    extractive_scores = None
    abstractive_scores = None

    if extractive_preds is not None:
        extractive_scores = evaluator.evaluate_single_method(
            extractive_preds, references, "Extractive"
        )

    if abstractive_preds is not None:
        abstractive_scores = evaluator.evaluate_single_method(
            abstractive_preds, references, "Abstractive"
        )

    # Print comparison if both methods were run
    if extractive_scores and abstractive_scores:
        evaluator.print_comparison_report(extractive_scores, abstractive_scores)
        evaluator.save_evaluation_report(
            extractive_preds=extractive_preds,
            abstractive_preds=abstractive_preds,
            references=references,
            extractive_scores=extractive_scores,
            abstractive_scores=abstractive_scores,
        )
    elif extractive_scores:
        # Save extractive-only results
        logger.info("Only extractive results available.")
        import pandas as pd

        ext_path = os.path.join(config.SUMMARIES_DIR, "extractive_summaries.csv")
        pd.DataFrame(
            {
                "reference": references,
                "extractive_summary": extractive_preds,
            }
        ).to_csv(ext_path, index=False)
        save_json(
            {"extractive": extractive_scores, "num_documents": len(references)},
            os.path.join(config.RESULTS_DIR, "comparison_summary.json"),
        )
    elif abstractive_scores:
        # Save abstractive-only results
        logger.info("Only abstractive results available.")
        import pandas as pd

        abs_path = os.path.join(config.SUMMARIES_DIR, "abstractive_summaries.csv")
        pd.DataFrame(
            {
                "reference": references,
                "abstractive_summary": abstractive_preds,
            }
        ).to_csv(abs_path, index=False)
        save_json(
            {"abstractive": abstractive_scores, "num_documents": len(references)},
            os.path.join(config.RESULTS_DIR, "comparison_summary.json"),
        )


def main() -> None:
    """
    Main pipeline orchestrator.

    Executes the full NLP summarization pipeline based on CLI arguments:
    1. Load configuration
    2. Set random seed
    3. Load and validate dataset
    4. Run preprocessing
    5. Run extractive summarization (if selected)
    6. Run abstractive summarization (if selected)
    7. Evaluate and compare methods
    8. Save results and print report
    """
    # Parse CLI arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(config.LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("  NLP SUMMARIZATION PIPELINE — STARTING")
    logger.info("  Mode: %s | Model: %s", args.mode, args.model)
    logger.info("=" * 60)

    # Step 1: Set random seed
    set_seed(config.RANDOM_SEED)

    # Step 2: Load and validate dataset
    dataset_path = args.data if args.data else config.DATASET_PATH
    data_loader = DataLoader(dataset_path=dataset_path)

    try:
        train_df, val_df, test_df = data_loader.load_and_prepare()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Dataset loading failed: %s", e)
        sys.exit(1)

    # Extract text and summary columns
    train_texts = train_df[config.TEXT_COLUMN].tolist()
    train_summaries_text = train_df[config.SUMMARY_COLUMN].tolist()
    val_texts = val_df[config.TEXT_COLUMN].tolist()
    val_summaries_text = val_df[config.SUMMARY_COLUMN].tolist()
    test_texts = test_df[config.TEXT_COLUMN].tolist()
    test_references = test_df[config.SUMMARY_COLUMN].tolist()

    logger.info("Test set: %d documents for evaluation.", len(test_texts))

    # Step 3: Run summarization methods
    extractive_preds = None
    abstractive_preds = None

    if args.model in ("extractive", "both"):
        try:
            extractive_preds = run_extractive(test_texts)
        except Exception as e:
            logger.error("Extractive summarization failed: %s", e)
            if args.model == "extractive":
                sys.exit(1)

    if args.model in ("abstractive", "both"):
        try:
            abstractive_preds = run_abstractive(
                train_texts=train_texts,
                train_summaries=train_summaries_text,
                val_texts=val_texts,
                val_summaries=val_summaries_text,
                test_texts=test_texts,
                mode=args.mode,
            )
        except Exception as e:
            logger.error("Abstractive summarization failed: %s", e)
            if args.model == "abstractive":
                sys.exit(1)

    # Step 4: Evaluate
    if args.mode in ("evaluate", "full"):
        if extractive_preds is not None or abstractive_preds is not None:
            run_evaluation(extractive_preds, abstractive_preds, test_references)
        else:
            logger.warning("No summaries generated. Skipping evaluation.")

    logger.info("=" * 60)
    logger.info("  NLP SUMMARIZATION PIPELINE — COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
