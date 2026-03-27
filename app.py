# -*- coding: utf-8 -*-
"""
Flask Web Application for NLP Summarization Pipeline.

Provides a web interface to upload documents, run preprocessing,
extractive/abstractive summarization, and ROUGE evaluation.
"""

import os
import sys
import json
import logging
import traceback
import tempfile
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import config
from src.utils import set_seed
from src.preprocessor import TextPreprocessor
from src.extractive_model import ExtractiveSummarizer
from src.evaluator import Evaluator

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()

ALLOWED_EXTENSIONS = {"csv", "json"}

set_seed(config.RANDOM_SEED)

preprocessor = TextPreprocessor()
extractive_summarizer = ExtractiveSummarizer()
evaluator = Evaluator()

# Lazy-load abstractive summarizer (heavy model)
_abstractive_summarizer = None


def get_abstractive_summarizer():
    """Lazy-load the abstractive summarizer to avoid slow startup."""
    global _abstractive_summarizer
    if _abstractive_summarizer is None:
        try:
            from src.abstractive_model import AbstractiveSummarizer
            _abstractive_summarizer = AbstractiveSummarizer()
            checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model")
            if os.path.exists(checkpoint_path):
                _abstractive_summarizer._load_checkpoint(checkpoint_path)
            else:
                _abstractive_summarizer._load_model()
        except Exception as e:
            logger.warning("Could not load abstractive model: %s", e)
            return None
    return _abstractive_summarizer


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload and validate a CSV/JSON dataset file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Use CSV or JSON."}), 400

        ext = file.filename.rsplit(".", 1)[1].lower()
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"upload.{ext}")
        file.save(filepath)

        if ext == "csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_json(filepath)

        # Validate columns
        text_col = config.TEXT_COLUMN
        summary_col = config.SUMMARY_COLUMN

        if text_col not in df.columns:
            return jsonify({"error": f"Column '{text_col}' not found. Available: {list(df.columns)}"}), 400
        if summary_col not in df.columns:
            return jsonify({"error": f"Column '{summary_col}' not found. Available: {list(df.columns)}"}), 400

        # Clean
        df = df.dropna(subset=[text_col, summary_col])
        df = df[df[text_col].str.strip().astype(bool)]
        df = df[df[summary_col].str.strip().astype(bool)]

        if len(df) == 0:
            return jsonify({"error": "Dataset is empty after cleaning"}), 400

        texts = df[text_col].tolist()
        summaries = df[summary_col].tolist()

        avg_text_len = sum(len(t) for t in texts) / len(texts)
        avg_summary_len = sum(len(s) for s in summaries) / len(summaries)

        return jsonify({
            "success": True,
            "num_documents": len(texts),
            "texts": texts,
            "summaries": summaries,
            "preview": [
                {"text": t[:200] + "..." if len(t) > 200 else t,
                 "summary": s[:150] + "..." if len(s) > 150 else s}
                for t, s in zip(texts[:5], summaries[:5])
            ],
            "stats": {
                "avg_text_length": round(avg_text_len),
                "avg_summary_length": round(avg_summary_len),
            }
        })

    except Exception as e:
        logger.error("Upload error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    """Run preprocessing step-by-step and return intermediate results."""
    try:
        data = request.get_json()
        texts = data.get("texts", [])

        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        results = []
        for text in texts:
            original = text

            # Step 1: Case folding
            step1 = preprocessor.case_folding(text)

            # Step 2: Cleaning
            step2 = preprocessor.clean_text(step1)

            # Step 3: Sentence tokenization
            step3 = preprocessor.sentence_tokenize(step2)

            # Step 4: Word tokenization (on full cleaned text)
            step4 = preprocessor.word_tokenize(step2)

            # Step 5: Stopword removal
            step5 = preprocessor.remove_stopwords(step4)

            # Step 6: Stemming
            step6 = preprocessor.stem_tokens(step5)

            results.append({
                "original": original[:500] + ("..." if len(original) > 500 else ""),
                "case_folding": step1[:500] + ("..." if len(step1) > 500 else ""),
                "cleaning": step2[:500] + ("..." if len(step2) > 500 else ""),
                "sentences": step3[:10],
                "word_tokens": step4[:50],
                "after_stopword_removal": step5[:50],
                "after_stemming": step6[:50],
                "num_sentences": len(step3),
                "num_tokens_before": len(step4),
                "num_tokens_after": len(step5),
            })

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error("Preprocess error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Run extractive and abstractive summarization."""
    try:
        data = request.get_json()
        texts = data.get("texts", [])

        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        result = {"success": True}

        # Extractive
        try:
            ext_summaries = extractive_summarizer.batch_summarize(texts)
            result["extractive"] = ext_summaries
        except Exception as e:
            logger.error("Extractive error: %s", e)
            result["extractive"] = None
            result["extractive_error"] = str(e)

        # Abstractive
        try:
            abs_model = get_abstractive_summarizer()
            if abs_model is not None:
                abs_summaries = abs_model.batch_summarize(texts)
                result["abstractive"] = abs_summaries
            else:
                result["abstractive"] = None
                result["abstractive_error"] = (
                    "Model abstractive belum tersedia. "
                    "Jalankan training terlebih dahulu dengan: python main.py --mode train --model abstractive"
                )
        except Exception as e:
            logger.error("Abstractive error: %s", e)
            result["abstractive"] = None
            result["abstractive_error"] = str(e)

        return jsonify(result)

    except Exception as e:
        logger.error("Summarize error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Run ROUGE evaluation on predictions vs references."""
    try:
        data = request.get_json()
        references = data.get("references", [])
        extractive_preds = data.get("extractive_preds")
        abstractive_preds = data.get("abstractive_preds")

        if not references:
            return jsonify({"error": "No reference summaries provided"}), 400

        result = {"success": True}

        if extractive_preds:
            ext_scores = evaluator.compute_rouge(extractive_preds, references)
            result["extractive_scores"] = ext_scores

        if abstractive_preds:
            abs_scores = evaluator.compute_rouge(abstractive_preds, references)
            result["abstractive_scores"] = abs_scores

        # Determine best method
        if extractive_preds and abstractive_preds:
            ext_avg = sum(
                ext_scores[m]["fmeasure"] for m in config.ROUGE_METRICS
            ) / len(config.ROUGE_METRICS)
            abs_avg = sum(
                abs_scores[m]["fmeasure"] for m in config.ROUGE_METRICS
            ) / len(config.ROUGE_METRICS)
            result["best_method"] = "Extractive" if ext_avg >= abs_avg else "Abstractive"
            result["extractive_avg_f1"] = round(ext_avg, 4)
            result["abstractive_avg_f1"] = round(abs_avg, 4)
        elif extractive_preds:
            result["best_method"] = "Extractive"
        elif abstractive_preds:
            result["best_method"] = "Abstractive"

        return jsonify(result)

    except Exception as e:
        logger.error("Evaluate error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/process-text", methods=["POST"])
def process_single_text():
    """Process a single text through the full pipeline."""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        reference = data.get("reference", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = {"success": True}

        # Step 1: Preprocessing
        step1 = preprocessor.case_folding(text)
        step2 = preprocessor.clean_text(step1)
        step3 = preprocessor.sentence_tokenize(step2)
        step4 = preprocessor.word_tokenize(step2)
        step5 = preprocessor.remove_stopwords(step4)
        step6 = preprocessor.stem_tokens(step5)

        result["preprocessing"] = {
            "original": text[:500] + ("..." if len(text) > 500 else ""),
            "case_folding": step1[:500] + ("..." if len(step1) > 500 else ""),
            "cleaning": step2[:500] + ("..." if len(step2) > 500 else ""),
            "sentences": step3[:10],
            "word_tokens": step4[:50],
            "after_stopword_removal": step5[:50],
            "after_stemming": step6[:50],
            "num_sentences": len(step3),
            "num_tokens_before": len(step4),
            "num_tokens_after": len(step5),
        }

        # Step 2: Summarization
        try:
            ext_summary = extractive_summarizer.summarize(text)
            result["extractive_summary"] = ext_summary
        except Exception as e:
            result["extractive_summary"] = None
            result["extractive_error"] = str(e)

        try:
            abs_model = get_abstractive_summarizer()
            if abs_model is not None:
                abs_summary = abs_model.summarize(text)
                result["abstractive_summary"] = abs_summary
            else:
                result["abstractive_summary"] = None
                result["abstractive_error"] = (
                    "Model abstractive belum tersedia. "
                    "Jalankan training terlebih dahulu."
                )
        except Exception as e:
            result["abstractive_summary"] = None
            result["abstractive_error"] = str(e)

        # Step 3: Evaluation (only if reference is provided)
        if reference:
            preds_ext = [result.get("extractive_summary", "")] if result.get("extractive_summary") else None
            preds_abs = [result.get("abstractive_summary", "")] if result.get("abstractive_summary") else None
            refs = [reference]

            if preds_ext:
                result["extractive_scores"] = evaluator.compute_rouge(preds_ext, refs)
            if preds_abs:
                result["abstractive_scores"] = evaluator.compute_rouge(preds_abs, refs)

            if preds_ext and preds_abs:
                ext_avg = sum(
                    result["extractive_scores"][m]["fmeasure"] for m in config.ROUGE_METRICS
                ) / len(config.ROUGE_METRICS)
                abs_avg = sum(
                    result["abstractive_scores"][m]["fmeasure"] for m in config.ROUGE_METRICS
                ) / len(config.ROUGE_METRICS)
                result["best_method"] = "Extractive" if ext_avg >= abs_avg else "Abstractive"

        return jsonify(result)

    except Exception as e:
        logger.error("Process text error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting NLP Summarization Web App...")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
