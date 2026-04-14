# -*- coding: utf-8 -*-
"""
Pre-compute all pipeline results and save to JSON.

Runs the full NLP pipeline (preprocessing, extractive, abstractive, evaluation)
on the default dataset and saves results to output/results/*.json.
This allows the web app to serve pre-computed results instantly during presentation.

Usage:
    python precompute.py
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import config
from src.preprocessor import TextPreprocessor
from src.extractive_model import ExtractiveSummarizer
from src.abstractive_model import AbstractiveSummarizer
from src.evaluator import Evaluator
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = config.RESULTS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

set_seed(config.RANDOM_SEED)


def save_json_file(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(path) / 1024
    logger.info("Saved %s (%.1f KB)", path, size_kb)


def load_dataset():
    logger.info("Loading dataset from %s", config.DATASET_PATH)
    df = pd.read_csv(config.DATASET_PATH)
    df = df.dropna(subset=[config.TEXT_COLUMN, config.SUMMARY_COLUMN])
    df = df[df[config.TEXT_COLUMN].str.strip().astype(bool)]
    df = df[df[config.SUMMARY_COLUMN].str.strip().astype(bool)]

    texts = df[config.TEXT_COLUMN].tolist()
    summaries = df[config.SUMMARY_COLUMN].tolist()
    logger.info("Loaded %d documents", len(texts))
    return texts, summaries


def compute_dataset_json(texts, summaries):
    logger.info("=== Generating dataset.json ===")
    avg_text_len = sum(len(str(t)) for t in texts) / len(texts)
    avg_summary_len = sum(len(str(s)) for s in summaries) / len(summaries)

    data = {
        "success": True,
        "num_documents": len(texts),
        "texts": texts,
        "summaries": summaries,
        "preview": [
            {
                "text": str(t)[:200] + ("..." if len(str(t)) > 200 else ""),
                "summary": str(s)[:150] + ("..." if len(str(s)) > 150 else ""),
            }
            for t, s in zip(texts[:5], summaries[:5])
        ],
        "stats": {
            "avg_text_length": round(avg_text_len),
            "avg_summary_length": round(avg_summary_len),
        },
    }
    save_json_file(data, "dataset.json")
    return data


def compute_preprocess_json(texts):
    logger.info("=== Generating preprocess.json ===")
    preprocessor = TextPreprocessor()
    results = []

    for i, text in enumerate(texts):
        logger.info("  Preprocessing doc %d/%d", i + 1, len(texts))
        step1 = preprocessor.case_folding(text)
        step2 = preprocessor.clean_text(step1)
        step3 = preprocessor.sentence_tokenize(step2)
        step4 = preprocessor.word_tokenize(step2)
        step5 = preprocessor.remove_stopwords(step4)
        step6 = preprocessor.stem_tokens(step5)

        results.append({
            "original": text,
            "case_folding": step1,
            "cleaning": step2,
            "sentences": step3,
            "word_tokens": step4,
            "after_stopword_removal": step5,
            "after_stemming": step6,
            "num_sentences": len(step3),
            "num_tokens_before": len(step4),
            "num_tokens_after": len(step5),
            "num_tokens_stemmed": len(step6),
        })

    data = {"success": True, "results": results}
    save_json_file(data, "preprocess.json")
    return data


def compute_summarize_json(texts):
    logger.info("=== Generating summarize.json ===")
    ext_summarizer = ExtractiveSummarizer()
    result = {"success": True}

    # --- Extractive ---
    logger.info("  Running extractive summarization...")
    ext_summaries = []
    ext_details = []

    for i, text in enumerate(texts):
        logger.info("  Extractive doc %d/%d", i + 1, len(texts))
        detail = {}
        sentences = ext_summarizer._get_sentences(text)
        detail["sentences"] = sentences
        detail["num_sentences"] = len(sentences)

        if not sentences:
            ext_summaries.append("")
            ext_details.append(detail)
            continue

        n_target = ext_summarizer.num_sentences
        if len(sentences) <= n_target:
            summary = " ".join(sentences)
            ext_summaries.append(summary)
            detail["summary"] = summary
            detail["note"] = "Jumlah kalimat kurang dari target"
            ext_details.append(detail)
            continue

        try:
            tfidf_matrix = ext_summarizer._build_tfidf_matrix(sentences)
            detail["tfidf_shape"] = [int(tfidf_matrix.shape[0]), int(tfidf_matrix.shape[1])]
            detail["num_features"] = int(tfidf_matrix.shape[1])
        except ValueError:
            summary = " ".join(sentences[:n_target])
            ext_summaries.append(summary)
            detail["tfidf_error"] = "Gagal membangun TF-IDF"
            ext_details.append(detail)
            continue

        sim_matrix = cos_sim(tfidf_matrix)
        n = len(sentences)
        upper_vals = sim_matrix[np.triu_indices(n, k=1)]
        detail["avg_similarity"] = round(float(np.mean(upper_vals)), 4) if len(upper_vals) > 0 else 0
        detail["max_similarity"] = round(float(np.max(upper_vals)), 4) if len(upper_vals) > 0 else 0

        pairs = []
        for i_idx in range(min(n, 20)):
            for j_idx in range(i_idx + 1, min(n, 20)):
                pairs.append({"i": i_idx, "j": j_idx, "sim": round(float(sim_matrix[i_idx][j_idx]), 4)})
        pairs.sort(key=lambda x: x["sim"], reverse=True)
        detail["top_pairs"] = pairs[:8]

        graph = ext_summarizer._build_similarity_graph(tfidf_matrix)
        scores = ext_summarizer._rank_sentences(graph)
        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        top_indices = sorted(ranked_indices[:n_target])

        sentence_scores = []
        for idx in range(len(sentences)):
            sentence_scores.append({
                "index": idx,
                "sentence": sentences[idx][:200] + ("..." if len(sentences[idx]) > 200 else ""),
                "score": round(float(scores[idx]), 6),
                "rank": ranked_indices.index(idx) + 1,
                "selected": idx in top_indices,
            })
        detail["sentence_scores"] = sentence_scores
        detail["selected_indices"] = [int(i) for i in top_indices]
        detail["num_selected"] = len(top_indices)

        summary = " ".join([sentences[i] for i in top_indices])
        ext_summaries.append(summary)
        detail["summary"] = summary
        ext_details.append(detail)

    result["extractive"] = ext_summaries
    result["extractive_details"] = ext_details

    # --- Abstractive ---
    logger.info("  Running abstractive summarization (this may take a few minutes)...")
    abs_model = AbstractiveSummarizer()
    checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model")
    if os.path.exists(checkpoint_path):
        abs_model._load_checkpoint(checkpoint_path)
    else:
        abs_model._load_model()

    start = time.time()
    abs_summaries = abs_model.batch_summarize(texts)
    elapsed = time.time() - start
    logger.info("  Abstractive done in %.1fs (%.1fs/doc)", elapsed, elapsed / len(texts))

    result["abstractive"] = abs_summaries
    result["abstractive_details"] = {
        "model_name": abs_model.model_name,
        "device": str(abs_model.device),
        "max_source_length": abs_model.max_source_length,
        "max_target_length": abs_model.max_target_length,
        "num_beams": abs_model.num_beams,
        "num_parameters": f"{sum(p.numel() for p in abs_model.model.parameters()):,}",
    }

    save_json_file(result, "summarize.json")
    return result


def compute_evaluate_json(references, ext_summaries, abs_summaries):
    logger.info("=== Generating evaluate.json ===")
    ev = Evaluator()
    result = {"success": True}

    ext_scores = ev.compute_rouge(ext_summaries, references)
    result["extractive_scores"] = ext_scores

    abs_scores = ev.compute_rouge(abs_summaries, references)
    result["abstractive_scores"] = abs_scores

    per_document = []
    for i in range(len(references)):
        doc = {"doc_id": i, "reference_preview": references[i][:150]}
        doc["extractive_preview"] = ext_summaries[i][:150]
        sc = ev.scorer.score(references[i], ext_summaries[i])
        doc["ext"] = {
            m: {"p": round(sc[m].precision, 4), "r": round(sc[m].recall, 4), "f": round(sc[m].fmeasure, 4)}
            for m in config.ROUGE_METRICS
        }
        doc["abstractive_preview"] = abs_summaries[i][:150]
        sc = ev.scorer.score(references[i], abs_summaries[i])
        doc["abs"] = {
            m: {"p": round(sc[m].precision, 4), "r": round(sc[m].recall, 4), "f": round(sc[m].fmeasure, 4)}
            for m in config.ROUGE_METRICS
        }
        per_document.append(doc)

    result["per_document"] = per_document

    ext_avg = sum(ext_scores[m]["fmeasure"] for m in config.ROUGE_METRICS) / len(config.ROUGE_METRICS)
    abs_avg = sum(abs_scores[m]["fmeasure"] for m in config.ROUGE_METRICS) / len(config.ROUGE_METRICS)
    result["extractive_avg_f1"] = round(ext_avg, 4)
    result["abstractive_avg_f1"] = round(abs_avg, 4)
    result["best_method"] = "Extractive" if ext_avg >= abs_avg else "Abstractive"

    save_json_file(result, "evaluate.json")
    return result


def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("PRE-COMPUTING ALL PIPELINE RESULTS")
    logger.info("=" * 60)

    texts, summaries = load_dataset()

    compute_dataset_json(texts, summaries)
    compute_preprocess_json(texts)
    summarize_result = compute_summarize_json(texts)
    compute_evaluate_json(
        summaries,
        summarize_result["extractive"],
        summarize_result["abstractive"],
    )

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("ALL DONE in %.1fs", total_elapsed)
    logger.info("JSON files saved to: %s", OUTPUT_DIR)
    logger.info("Files: dataset.json, preprocess.json, summarize.json, evaluate.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
