# Agent Code Prompt — Automatic Text Summarization NLP Pipeline
**Project:** Automatic Text Summarization on Indonesian Academic Documents  
**Environment:** Python scripts (.py files)  
**Task:** Build a full end-to-end NLP pipeline comparing Extractive vs Abstractive summarization

---

## 🎯 Project Overview

You are an expert Python developer and NLP engineer. Your task is to build a complete, modular, and well-documented NLP pipeline for **Automatic Text Summarization** on Indonesian academic documents (skripsi/journal papers).

The system must:
- Accept Indonesian academic documents as input
- Preprocess the text through a complete NLP pipeline
- Implement **two summarization methods** for comparison:
  - **Method 1:** Extractive summarization using TextRank + TF-IDF
  - **Method 2:** Abstractive summarization using IndoBERT / mT5
- Evaluate both methods using ROUGE metrics
- Output a structured comparison report

---

## 📁 Project Structure

Organize all code into the following modular structure:

```
nlp_summarization/
│
├── data/
│   ├── raw/                  # Raw dataset files (CSV/JSON)
│   └── processed/            # Cleaned and preprocessed data
│
├── src/
│   ├── data_loader.py        # Dataset loading and validation
│   ├── preprocessor.py       # Full NLP preprocessing pipeline
│   ├── extractive_model.py   # TextRank + TF-IDF summarization
│   ├── abstractive_model.py  # IndoBERT / mT5 summarization
│   ├── evaluator.py          # ROUGE evaluation
│   └── utils.py              # Helper functions
│
├── output/
│   ├── summaries/            # Generated summaries
│   └── results/              # Evaluation results and reports
│
├── main.py                   # Main entry point to run full pipeline
├── config.py                 # All configurations and hyperparameters
└── requirements.txt          # All required dependencies
```

---

## ⚙️ Configuration (`config.py`)

Define all configurable parameters in a single `config.py` file. This must include:

- Path to dataset file (CSV/JSON)
- Column names for `full_text` and `summary` (ground truth)
- Maximum input text length
- Maximum summary length
- Model name for abstractive method (e.g., `"cahya/bert-base-indonesian-522M"` or `"google/mt5-small"`)
- Number of sentences for extractive summary
- ROUGE metric types to compute (`rouge1`, `rouge2`, `rougeL`)
- Output directory paths
- Random seed for reproducibility
- Batch size and training epochs (for abstractive model)

---

## 📦 Dataset Loader (`data_loader.py`)

Build a dataset loader that handles the following responsibilities:

- Load dataset from CSV or JSON format
- Validate that required columns (`full_text`, `summary`) exist
- Check and report the total number of documents loaded
- Filter out rows where `full_text` or `summary` is empty or null
- Optionally truncate `full_text` to a maximum character length defined in config
- Split dataset into train, validation, and test sets (ratio: 80/10/10)
- Return clean pandas DataFrames for each split
- Log a summary of dataset statistics (total docs, avg text length, avg summary length)

---

## 🧹 Preprocessing Pipeline (`preprocessor.py`)

Build a preprocessing class with the following sequential steps. Each step must be a separate method and must be callable independently:

**Step 1 — Case Folding**
- Convert all text to lowercase

**Step 2 — Cleaning**
- Remove URLs, email addresses, HTML tags
- Remove special characters and extra whitespace
- Remove non-alphabetic tokens that do not contribute to meaning
- Preserve sentence-ending punctuation (`.`, `?`, `!`) for sentence tokenization

**Step 3 — Sentence Tokenization**
- Split text into a list of sentences
- Use NLTK `sent_tokenize` with Indonesian language support

**Step 4 — Word Tokenization**
- Tokenize each sentence into words

**Step 5 — Stopword Removal**
- Remove Indonesian stopwords using NLTK Indonesian stopword list
- Allow custom stopword list to be passed as parameter

**Step 6 — Stemming**
- Apply Indonesian stemming using `PySastrawi` `StemmerFactory`

**Step 7 — Full Pipeline Method**
- Provide a single `preprocess(text)` method that runs all steps in order
- Return both the fully preprocessed text and the sentence-tokenized version (needed for extractive model)

---

## 📄 Extractive Summarization (`extractive_model.py`)

Build a TextRank-based extractive summarizer with TF-IDF sentence scoring:

**Requirements:**
- Accept a raw text string and number of sentences as input
- Internally run preprocessing to get clean sentences
- Build a TF-IDF matrix from the sentences using `sklearn TfidfVectorizer`
- Compute cosine similarity between all sentence pairs to build a similarity graph
- Apply PageRank algorithm (via `networkx`) to score each sentence
- Select the top-N highest-scored sentences
- Return the selected sentences **in their original document order** (not ranked order)
- Handle edge cases: text shorter than requested summary length, empty input

---

## 🤖 Abstractive Summarization (`abstractive_model.py`)

Build a fine-tuned transformer-based abstractive summarizer:

**Requirements:**
- Support two model options configurable via `config.py`: IndoBERT-based seq2seq or mT5
- Load the pre-trained model and tokenizer from HuggingFace
- Implement a `fine_tune(train_data, val_data)` method to fine-tune on the dataset
  - Use the `full_text` column as input and `summary` column as target
  - Apply proper tokenization with padding and truncation
  - Use `Seq2SeqTrainer` from HuggingFace for training
  - Save the fine-tuned model checkpoint to the output directory
- Implement a `summarize(text)` method for inference
  - Tokenize input text
  - Generate summary using beam search (num_beams=4)
  - Decode and return the generated summary string
- Implement a `batch_summarize(texts)` method for processing multiple documents

---

## 📊 Evaluator (`evaluator.py`)

Build an evaluation module that:

- Accepts two lists: `predictions` (generated summaries) and `references` (ground truth abstracts)
- Computes ROUGE-1, ROUGE-2, and ROUGE-L scores using the `rouge-score` library
- Returns scores as a structured dictionary with `precision`, `recall`, and `f-measure` for each metric
- Generates a comparison table between Extractive and Abstractive results
- Saves the full evaluation report as a CSV file in the `output/results/` directory
- Prints a formatted summary table to the console showing side-by-side comparison
- Includes a method to evaluate on a per-document basis and save individual scores

---

## 🛠️ Utilities (`utils.py`)

Provide the following utility functions:

- `save_json(data, path)` — save any dict/list to JSON file
- `load_json(path)` — load JSON file
- `save_summaries(summaries, path)` — save list of summaries to text file
- `log_message(message, level)` — simple logging wrapper (INFO, WARNING, ERROR)
- `set_seed(seed)` — set random seed for Python, NumPy, and PyTorch for reproducibility
- `compute_text_stats(texts)` — compute average, min, max length of a list of texts
- `truncate_text(text, max_chars)` — safely truncate text to a maximum character limit

---

## 🚀 Main Entry Point (`main.py`)

The `main.py` file must orchestrate the full pipeline in this order:

1. Load configuration from `config.py`
2. Set random seed via `utils.set_seed()`
3. Load and validate dataset via `data_loader.py`
4. Run preprocessing on all documents
5. Run **Extractive summarization** on the test set
6. Run **Abstractive summarization** on the test set (load fine-tuned model if checkpoint exists, otherwise fine-tune first)
7. Evaluate both methods using ROUGE via `evaluator.py`
8. Save all summaries and evaluation results to `output/`
9. Print final comparison report to console

Support command-line arguments:
- `--mode` : `train`, `evaluate`, or `full` (default: `full`)
- `--model` : `extractive`, `abstractive`, or `both` (default: `both`)
- `--data` : path to dataset file (overrides config)

---

## 📋 Requirements (`requirements.txt`)

Include all necessary dependencies with pinned or minimum versions:

- `pandas` — data handling
- `numpy` — numerical operations
- `nltk` — tokenization and stopwords
- `PySastrawi` — Indonesian stemming
- `scikit-learn` — TF-IDF and cosine similarity
- `networkx` — PageRank for TextRank
- `transformers` — HuggingFace models (IndoBERT, mT5)
- `torch` — PyTorch backend
- `datasets` — HuggingFace datasets utility
- `rouge-score` — ROUGE evaluation
- `tqdm` — progress bars
- `argparse` — CLI argument parsing

---

## ✅ Code Quality Standards

Enforce the following standards across all files:

- Every class and function must have a **docstring** explaining purpose, parameters, and return values
- Use **type hints** for all function signatures
- Use **logging** module (not `print`) for all runtime messages, except final report output
- Handle all exceptions with meaningful error messages
- Avoid hardcoded values — all parameters must come from `config.py`
- Each `.py` file must be independently runnable for testing its own module
- Follow **PEP 8** coding style

---

## 📝 Output Format

At the end of a full pipeline run, the agent must produce:

**Console Output:**
```
========================================
  TEXT SUMMARIZATION EVALUATION REPORT
========================================
Method          ROUGE-1   ROUGE-2   ROUGE-L
-----------------------------------------
Extractive      0.XXXX    0.XXXX    0.XXXX
Abstractive     0.XXXX    0.XXXX    0.XXXX
-----------------------------------------
Best Method:    [Extractive / Abstractive]
========================================
```

**Saved Files:**
- `output/summaries/extractive_summaries.csv` — all extractive results
- `output/summaries/abstractive_summaries.csv` — all abstractive results
- `output/results/evaluation_report.csv` — full ROUGE scores per document
- `output/results/comparison_summary.json` — final aggregated comparison

---

## ⚠️ Important Constraints

- **Do NOT** mix preprocessing logic inside model files — keep preprocessing strictly in `preprocessor.py`
- **Do NOT** hardcode file paths — always use `config.py` or CLI arguments
- **Do NOT** load the full dataset into GPU memory at once — use batching
- **Always** check if a fine-tuned model checkpoint already exists before re-training
- **Always** validate dataset columns before starting the pipeline
- The extractive model must **never** require GPU — it must run on CPU only
- The abstractive model must **gracefully fallback** to CPU if GPU is not available