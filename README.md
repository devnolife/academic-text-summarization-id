# Automatic Text Summarization — Indonesian Academic Documents

Pipeline NLP end-to-end untuk **Automatic Text Summarization** pada dokumen akademik berbahasa Indonesia (skripsi/jurnal). Membandingkan dua pendekatan: **Extractive** (TextRank + TF-IDF) vs **Abstractive** (mT5 / IndoBERT).

> Dibuat oleh **devnolife**

---

## Struktur Project

```
.
├── config.py                # Konfigurasi dan hyperparameter
├── main.py                  # Entry point utama
├── requirements.txt         # Dependencies
├── data/
│   ├── raw/                 # Dataset mentah (CSV/JSON)
│   └── processed/           # Data hasil preprocessing
├── output/
│   ├── summaries/           # Hasil ringkasan
│   ├── results/             # Laporan evaluasi
│   └── checkpoints/         # Model checkpoint
└── src/
    ├── utils.py             # Fungsi helper
    ├── data_loader.py       # Loading & validasi dataset
    ├── preprocessor.py      # Pipeline preprocessing NLP
    ├── extractive_model.py  # Summarization TextRank + TF-IDF
    ├── abstractive_model.py # Summarization mT5 / IndoBERT
    └── evaluator.py         # Evaluasi ROUGE
```

---

## Fitur Utama

| Komponen | Deskripsi |
|---|---|
| **Data Loader** | Load CSV/JSON, validasi kolom, cleaning, split 80/10/10 |
| **Preprocessor** | Case folding, cleaning (URL/HTML/email), tokenisasi kalimat & kata, stopword removal (Indonesian), stemming (PySastrawi) |
| **Extractive** | TF-IDF vectorization, cosine similarity graph, PageRank scoring, output urutan asli dokumen. CPU-only |
| **Abstractive** | Fine-tune `google/mt5-small` atau model IndoBERT via HuggingFace `Seq2SeqTrainer`, beam search decoding, auto-load checkpoint |
| **Evaluator** | ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1), laporan per-dokumen & agregat, export CSV/JSON |

---

## Instalasi

```bash
pip install -r requirements.txt
```

Dependencies utama:
- `pandas`, `numpy` — data handling
- `nltk`, `PySastrawi` — preprocessing bahasa Indonesia
- `scikit-learn`, `networkx` — TF-IDF & TextRank
- `transformers`, `torch`, `datasets` — model abstractive
- `rouge-score` — evaluasi ROUGE

---

## Cara Pakai

### Format Dataset

Siapkan file CSV atau JSON di `data/raw/` dengan minimal dua kolom:

| full_text | summary |
|---|---|
| Isi lengkap dokumen... | Ringkasan referensi... |

### Menjalankan Pipeline

```bash
# Full pipeline (train + evaluate, kedua metode)
python main.py

# Hanya extractive
python main.py --model extractive

# Hanya abstractive
python main.py --model abstractive

# Train model abstractive saja
python main.py --mode train --model abstractive

# Evaluate saja (butuh checkpoint)
python main.py --mode evaluate

# Override path dataset
python main.py --data path/ke/dataset.csv
```

### CLI Arguments

| Argument | Pilihan | Default | Keterangan |
|---|---|---|---|
| `--mode` | `train`, `evaluate`, `full` | `full` | Mode pipeline |
| `--model` | `extractive`, `abstractive`, `both` | `both` | Metode yang dijalankan |
| `--data` | path ke file | dari `config.py` | Override path dataset |

---

## Konfigurasi

Semua parameter ada di `config.py`:

```python
DATASET_PATH = "data/raw/dataset.csv"
TEXT_COLUMN = "full_text"
SUMMARY_COLUMN = "summary"

NUM_EXTRACTIVE_SENTENCES = 5
ABSTRACTIVE_MODEL_NAME = "google/mt5-small"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
NUM_BEAMS = 4
BATCH_SIZE = 4
NUM_EPOCHS = 3

ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
RANDOM_SEED = 42
```

---

## Output

### Console

```
================================================
  TEXT SUMMARIZATION EVALUATION REPORT
================================================
Method              ROUGE-1   ROUGE-2   ROUGE-L
------------------------------------------------
Extractive           0.XXXX    0.XXXX    0.XXXX
Abstractive          0.XXXX    0.XXXX    0.XXXX
------------------------------------------------
Best Method:                      [Extractive / Abstractive]
================================================
```

### File Output

| File | Keterangan |
|---|---|
| `output/summaries/extractive_summaries.csv` | Hasil extractive |
| `output/summaries/abstractive_summaries.csv` | Hasil abstractive |
| `output/results/evaluation_report.csv` | Skor ROUGE per dokumen |
| `output/results/comparison_summary.json` | Perbandingan agregat |

---

## Testing Per Modul

Setiap modul bisa dijalankan secara independen untuk testing:

```bash
python -m src.utils
python -m src.data_loader
python -m src.preprocessor
python -m src.extractive_model
python -m src.abstractive_model
python -m src.evaluator
```

---

## Author

**devnolife**
