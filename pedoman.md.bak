# Pedoman Project: NLP Text Summarization Pipeline

> **Panduan lengkap dan mendetail untuk memahami seluruh project Summarization Teks Bahasa Indonesia — dari konsep dasar hingga implementasi kode.**
> Dibuat oleh: Ainur Ridha Surya (D082251037), Nirwana Samrin (D082251003), Andi Agung Dwi Arya B (D082251057)

---

## Daftar Isi

1. [Apa Itu Project Ini?](#1-apa-itu-project-ini)
2. [Konsep Dasar yang Harus Dipahami](#2-konsep-dasar-yang-harus-dipahami)
3. [Struktur Folder (Seluruh File Dijelaskan)](#3-struktur-folder-seluruh-file-dijelaskan)
4. [Alur Kerja Pipeline dari Awal Sampai Akhir](#4-alur-kerja-pipeline-dari-awal-sampai-akhir)
5. [File `config.py` — Pusat Konfigurasi (Source Code Lengkap)](#5-file-configpy--pusat-konfigurasi-source-code-lengkap)
6. [File `app.py` — Server Web Flask (Penjelasan Detail)](#6-file-apppy--server-web-flask-penjelasan-detail)
7. [Tahap 1 — Preprocessing (`src/preprocessor.py`) — Penjelasan Baris per Baris](#7-tahap-1--preprocessing-srcpreprocessorpy--penjelasan-baris-per-baris)
8. [Tahap 2 — Extractive Summarization (`src/extractive_model.py`) — Penjelasan Detail](#8-tahap-2--extractive-summarization-srcextractive_modelpy--penjelasan-detail)
9. [Tahap 3 — Abstractive Summarization (`src/abstractive_model.py`) — Penjelasan Detail](#9-tahap-3--abstractive-summarization-srcabstractive_modelpy--penjelasan-detail)
10. [Tahap 4 — Evaluasi ROUGE (`src/evaluator.py`) — Penjelasan Detail](#10-tahap-4--evaluasi-rouge-srcevaluatorpy--penjelasan-detail)
11. [File `src/data_loader.py` — Memuat Dataset](#11-file-srcdata_loaderpy--memuat-dataset)
12. [File `src/utils.py` — Fungsi Pembantu](#12-file-srcutilspy--fungsi-pembantu)
13. [File `main.py` — Pipeline CLI (Command Line)](#13-file-mainpy--pipeline-cli-command-line)
14. [Frontend — `templates/index.html`](#14-frontend--templatesindexhtml)
15. [Dataset yang Digunakan](#15-dataset-yang-digunakan)
16. [Cara Menjalankan Project](#16-cara-menjalankan-project)
17. [API Endpoints — Request & Response Lengkap](#17-api-endpoints--request--response-lengkap)
18. [Library yang Digunakan (Lengkap dengan Penjelasan)](#18-library-yang-digunakan-lengkap-dengan-penjelasan)
19. [FAQ / Pertanyaan Umum](#19-faq--pertanyaan-umum)

---

## 1. Apa Itu Project Ini?

Project ini adalah **sistem peringkasan teks otomatis (automatic text summarization)** yang dirancang khusus untuk dokumen **Bahasa Indonesia**. Tujuan utama project ini adalah **membandingkan dua pendekatan summarization** dan menentukan mana yang lebih efektif.

### 1.1 Latar Belakang

Dalam dunia akademik, kita sering berhadapan dengan dokumen-dokumen panjang seperti jurnal, skripsi, atau artikel berita. Membaca dan merangkum semuanya secara manual sangat memakan waktu. Maka dibuatlah sistem yang bisa **merangkum teks secara otomatis** menggunakan teknik Natural Language Processing (NLP).

### 1.2 Dua Pendekatan yang Dibandingkan

| Pendekatan | Cara Kerja | Analogi Sederhana | Output |
|---|---|---|---|
| **Extractive** | Memilih kalimat-kalimat asli terpenting dari teks, tanpa mengubah kata-katanya sama sekali | Seperti mahasiswa yang pakai **highlighter** — baca buku lalu tandai/highlight kalimat paling penting. Kalimat yang di-highlight itulah ringkasannya | Kalimat asli dari teks (copy-paste) |
| **Abstractive** | Membaca seluruh teks, memahami maknanya, lalu **menulis ulang** ringkasan dengan kalimat baru yang belum tentu ada di teks asli | Seperti mahasiswa yang **baca materi** lalu **tulis rangkuman pakai kata-kata sendiri** di buku catatan | Kalimat baru yang digenerate oleh AI |

### 1.3 Kenapa Membandingkan Keduanya?

Setiap pendekatan punya kelebihan dan kekurangan. Dengan membandingkan keduanya secara objektif menggunakan **metrik ROUGE**, kita bisa menentukan pendekatan mana yang lebih cocok untuk teks akademik Bahasa Indonesia.

### 1.4 Teknologi yang Digunakan

- **Extractive**: Algoritma TextRank (berbasis TF-IDF + PageRank dari Google)
- **Abstractive**: Model Transformer mT5 dari Google (deep learning)
- **Evaluasi**: Metrik ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- **Web Interface**: Flask (Python) + Tailwind CSS (frontend)

---

## 2. Konsep Dasar yang Harus Dipahami

Sebelum membaca kode, pahami dulu konsep-konsep ini:

### 2.1 Apa Itu NLP (Natural Language Processing)?

NLP adalah cabang kecerdasan buatan (AI) yang mengajarkan komputer untuk **memahami, menganalisis, dan mengolah bahasa manusia** (teks dan ucapan). Contoh aplikasi NLP: Google Translate, ChatGPT, autocomplete di keyboard HP.

### 2.2 Apa Itu Tokenisasi?

Tokenisasi adalah proses **memecah teks menjadi unit-unit kecil** yang disebut **token**. Ada dua jenis:

- **Tokenisasi kalimat**: Memecah paragraf menjadi kalimat-kalimat
  ```
  "Saya belajar. Dia bekerja." → ["Saya belajar.", "Dia bekerja."]
  ```
- **Tokenisasi kata**: Memecah kalimat menjadi kata-kata
  ```
  "Saya belajar NLP" → ["Saya", "belajar", "NLP"]
  ```

### 2.3 Apa Itu Stopword?

Stopword adalah kata-kata yang **sangat umum** dan **tidak bermakna penting** dalam analisis teks. Contoh stopword Bahasa Indonesia: "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada", "adalah". Kata-kata ini dihapus karena tidak membantu membedakan kalimat penting dari yang tidak penting.

### 2.4 Apa Itu Stemming?

Stemming adalah proses mengembalikan kata ke **bentuk dasarnya** dengan menghapus imbuhan (awalan, akhiran, sisipan). Contoh:
```
"pembelajaran"  → "ajar"
"berlari"       → "lari"  
"memakan"       → "makan"
"penerbangan"   → "terbang"
"mempermasalahkan" → "masalah"
```
Tujuannya agar komputer tahu bahwa "berlari", "pelari", "berlarian" semuanya berasal dari kata yang sama: "lari".

### 2.5 Apa Itu TF-IDF?

**TF-IDF (Term Frequency–Inverse Document Frequency)** adalah cara mengukur seberapa penting sebuah kata dalam sebuah dokumen:

- **TF (Term Frequency)**: Seberapa sering kata muncul dalam **satu kalimat/dokumen**. Semakin sering → semakin penting untuk kalimat itu.
- **IDF (Inverse Document Frequency)**: Seberapa jarang kata muncul di **seluruh koleksi dokumen**. Kata yang jarang muncul di banyak dokumen → semakin informatif/penting.
- **TF-IDF = TF × IDF**: Kata yang sering di satu dokumen tapi jarang di dokumen lain mendapat skor tinggi.

Contoh: Kata "algoritma" muncul 5 kali di satu paper tapi jarang di paper lain → skor TF-IDF tinggi. Kata "dan" muncul di mana-mana → skor TF-IDF rendah.

### 2.6 Apa Itu Cosine Similarity?

Cosine Similarity mengukur **kemiripan antara dua vektor** dengan menghitung kosinus sudut di antara keduanya. Nilainya antara 0 (tidak mirip sama sekali) sampai 1 (identik). Dalam konteks kita, digunakan untuk mengukur kemiripan antara dua kalimat yang sudah direpresentasikan sebagai vektor TF-IDF.

### 2.7 Apa Itu PageRank?

PageRank adalah algoritma yang diciptakan oleh **Larry Page (pendiri Google)** untuk menentukan halaman web mana yang paling penting. Prinsipnya: sebuah halaman dianggap penting jika banyak halaman penting lain mengarah ke halaman tersebut.

Dalam project ini, prinsip yang sama diterapkan pada **kalimat**: kalimat dianggap penting jika banyak kalimat penting lain "mirip" dengannya (cosine similarity tinggi).

### 2.8 Apa Itu Transformer / mT5?

**Transformer** adalah arsitektur deep learning yang revolusioner untuk NLP. Model **mT5 (multilingual T5)** dari Google adalah transformer yang mendukung 101 bahasa termasuk Bahasa Indonesia. Arsitekturnya:

- **Encoder**: Membaca dan memahami seluruh teks input
- **Decoder**: Menghasilkan teks output (ringkasan) kata per kata
- **Self-Attention**: Mekanisme yang membuat setiap kata bisa "memperhatikan" semua kata lain dalam teks untuk memahami konteks

### 2.9 Apa Itu ROUGE?

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** adalah metrik standar untuk mengevaluasi kualitas ringkasan. Cara kerjanya: membandingkan **kata-kata yang ada di ringkasan** dengan **kata-kata yang ada di ringkasan referensi** (ringkasan buatan manusia yang dianggap benar).

---

## 3. Struktur Folder (Seluruh File Dijelaskan)

```
academic-text-summarization-id/
│
├── app.py                      ← Server web Flask. File ini yang dijalankan untuk 
│                                  menyalakan web app. Berisi semua API endpoint.
│                                  Menghubungkan frontend (HTML) dengan backend (NLP).
│
├── main.py                     ← Entry point untuk menjalankan pipeline dari 
│                                  command line (tanpa web). Digunakan untuk 
│                                  training model dan evaluasi batch.
│
├── config.py                   ← PUSAT KONFIGURASI. Semua pengaturan yang bisa 
│                                  diubah ada di sini: path dataset, nama model, 
│                                  hyperparameter, jumlah kalimat extractive, dll.
│                                  INI FILE PERTAMA YANG HARUS DIBACA.
│
├── gunicorn_config.py          ← Konfigurasi Gunicorn (web server production).
│                                  Mengatur jumlah worker, port, timeout.
│
├── requirements.txt            ← Daftar semua library Python yang dibutuhkan.
│                                  Install dengan: pip install -r requirements.txt
│
├── pedoman.md                  ← FILE INI. Dokumentasi lengkap project.
│
├── plan.md                     ← Rencana pengembangan project.
│
├── README.md                   ← Deskripsi singkat project.
│
├── src/                        ← FOLDER INTI — semua logika NLP ada di sini
│   ├── __init__.py             ← File kosong yang menandai folder ini sebagai 
│   │                              Python package (agar bisa di-import)
│   │
│   ├── preprocessor.py         ← CLASS TextPreprocessor
│   │                              Berisi 6 tahap preprocessing:
│   │                              1. case_folding()  - huruf kecil
│   │                              2. clean_text()    - hapus simbol
│   │                              3. sentence_tokenize() - pecah kalimat
│   │                              4. word_tokenize() - pecah kata
│   │                              5. remove_stopwords() - hapus kata umum
│   │                              6. stem_tokens()   - kata dasar
│   │
│   ├── extractive_model.py     ← CLASS ExtractiveSummarizer
│   │                              Implementasi TextRank:
│   │                              1. _get_sentences()         - ambil kalimat
│   │                              2. _build_tfidf_matrix()    - vektor TF-IDF
│   │                              3. _build_similarity_graph() - graph kemiripan
│   │                              4. _rank_sentences()        - PageRank
│   │                              5. summarize()              - pilih top-N
│   │
│   ├── abstractive_model.py    ← CLASS AbstractiveSummarizer + SummarizationDataset
│   │                              Implementasi mT5 Transformer:
│   │                              - _load_model()    - muat model dari HuggingFace
│   │                              - _load_checkpoint() - muat model yang sudah di-train
│   │                              - fine_tune()      - training model
│   │                              - summarize()      - generate ringkasan
│   │                              - batch_summarize() - ringkas banyak dokumen
│   │
│   ├── evaluator.py            ← CLASS Evaluator
│   │                              Evaluasi ROUGE:
│   │                              - compute_rouge()           - skor agregat
│   │                              - compute_per_document()    - skor per dokumen
│   │                              - generate_comparison_table() - tabel perbandingan
│   │
│   ├── data_loader.py          ← CLASS DataLoader
│   │                              Memuat dan memproses dataset:
│   │                              - load_dataset()     - baca CSV/JSON
│   │                              - validate_columns() - cek kolom
│   │                              - clean_dataset()    - bersihkan data
│   │                              - split_dataset()    - bagi train/val/test
│   │
│   └── utils.py                ← Fungsi pembantu:
│                                  - setup_logging()    - konfigurasi log
│                                  - save_json()        - simpan hasil ke JSON
│                                  - set_seed()         - reproducibility
│                                  - compute_text_stats() - statistik teks
│
├── templates/
│   └── index.html              ← HALAMAN WEB UTAMA (frontend)
│                                  Single page application dengan:
│                                  - Upload dataset / input manual
│                                  - Tampilan hasil preprocessing (6 tahap)
│                                  - Hasil summarization (extractive vs abstractive)
│                                  - Evaluasi ROUGE dengan bar chart
│                                  - Source code Python di setiap langkah
│                                  - Penjelasan "Apa Yang Terjadi" di setiap proses
│                                  Teknologi: Tailwind CSS, Lucide Icons, vanilla JS
│
├── data/
│   ├── raw/
│   │   └── dataset.csv         ← DATASET UTAMA
│   │                              20 artikel berita Indonesia dari XL-Sum
│   │                              Kolom: full_text (teks), summary (ringkasan)
│   └── processed/              ← Folder untuk data yang sudah diproses (kosong)
│
├── output/
│   ├── checkpoints/
│   │   └── best_model/         ← MODEL mT5 YANG SUDAH DI-FINETUNING
│   │       ├── config.json     ← Konfigurasi arsitektur model
│   │       ├── generation_config.json ← Pengaturan generasi teks
│   │       ├── model.safetensors ← Bobot/weight model (~1.2GB)
│   │       ├── tokenizer_config.json ← Konfigurasi tokenizer
│   │       └── tokenizer.json  ← Vocabulary tokenizer (250K subwords)
│   ├── results/                ← Hasil evaluasi disimpan di sini (JSON)
│   └── summaries/              ← Ringkasan yang dihasilkan disimpan di sini
│
└── deploy/                     ← File untuk deploy ke server production
    ├── setup.sh                ← Script instalasi otomatis
    ├── nlp-summarization.service ← Systemd service file
    └── nginx-nlp-summarization.conf ← Konfigurasi Nginx reverse proxy
```

---

## 4. Alur Kerja Pipeline dari Awal Sampai Akhir

Ketika user menekan tombol "Mulai Proses Summarization" di web, inilah yang terjadi **langkah demi langkah**:

### 4.1 Diagram Alur Lengkap

```
USER
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 0: INPUT DATA                                          │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │ Upload CSV           │ OR│ Input teks manual            │  │
│  │ (kolom: full_text,  │   │ (paste teks + referensi)     │  │
│  │  summary)           │   │                              │  │
│  └─────────┬───────────┘   └──────────────┬───────────────┘  │
│            └──────────────┬───────────────┘                   │
│                           ▼                                   │
│            JavaScript menyimpan array texts[]                 │
│            dan summaries[] di appState                        │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING (POST /api/preprocess)                │
│                                                              │
│  Untuk SETIAP teks di array texts[]:                         │
│                                                              │
│  Teks Asli ──→ case_folding() ──→ Semua huruf kecil         │
│       │                                                      │
│       ▼                                                      │
│  clean_text() ──→ Hapus URL, email, HTML, angka, simbol      │
│       │                                                      │
│       ▼                                                      │
│  sentence_tokenize() ──→ Pecah jadi list kalimat             │
│       │                                                      │
│       ▼                                                      │
│  word_tokenize() ──→ Pecah jadi list kata                    │
│       │                                                      │
│       ▼                                                      │
│  remove_stopwords() ──→ Hapus kata umum (dan, yang, di...)   │
│       │                                                      │
│       ▼                                                      │
│  stem_tokens() ──→ Ubah ke bentuk dasar (pembelajaran→ajar)  │
│                                                              │
│  RETURN: original, case_folding, cleaning, sentences,        │
│          word_tokens, after_stopword_removal, after_stemming, │
│          num_sentences, num_tokens_before/after/stemmed       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: SUMMARIZATION (POST /api/summarize)                 │
│                                                              │
│  ┌─── EXTRACTIVE ─────────────────────────────────────────┐  │
│  │ Untuk setiap teks:                                     │  │
│  │ 1. _get_sentences(text)     → list kalimat bersih      │  │
│  │ 2. _build_tfidf_matrix()    → matriks TF-IDF           │  │
│  │ 3. cosine_similarity()      → matriks kemiripan        │  │
│  │ 4. nx.from_numpy_array()    → buat graph               │  │
│  │ 5. nx.pagerank()            → skor setiap kalimat      │  │
│  │ 6. sorted() top-N           → pilih 5 kalimat terbaik  │  │
│  │ 7. " ".join()               → gabung jadi ringkasan    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─── ABSTRACTIVE ────────────────────────────────────────┐  │
│  │ 1. Load model mT5 + tokenizer (lazy, pertama kali)    │  │
│  │ 2. tokenizer(text)          → ubah teks ke token ID    │  │
│  │ 3. model.generate()         → beam search decode       │  │
│  │ 4. tokenizer.decode()       → ubah token ID ke teks    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  RETURN: extractive[], abstractive[],                        │
│          extractive_details[], abstractive_details{}          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: EVALUASI (POST /api/evaluate)                       │
│                                                              │
│  Untuk setiap dokumen:                                       │
│  1. scorer.score(referensi, prediksi_extractive)             │
│     → ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)     │
│  2. scorer.score(referensi, prediksi_abstractive)            │
│     → ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)     │
│                                                              │
│  3. Hitung rata-rata F1 extractive vs abstractive            │
│  4. Tentukan best_method (yang rata-rata F1 lebih tinggi)    │
│                                                              │
│  RETURN: per_document[{ext:{rouge1,rouge2,rougeL},           │
│          abs:{rouge1,rouge2,rougeL}}],                       │
│          best_method, extractive_avg_f1, abstractive_avg_f1  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: PERBANDINGAN & KESIMPULAN (di frontend)             │
│                                                              │
│  - Bar chart ROUGE scores (extractive vs abstractive)        │
│  - Tabel detail precision/recall/F1                          │
│  - Badge "Metode Terbaik" dengan skor rata-rata F1           │
│  - Analisis tertulis otomatis                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Alur Data di Kode (Bagaimana File-File Terhubung)

```
User klik "Mulai" di browser
        │
        ▼
templates/index.html (JavaScript)
  │  startPipeline()
  │    ├── doPreprocessing()  ──→ fetch('/api/preprocess')
  │    ├── doSummarization()  ──→ fetch('/api/summarize')
  │    └── doEvaluation()     ──→ fetch('/api/evaluate')
  │
  ▼
app.py (Flask Server)
  │  @app.route("/api/preprocess")
  │    └── Panggil preprocessor.case_folding(), .clean_text(), dll
  │                └── src/preprocessor.py (TextPreprocessor)
  │
  │  @app.route("/api/summarize")
  │    ├── Panggil extractive_summarizer._get_sentences(), dll
  │    │          └── src/extractive_model.py (ExtractiveSummarizer)
  │    └── Panggil abs_model.batch_summarize()
  │               └── src/abstractive_model.py (AbstractiveSummarizer)
  │
  │  @app.route("/api/evaluate")
  │    └── Panggil evaluator.compute_rouge(), evaluator.scorer.score()
  │               └── src/evaluator.py (Evaluator)
  │
  ▼
Hasil JSON dikembalikan ke frontend → ditampilkan di browser
```

---

## 5. File `config.py` — Pusat Konfigurasi (Source Code Lengkap)

**INI FILE PERTAMA YANG HARUS DIBACA.** Semua pengaturan yang bisa diubah terkumpul di sini.

```python
# config.py — Konfigurasi lengkap project

import os

# ═══════════════════════════════════════════════════════════════
# PATH — Lokasi file dan folder
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # Root folder project
DATA_DIR = os.path.join(BASE_DIR, "data")                # Folder data/
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")             # Folder data/raw/
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed") # Folder data/processed/
OUTPUT_DIR = os.path.join(BASE_DIR, "output")            # Folder output/
SUMMARIES_DIR = os.path.join(OUTPUT_DIR, "summaries")    # Simpan ringkasan
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")        # Simpan hasil evaluasi
MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints") # Simpan model

# ═══════════════════════════════════════════════════════════════
# DATASET — Pengaturan dataset
# ═══════════════════════════════════════════════════════════════
DATASET_PATH = os.path.join(RAW_DATA_DIR, "dataset.csv") # Lokasi file dataset
TEXT_COLUMN = "full_text"          # Nama kolom yang berisi teks lengkap
SUMMARY_COLUMN = "summary"        # Nama kolom yang berisi ringkasan referensi
MAX_INPUT_LENGTH = 10000           # Batas maksimal karakter teks input

# Rasio pembagian dataset untuk training
TRAIN_RATIO = 0.8    # 80% untuk training
VAL_RATIO = 0.1      # 10% untuk validasi
TEST_RATIO = 0.1     # 10% untuk testing

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING — Pengaturan preprocessing teks
# ═══════════════════════════════════════════════════════════════
CUSTOM_STOPWORDS = []  # Tambahkan stopword kustom di sini jika perlu
                       # Contoh: ["tersebut", "merupakan", "sehingga"]

# ═══════════════════════════════════════════════════════════════
# EXTRACTIVE — Pengaturan extractive summarization
# ═══════════════════════════════════════════════════════════════
NUM_EXTRACTIVE_SENTENCES = 5    # Jumlah kalimat yang akan dipilih
                                # Semakin banyak → ringkasan semakin panjang
TFIDF_MAX_FEATURES = 5000       # Maks fitur TF-IDF (dimensi vektor)
                                # Semakin tinggi → lebih detail tapi lebih lambat

# ═══════════════════════════════════════════════════════════════
# ABSTRACTIVE — Pengaturan abstractive summarization
# ═══════════════════════════════════════════════════════════════
ABSTRACTIVE_MODEL_NAME = "google/mt5-small"
    # Pilihan model:
    # "google/mt5-small"          → model pre-trained umum, perlu fine-tuning
    # "csebuetnlp/mT5_multilingual_XLSum-small" → sudah fine-tuned untuk summarization
    # "LazarusNLP/IndoNanoT5-base" → model T5 khusus Indonesia

MAX_SOURCE_LENGTH = 256   # Maks token input (256 token ≈ 130 kata Indonesia)
                          # Selebihnya dipotong (truncation)
MAX_TARGET_LENGTH = 64    # Maks token output ringkasan (64 token ≈ 32 kata)
NUM_BEAMS = 4             # Jumlah beam search — semakin banyak:
                          # ✅ Kualitas lebih baik
                          # ❌ Semakin lambat
BATCH_SIZE = 1            # Jumlah dokumen diproses sekaligus (1 karena pakai CPU)
NUM_EPOCHS = 3            # Jumlah epoch fine-tuning (berapa kali model melihat
                          # seluruh data training)
LEARNING_RATE = 5e-5      # Learning rate (seberapa besar langkah belajar model)
                          # Terlalu besar → model tidak stabil
                          # Terlalu kecil → model lambat belajar
WEIGHT_DECAY = 0.01       # Regularisasi (mencegah overfitting)
WARMUP_STEPS = 0          # Langkah pemanasan learning rate
GRADIENT_ACCUMULATION_STEPS = 1  # Akumulasi gradient (simulasi batch lebih besar)
FP16 = False              # Mixed precision (True hanya kalau ada GPU NVIDIA)

# ═══════════════════════════════════════════════════════════════
# EVALUASI — Metrik yang dihitung
# ═══════════════════════════════════════════════════════════════
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]

# ═══════════════════════════════════════════════════════════════
# UMUM
# ═══════════════════════════════════════════════════════════════
RANDOM_SEED = 42          # Seed agar hasil bisa direproduksi (sama setiap kali)
LOG_LEVEL = "INFO"        # Level logging: DEBUG (paling detail), INFO, WARNING, ERROR
```

**Penjelasan penting:**

- **`MAX_SOURCE_LENGTH = 256`**: Ini berarti model abstractive hanya membaca **256 token pertama** dari teks. Jika teksnya sangat panjang, bagian akhir akan dipotong. Ini trade-off antara memori/kecepatan vs kelengkapan.
- **`NUM_BEAMS = 4`**: Beam search mengeksplorasi 4 kandidat ringkasan secara paralel, mengurangi risiko output yang jelek. Tapi 4 beam membutuhkan 4× komputasi.
- **`RANDOM_SEED = 42`**: Angka 42 tidak spesial — hanya konvensi. Yang penting semua operasi random menggunakan seed yang sama agar hasilnya reproducible.

---

## 6. File `app.py` — Server Web Flask (Penjelasan Detail)

File ini adalah **jembatan** antara frontend (browser) dan backend (kode NLP). Inilah source code lengkapnya dengan penjelasan:

### 6.1 Inisialisasi Server

```python
# app.py — Bagian Inisialisasi

import os, sys, json, logging, traceback, tempfile
from typing import Dict, List, Optional
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import config
from src.utils import set_seed
from src.preprocessor import TextPreprocessor
from src.extractive_model import ExtractiveSummarizer
from src.evaluator import Evaluator

# Buat instance Flask
app = Flask(__name__)
CORS(app)  # Izinkan cross-origin request (agar bisa diakses dari domain lain)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Batas upload 16MB

# Set random seed untuk reproducibility
set_seed(config.RANDOM_SEED)

# Buat instance kelas-kelas NLP
preprocessor = TextPreprocessor()           # Siap pakai langsung
extractive_summarizer = ExtractiveSummarizer()  # Siap pakai langsung (ringan)
evaluator = Evaluator()                     # Siap pakai langsung
```

**Yang perlu diperhatikan:**

1. Ketiga instance di atas dibuat **saat server startup** karena mereka ringan
2. Model abstractive **TIDAK** di-load saat startup karena ukurannya besar (~1.2GB)
3. Abstractive menggunakan **lazy loading** — baru dimuat saat pertama kali dipanggil:

```python
# app.py — Lazy loading model abstractive

_abstractive_summarizer = None  # Awalnya None (belum dimuat)

def get_abstractive_summarizer():
    """Muat model abstractive saat pertama kali dipanggil."""
    global _abstractive_summarizer
    
    if _abstractive_summarizer is None:  # Belum pernah dimuat
        from src.abstractive_model import AbstractiveSummarizer
        _abstractive_summarizer = AbstractiveSummarizer()
        
        # Cek apakah ada checkpoint model yang sudah di-finetuning
        checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model")
        if os.path.exists(checkpoint_path):
            # Ada! Load model yang sudah ditrain
            _abstractive_summarizer._load_checkpoint(checkpoint_path)
        else:
            # Tidak ada. Load model pre-trained dari HuggingFace
            _abstractive_summarizer._load_model()
    
    return _abstractive_summarizer
```

### 6.2 API Endpoint: Upload File (`/api/upload`)

```python
@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Terima file CSV/JSON dari user, validasi, dan kembalikan isinya."""
    
    # 1. Cek apakah ada file yang dikirim
    file = request.files["file"]
    
    # 2. Simpan file sementara
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"upload.{ext}")
    file.save(filepath)
    
    # 3. Baca file dengan pandas
    if ext == "csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_json(filepath)
    
    # 4. Validasi: apakah kolom 'full_text' dan 'summary' ada?
    if text_col not in df.columns:
        return jsonify({"error": f"Column '{text_col}' not found"}), 400
    
    # 5. Bersihkan: hapus baris yang kosong/null
    df = df.dropna(subset=[text_col, summary_col])
    
    # 6. Ekstrak teks dan kirim ke frontend
    texts = df[text_col].tolist()      # List semua teks
    summaries = df[summary_col].tolist()  # List semua ringkasan referensi
    
    return jsonify({
        "num_documents": len(texts),
        "texts": texts,
        "summaries": summaries,
        "preview": [...],  # 5 preview pertama (dipotong 200 karakter)
        "stats": {"avg_text_length": ..., "avg_summary_length": ...}
    })
```

### 6.3 API Endpoint: Preprocessing (`/api/preprocess`)

```python
@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    """Jalankan 6 tahap preprocessing dan kembalikan hasil setiap tahap."""
    
    data = request.get_json()
    texts = data.get("texts", [])  # Array teks dari frontend
    
    results = []
    for text in texts:
        original = text
        
        # Tahap 1: Case Folding — huruf kecil semua
        step1 = preprocessor.case_folding(text)
        
        # Tahap 2: Cleaning — hapus URL, email, simbol
        step2 = preprocessor.clean_text(step1)
        
        # Tahap 3: Tokenisasi Kalimat — pecah jadi list kalimat
        step3 = preprocessor.sentence_tokenize(step2)
        
        # Tahap 4: Tokenisasi Kata — pecah jadi list kata
        step4 = preprocessor.word_tokenize(step2)
        
        # Tahap 5: Stopword Removal — hapus kata umum
        step5 = preprocessor.remove_stopwords(step4)
        
        # Tahap 6: Stemming — ubah ke bentuk dasar
        step6 = preprocessor.stem_tokens(step5)
        
        results.append({
            "original": original,             # Teks asli
            "case_folding": step1,            # Setelah lowercase
            "cleaning": step2,                # Setelah cleaning
            "sentences": step3,               # List kalimat
            "word_tokens": step4,             # List kata
            "after_stopword_removal": step5,  # Kata tanpa stopword
            "after_stemming": step6,          # Kata bentuk dasar
            "num_sentences": len(step3),      # Jumlah kalimat
            "num_tokens_before": len(step4),  # Jumlah kata sebelum
            "num_tokens_after": len(step5),   # Jumlah kata setelah stopword
            "num_tokens_stemmed": len(step6), # Jumlah kata setelah stemming
        })
    
    return jsonify({"success": True, "results": results})
```

### 6.4 API Endpoint: Summarization (`/api/summarize`)

Ini endpoint yang paling kompleks karena menjalankan **dua metode summarization sekaligus** dan mengembalikan detail langkah-langkah extractive:

```python
@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    texts = data.get("texts", [])
    
    result = {"success": True}
    
    # ═══ EXTRACTIVE ═══
    for text in texts:
        # Step 1: Tokenisasi kalimat
        sentences = extractive_summarizer._get_sentences(text)
        
        # Step 2: TF-IDF — ubah kalimat jadi vektor
        tfidf_matrix = extractive_summarizer._build_tfidf_matrix(sentences)
        # detail["tfidf_shape"] = [jumlah_kalimat, jumlah_fitur]
        
        # Step 3: Cosine Similarity — hitung kemiripan antar kalimat
        sim_matrix = cos_sim(tfidf_matrix)
        # Simpan rata-rata kemiripan dan pasangan kalimat paling mirip
        
        # Step 4: PageRank — ranking kalimat
        graph = extractive_summarizer._build_similarity_graph(tfidf_matrix)
        scores = extractive_summarizer._rank_sentences(graph)
        
        # Step 5: Seleksi top-N kalimat
        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        top_indices = sorted(ranked_indices[:n_target])
        
        # Simpan skor setiap kalimat untuk ditampilkan di frontend
        # sentence_scores = [{index, sentence, score, rank, selected}, ...]
        
        # Step 6: Gabung kalimat terpilih
        summary = " ".join([sentences[i] for i in top_indices])
    
    # ═══ ABSTRACTIVE ═══
    abs_model = get_abstractive_summarizer()  # Lazy load model
    abs_summaries = abs_model.batch_summarize(texts)  # Generate ringkasan
    
    return jsonify({
        "extractive": ext_summaries,           # List ringkasan extractive
        "abstractive": abs_summaries,          # List ringkasan abstractive
        "extractive_details": ext_details,     # Detail per dokumen
        "abstractive_details": {               # Info model
            "model_name": "google/mt5-small",
            "device": "cpu",
            "num_parameters": "300,000,000",
            ...
        }
    })
```

### 6.5 API Endpoint: Evaluasi (`/api/evaluate`)

```python
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    references = data.get("references", [])          # Ringkasan referensi
    extractive_preds = data.get("extractive_preds")  # Hasil extractive
    abstractive_preds = data.get("abstractive_preds") # Hasil abstractive
    
    # Hitung ROUGE per dokumen
    per_document = []
    for i in range(len(references)):
        doc = {}
        
        # ROUGE untuk extractive
        sc = evaluator.scorer.score(references[i], extractive_preds[i])
        doc["ext"] = {
            "rouge1": {"p": sc["rouge1"].precision, "r": ..., "f": ...},
            "rouge2": {"p": ..., "r": ..., "f": ...},
            "rougeL": {"p": ..., "r": ..., "f": ...},
        }
        
        # ROUGE untuk abstractive (sama caranya)
        doc["abs"] = { ... }
        
        per_document.append(doc)
    
    # Tentukan metode terbaik berdasarkan rata-rata F1
    ext_avg = rata-rata F1 dari semua metrik ROUGE extractive
    abs_avg = rata-rata F1 dari semua metrik ROUGE abstractive
    best_method = "Extractive" if ext_avg >= abs_avg else "Abstractive"
    
    return jsonify({
        "per_document": per_document,
        "best_method": best_method,
        "extractive_avg_f1": ext_avg,
        "abstractive_avg_f1": abs_avg
    })
```

---

## 7. Tahap 1 — Preprocessing (`src/preprocessor.py`) — Penjelasan Baris per Baris

### 7.0 Inisialisasi Class

Saat `TextPreprocessor()` dibuat, ia melakukan 3 hal:

```python
class TextPreprocessor:
    def __init__(self, custom_stopwords=None):
        # 1. LOAD STOPWORDS — Daftar kata umum Bahasa Indonesia dari NLTK
        #    NLTK punya ~759 stopword Indonesia bawaan
        self.indonesian_stopwords = set(stopwords.words("indonesian"))
        # Contoh isi: {"yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", ...}
        
        # 2. TAMBAH CUSTOM STOPWORDS dari config.py (jika ada)
        if config.CUSTOM_STOPWORDS:
            self.indonesian_stopwords.update(config.CUSTOM_STOPWORDS)
        
        # 3. INISIALISASI STEMMER — PySastrawi (stemmer khusus Bahasa Indonesia)
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        # Stemmer ini punya kamus 28.000+ kata dasar Bahasa Indonesia
```

### 7.1 Case Folding — `case_folding(text)`

**Source code lengkap:**
```python
def case_folding(self, text: str) -> str:
    """Step 1: Convert all text to lowercase."""
    if not text:        # Jika teks kosong/None, kembalikan string kosong
        return ""
    return text.lower() # Ubah semua huruf jadi lowercase
```

**Apa yang terjadi:**
- Fungsi bawaan Python `str.lower()` mengubah setiap karakter huruf besar menjadi huruf kecil
- Karakter non-huruf (angka, simbol, spasi) tidak berubah
- Tujuannya: agar "Mahasiswa", "MAHASISWA", dan "mahasiswa" dianggap **kata yang sama** oleh komputer

**Contoh step-by-step:**
```
Input:  "Penelitian Ini Bertujuan untuk MENGANALISIS Pengaruh Media SOSIAL"
Output: "penelitian ini bertujuan untuk menganalisis pengaruh media sosial"
```

**Kenapa penting?** Tanpa case folding, TF-IDF akan menghitung "Penelitian" dan "penelitian" sebagai dua kata berbeda, padahal maknanya sama.

---

### 7.2 Cleaning — `clean_text(text)`

**Source code lengkap:**
```python
def clean_text(self, text: str) -> str:
    """Step 2: Clean text by removing noise."""
    if not text:
        return ""

    # Hapus URL (http://... atau www...)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # regex: cocokkan "http://" atau "https://" diikuti karakter non-spasi
    # Contoh: "lihat https://example.com/page di sini" → "lihat   di sini"

    # Hapus alamat email
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # regex: cocokkan pola "sesuatu@sesuatu.sesuatu"
    # Contoh: "kontak admin@unhas.ac.id ya" → "kontak   ya"

    # Hapus tag HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # regex: cocokkan "<" diikuti karakter apapun kecuali ">" lalu ">"
    # Contoh: "<b>penting</b>" → " penting "

    # Hapus referensi angka dalam kurung siku
    text = re.sub(r"\[\d+\]", " ", text)
    # regex: cocokkan "[angka]"
    # Contoh: "menurut penelitian [3] bahwa" → "menurut penelitian   bahwa"

    # Hapus semua karakter kecuali huruf, spasi, dan tanda baca akhir kalimat
    text = re.sub(r"[^a-zA-Z\s.?!]", " ", text)
    # regex: ganti semua yang BUKAN huruf/spasi/titik/tanda-tanya/seru
    # Yang dihapus: angka (0-9), koma, titik dua, kurung, dll
    # Yang DIPERTAHANKAN: huruf, spasi, . ? !
    # Kenapa . ? ! dipertahankan? Karena dibutuhkan untuk tokenisasi kalimat nanti

    # Hapus spasi berlebih (multiple spasi jadi single spasi)
    text = re.sub(r"\s+", " ", text).strip()
    # Contoh: "kata   banyak    spasi" → "kata banyak spasi"

    return text
```

**Contoh step-by-step:**
```
Input:  "Lihat di https://jurnal.com/artikel [3]. NLP ver.2.0 (Natural!) email@test.com"

1. Hapus URL:      "Lihat di   [3]. NLP ver.2.0 (Natural!) email@test.com"
2. Hapus email:    "Lihat di   [3]. NLP ver.2.0 (Natural!)  "
3. Hapus HTML:     (tidak ada, skip)
4. Hapus [angka]:  "Lihat di    . NLP ver.2.0 (Natural!)  "
5. Hapus non-alfa: "Lihat di    . NLP ver   Natural!   "
6. Hapus spasi:    "Lihat di . NLP ver Natural!"

Output: "Lihat di . NLP ver Natural!"
```

---

### 7.3 Tokenisasi Kalimat — `sentence_tokenize(text)`

**Source code lengkap:**
```python
def sentence_tokenize(self, text: str) -> List[str]:
    """Step 3: Split text into sentences."""
    if not text:
        return []

    try:
        # Gunakan NLTK sent_tokenize dengan model bahasa Indonesia
        sentences = sent_tokenize(text, language="indonesian")
    except LookupError:
        # Fallback ke model default jika model Indonesia tidak ada
        sentences = sent_tokenize(text)

    # Filter kalimat kosong dan hapus spasi di awal/akhir
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
```

**Apa yang terjadi:**
1. NLTK `sent_tokenize` membaca teks dan mencari **batas kalimat**
2. Batas kalimat ditentukan oleh tanda baca akhir: titik (.), tanda tanya (?), tanda seru (!)
3. Tapi cerdas — tidak memecah di singkatan seperti "Dr.", "Prof.", "dll."
4. Parameter `language="indonesian"` menggunakan model tokenisasi khusus Indonesia
5. Hasil akhir: list kalimat tanpa kalimat kosong

**Contoh:**
```
Input:  "Penelitian ini bertujuan menganalisis pengaruh media sosial. Metode yang 
         digunakan adalah survei kuantitatif. Hasilnya signifikan."

Output: [
    "Penelitian ini bertujuan menganalisis pengaruh media sosial.",
    "Metode yang digunakan adalah survei kuantitatif.",
    "Hasilnya signifikan."
]
```

**Kenapa penting untuk extractive?** Extractive summarization bekerja pada level kalimat — ia harus tahu di mana satu kalimat berakhir dan kalimat berikutnya dimulai agar bisa memilih kalimat terpenting.

---

### 7.4 Tokenisasi Kata — `word_tokenize(text)`

**Source code lengkap:**
```python
def word_tokenize(self, text: str) -> List[str]:
    """Step 4: Tokenize text into individual words."""
    if not text:
        return []

    try:
        tokens = word_tokenize(text, language="indonesian")
    except LookupError:
        tokens = word_tokenize(text)

    return tokens
```

**Apa yang terjadi:**
1. NLTK `word_tokenize` memecah teks menjadi kata-kata individual
2. Berbeda dari sekedar `.split(" ")` — tokenizer ini juga memisahkan tanda baca sebagai token tersendiri
3. Menggunakan model bahasa Indonesia untuk menangani kasus khusus

**Contoh:**
```
Input:  "Metode yang digunakan adalah survei kuantitatif."
Output: ["Metode", "yang", "digunakan", "adalah", "survei", "kuantitatif", "."]
```

---

### 7.5 Stopword Removal — `remove_stopwords(tokens)`

**Source code lengkap:**
```python
def remove_stopwords(self, tokens: List[str]) -> List[str]:
    """Step 5: Remove common Indonesian stopwords."""
    if not tokens:
        return []

    filtered = [
        token
        for token in tokens
        if token.lower() not in self.indonesian_stopwords  # Bukan stopword
        and len(token) > 1                                  # Bukan satu karakter
    ]
    return filtered
```

**Apa yang terjadi secara detail:**
1. Loop setiap token dalam list
2. Untuk setiap token, cek 2 kondisi:
   - Apakah token (dalam lowercase) **ada** di set stopwords Indonesia? Jika ya → buang
   - Apakah panjang token ≤ 1 karakter? Jika ya → buang (karena huruf tunggal biasanya noise)
3. Hanya token yang lolos kedua pengecekan yang masuk ke list hasil

**Contoh stopword Indonesia yang dihapus:**
```
yang, dan, di, ke, dari, ini, itu, dengan, untuk, pada, adalah, 
dalam, tidak, akan, juga, sudah, saya, kami, mereka, dia, kita,
bisa, harus, ada, telah, dapat, bagi, seperti, oleh, ...
```

**Contoh step-by-step:**
```
Input:  ["penelitian", "ini", "bertujuan", "untuk", "menganalisis", 
         "pengaruh", "media", "sosial", "yang", "ada", "di", "indonesia"]

Proses:
  "penelitian"  → BUKAN stopword, len > 1 → KEEP ✅
  "ini"         → ADALAH stopword          → BUANG ❌
  "bertujuan"   → BUKAN stopword, len > 1 → KEEP ✅
  "untuk"       → ADALAH stopword          → BUANG ❌
  "menganalisis"→ BUKAN stopword, len > 1 → KEEP ✅
  "pengaruh"    → BUKAN stopword, len > 1 → KEEP ✅
  "media"       → BUKAN stopword, len > 1 → KEEP ✅
  "sosial"      → BUKAN stopword, len > 1 → KEEP ✅
  "yang"        → ADALAH stopword          → BUANG ❌
  "ada"         → ADALAH stopword          → BUANG ❌
  "di"          → ADALAH stopword          → BUANG ❌
  "indonesia"   → BUKAN stopword, len > 1 → KEEP ✅

Output: ["penelitian", "bertujuan", "menganalisis", "pengaruh", 
         "media", "sosial", "indonesia"]
```

**Kenapa penting?** Stopword sangat sering muncul di SEMUA dokumen. Jika tidak dihapus, TF-IDF akan memberikan bobot tinggi pada kata-kata ini padahal mereka tidak informatif.

---

### 7.6 Stemming — `stem_tokens(tokens)`

**Source code lengkap:**
```python
def stem_tokens(self, tokens: List[str]) -> List[str]:
    """Step 6: Apply Indonesian stemming using PySastrawi."""
    if not tokens:
        return []

    if self.stemmer is None:  # Jika PySastrawi tidak terinstall
        return tokens         # Kembalikan apa adanya

    stemmed = [self.stemmer.stem(token) for token in tokens]
    return stemmed
```

**Apa yang terjadi secara detail:**
1. **PySastrawi** menggunakan algoritma stemming Bahasa Indonesia yang dikembangkan oleh **Nazief & Adriani**
2. Algoritma ini mengenali pola imbuhan Bahasa Indonesia:
   - **Prefiks (awalan):** me-, mem-, men-, meng-, meny-, ber-, per-, di-, ke-, se-
   - **Sufiks (akhiran):** -kan, -an, -i
   - **Konfiks (gabungan):** me-...-kan, mem-...-i, per-...-an, dll
3. PySastrawi punya **kamus ~28.000 kata dasar** Indonesia untuk memvalidasi hasil stemming
4. Setiap token diproses satu per satu oleh stemmer

**Contoh step-by-step:**
```
Input:  ["penelitian", "bertujuan", "menganalisis", "pengaruh", 
         "media", "sosial", "indonesia"]

Proses:
  "penelitian"  → hapus pe-...-an  → "teliti"     (kata dasar valid ✅)
  "bertujuan"   → hapus ber-...-an → "tuju"       (kata dasar valid ✅)
  "menganalisis"→ hapus meng-...-is→ "analisis"   (kata dasar valid ✅)
  "pengaruh"    → hapus peng-      → "aruh"? TIDAK → coba lagi → "pengaruh"
                  (ternyata "pengaruh" sendiri adalah kata dasar ✅)
  "media"       → sudah kata dasar → "media"      ✅
  "sosial"      → sudah kata dasar → "sosial"     ✅
  "indonesia"   → sudah kata dasar → "indonesia"  ✅

Output: ["teliti", "tuju", "analisis", "pengaruh", "media", "sosial", "indonesia"]
```

### 7.7 Method Tambahan: `preprocess_for_extractive(text)`

```python
def preprocess_for_extractive(self, text: str) -> List[str]:
    """Preprocessing khusus untuk extractive summarization.
    Hanya case folding + cleaning + sentence tokenize.
    TIDAK melakukan word_tokenize, stopword removal, atau stemming.
    Karena extractive butuh kalimat UTUH untuk dipilih."""
    
    text_lower = self.case_folding(text)       # Step 1
    text_clean = self.clean_text(text_lower)   # Step 2
    sentences = self.sentence_tokenize(text_clean)  # Step 3
    return sentences  # List kalimat yang sudah bersih
```

**Kenapa berbeda?** Extractive summarization memilih **kalimat utuh** sebagai ringkasan. Jika kita lakukan stopword removal dan stemming, kalimatnya rusak dan tidak bisa dibaca manusia. Jadi untuk extractive, cukup case folding + cleaning + tokenisasi kalimat saja.

---

## 8. Tahap 2 — Extractive Summarization (`src/extractive_model.py`) — Penjelasan Detail

### 8.0 Inisialisasi Class

```python
class ExtractiveSummarizer:
    def __init__(self, num_sentences=None, max_features=None):
        # Jumlah kalimat yang akan diekstrak (default: 5 dari config.py)
        self.num_sentences = num_sentences or config.NUM_EXTRACTIVE_SENTENCES  # 5
        
        # Jumlah fitur TF-IDF maksimal (default: 5000 dari config.py)
        self.max_features = max_features or config.TFIDF_MAX_FEATURES  # 5000
        
        # Buat preprocessor sendiri untuk tokenisasi kalimat
        self.preprocessor = TextPreprocessor()
```

### 8.1 Step 1: Ambil Kalimat — `_get_sentences(text)`

```python
def _get_sentences(self, text: str) -> List[str]:
    """Ambil kalimat-kalimat bersih dari teks mentah."""
    sentences = self.preprocessor.preprocess_for_extractive(text)
    return sentences
    # Ini memanggil: case_folding → clean_text → sentence_tokenize
    # Hasilnya: list kalimat yang sudah dibersihkan tapi masih utuh
```

**Contoh:**
```
Input:  "PENELITIAN ini bertujuan... [1] Lihat https://jurnal.com. Metode survei."

Step 1 (case_folding):  "penelitian ini bertujuan... [1] lihat https://jurnal.com. metode survei."
Step 2 (clean_text):    "penelitian ini bertujuan. lihat . metode survei."
Step 3 (sent_tokenize): ["penelitian ini bertujuan.", "lihat .", "metode survei."]

Output: ["penelitian ini bertujuan.", "lihat .", "metode survei."]
```

### 8.2 Step 2: TF-IDF Matrix — `_build_tfidf_matrix(sentences)`

```python
def _build_tfidf_matrix(self, sentences: List[str]) -> np.ndarray:
    """Bangun matriks TF-IDF dari list kalimat."""
    
    # TfidfVectorizer dari scikit-learn
    vectorizer = TfidfVectorizer(
        max_features=self.max_features,  # Maks 5000 kata unik
        stop_words=None,                 # Stopword sudah dihandle di preprocessing
    )
    
    # fit_transform: pelajari vocabulary + hitung TF-IDF sekaligus
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # Hasilnya: sparse matrix berukuran (jumlah_kalimat × jumlah_kata_unik)
    
    return tfidf_matrix
```

**Penjelasan detail cara kerja TF-IDF:**

Anggap kita punya 4 kalimat:
```
Kalimat 0: "penelitian analisis pengaruh media"
Kalimat 1: "metode survei kuantitatif"
Kalimat 2: "hasil penelitian menunjukkan korelasi"
Kalimat 3: "media sosial pengaruh akademik"
```

**Langkah 1 — Hitung TF (Term Frequency):**
```
              penelitian  analisis  pengaruh  media  metode  survei  kuantitatif  hasil  menunjukkan  korelasi  sosial  akademik
Kalimat 0:      1/4       1/4       1/4       1/4     0       0         0          0        0           0        0        0
Kalimat 1:       0          0        0         0      1/3     1/3       1/3         0        0           0        0        0
Kalimat 2:      1/4         0        0         0       0       0         0         1/4      1/4         1/4       0        0
Kalimat 3:       0          0       1/4       1/4      0       0         0          0        0           0       1/4      1/4
```

**Langkah 2 — Hitung IDF (Inverse Document Frequency):**
```
penelitian:  log(4/2) = 0.69  (muncul di 2 dari 4 kalimat → cukup penting)
analisis:    log(4/1) = 1.39  (muncul di 1 dari 4 kalimat → sangat penting!)
media:       log(4/2) = 0.69  (muncul di 2 dari 4 kalimat)
metode:      log(4/1) = 1.39  (muncul di 1 kalimat → sangat penting!)
...
```

**Langkah 3 — TF-IDF = TF × IDF:**

Setiap sel di matriks = TF × IDF. Kata yang sering di satu kalimat tapi jarang di kalimat lain → skor tinggi.

**Hasil:** Matriks berukuran `(4 × 12)` di mana setiap baris adalah **vektor representasi** kalimat tersebut.

### 8.3 Step 3: Cosine Similarity — `_build_similarity_graph(tfidf_matrix)`

```python
def _build_similarity_graph(self, tfidf_matrix: np.ndarray) -> nx.Graph:
    """Bangun graph kemiripan dari matriks TF-IDF."""
    
    # Hitung cosine similarity antar SEMUA pasangan kalimat
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Hasilnya: matriks simetris (jumlah_kalimat × jumlah_kalimat)
    # Contoh untuk 4 kalimat:
    # [[1.00, 0.05, 0.30, 0.40],    ← Kalimat 0 vs semua kalimat
    #  [0.05, 1.00, 0.08, 0.02],    ← Kalimat 1 vs semua kalimat
    #  [0.30, 0.08, 1.00, 0.15],    ← Kalimat 2 vs semua kalimat
    #  [0.40, 0.02, 0.15, 1.00]]    ← Kalimat 3 vs semua kalimat
    #
    # Diagonal selalu 1.00 (kalimat mirip dengan dirinya sendiri)
    # Kalimat 0 & 3 punya similarity 0.40 (cukup mirip)
    
    # Buat graph dari matriks kemiripan
    graph = nx.from_numpy_array(similarity_matrix)
    # Setiap kalimat = node
    # Setiap pasangan kalimat = edge dengan bobot = cosine similarity
    
    # Hapus self-loop (kalimat tidak perlu terhubung ke dirinya sendiri)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    return graph
```

**Visualisasi graph:**
```
    Kalimat 0 ←──0.40──→ Kalimat 3
        │                     │
       0.30                 0.15
        │                     │
    Kalimat 2 ←──0.08──→ Kalimat 1
        
Angka di edge = cosine similarity (kemiripan)
Edge lebih tebal = kalimat lebih mirip
```

### 8.4 Step 4: PageRank — `_rank_sentences(graph)`

```python
def _rank_sentences(self, graph: nx.Graph) -> dict:
    """Jalankan PageRank untuk memberi skor setiap kalimat."""
    
    scores = nx.pagerank(graph, weight="weight")
    # PageRank iteratif:
    # 1. Awalnya semua node punya skor yang sama: 1/N
    # 2. Di setiap iterasi, skor node di-update berdasarkan:
    #    skor_baru(node) = (1-d)/N + d × Σ(skor(tetangga) × bobot_edge / total_bobot_tetangga)
    #    di mana d = damping factor (default 0.85)
    # 3. Proses diulang sampai konvergen (skor tidak berubah lagi)
    
    return scores
    # Contoh output: {0: 0.32, 1: 0.15, 2: 0.28, 3: 0.25}
    # Kalimat 0 paling penting (skor 0.32)
    # Kalimat 1 paling tidak penting (skor 0.15)
```

**Kenapa PageRank bekerja untuk ranking kalimat?**

Intuisi: kalimat yang membahas **topik utama** dokumen akan **mirip dengan banyak kalimat lain** (karena banyak kalimat membahas topik yang sama). Di graph, kalimat ini terhubung dengan banyak edge berbobot tinggi → mendapat skor PageRank tinggi.

### 8.5 Step 5: Seleksi Kalimat — `summarize(text)`

```python
def summarize(self, text: str, num_sentences=None) -> str:
    n_sentences = num_sentences or self.num_sentences  # Default: 5
    
    # Step 1: Ambil kalimat bersih
    sentences = self._get_sentences(text)
    
    # Edge case: jika kalimat lebih sedikit dari yang diminta
    if len(sentences) <= n_sentences:
        return " ".join(sentences)  # Kembalikan semua kalimat
    
    # Step 2-4: TF-IDF → Similarity Graph → PageRank
    tfidf_matrix = self._build_tfidf_matrix(sentences)
    graph = self._build_similarity_graph(tfidf_matrix)
    scores = self._rank_sentences(graph)
    
    # Step 5: Ambil N kalimat dengan skor tertinggi
    ranked_indices = sorted(scores, key=scores.get, reverse=True)
    # Contoh: [0, 2, 3, 1] (indeks diurutkan dari skor tertinggi)
    
    top_indices = sorted(ranked_indices[:n_sentences])
    # Ambil 5 pertama, lalu URUTKAN KEMBALI sesuai posisi asli
    # Kenapa? Agar ringkasan tetap koheren dan runtut
    # Contoh: [0, 2] (bukan [2, 0])
    
    # Step 6: Gabungkan kalimat terpilih
    summary_sentences = [sentences[i] for i in top_indices]
    summary = " ".join(summary_sentences)
    
    return summary
```

### 8.6 Contoh End-to-End Extractive

```
INPUT (10 kalimat):
"Penelitian ini bertujuan menganalisis pengaruh media sosial.        ← Kalimat 0
 Topik ini penting karena penggunaan media sosial meningkat.         ← Kalimat 1
 Metode yang digunakan adalah survei kuantitatif.                    ← Kalimat 2
 Sampel terdiri dari 200 mahasiswa universitas.                      ← Kalimat 3
 Data dikumpulkan menggunakan kuesioner online.                      ← Kalimat 4
 Hasil menunjukkan korelasi negatif antara media sosial dan IPK.     ← Kalimat 5
 Mahasiswa yang menggunakan lebih dari 4 jam memiliki IPK lebih rendah. ← Kalimat 6
 Temuan ini konsisten dengan penelitian sebelumnya.                  ← Kalimat 7
 Disarankan pembatasan penggunaan media sosial di kampus.            ← Kalimat 8
 Penelitian lanjutan perlu dilakukan dengan sampel lebih besar."     ← Kalimat 9

PROSES:
1. TF-IDF matrix: 10×73 (10 kalimat, 73 kata unik)
2. Cosine Similarity: 10×10 matrix
3. PageRank scores:
   Kalimat 0: 0.142 ← SELECTED (rank 1)
   Kalimat 1: 0.128 ← SELECTED (rank 3)
   Kalimat 5: 0.135 ← SELECTED (rank 2)
   Kalimat 6: 0.118 ← SELECTED (rank 4)
   Kalimat 8: 0.107 ← SELECTED (rank 5)
   (sisanya skor lebih rendah)

OUTPUT (5 kalimat, urutan asli):
"Penelitian ini bertujuan menganalisis pengaruh media sosial. 
 Topik ini penting karena penggunaan media sosial meningkat. 
 Hasil menunjukkan korelasi negatif antara media sosial dan IPK. 
 Mahasiswa yang menggunakan lebih dari 4 jam memiliki IPK lebih rendah. 
 Disarankan pembatasan penggunaan media sosial di kampus."
```

---

## 9. Tahap 3 — Abstractive Summarization (`src/abstractive_model.py`) — Penjelasan Detail

### 9.0 Class `SummarizationDataset` (untuk Training)

Class ini digunakan **hanya saat training** (fine-tuning) model:

```python
class SummarizationDataset(Dataset):
    """PyTorch Dataset untuk training model summarization."""
    
    def __init__(self, texts, summaries, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.texts = texts                    # List teks input
        self.summaries = summaries            # List ringkasan target (ground truth)
        self.max_source_length = max_source_length  # 256 token
        self.max_target_length = max_target_length  # 64 token
    
    def __getitem__(self, idx):
        source = self.texts[idx]        # Teks input ke-idx
        target = self.summaries[idx]    # Ringkasan target ke-idx
        
        # Tambahkan prefix "summarize: " untuk model mT5 mentah
        # (model perlu tahu tugasnya adalah merangkum)
        if "mt5" in config.ABSTRACTIVE_MODEL_NAME.lower():
            source = "summarize: " + source
        
        # Tokenisasi input: teks → list angka (token IDs)
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,  # Maks 256 token
            padding="max_length",                # Pad dengan 0 sampai 256
            truncation=True,                     # Potong jika lebih dari 256
            return_tensors="pt",                 # Return sebagai PyTorch tensor
        )
        # source_encoding["input_ids"] = [234, 567, 89, ..., 0, 0, 0]
        # source_encoding["attention_mask"] = [1, 1, 1, ..., 0, 0, 0]
        #   (1 = token asli, 0 = padding)
        
        # Tokenisasi target (ringkasan yang diinginkan)
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,  # Maks 64 token
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        labels = target_encoding["input_ids"].squeeze()
        # Ganti padding token (0) dengan -100
        # -100 adalah konvensi PyTorch: loss function mengabaikan token dengan label -100
        # Ini supaya model tidak dihukum karena tidak menebak padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }
```

### 9.1 Inisialisasi Class `AbstractiveSummarizer`

```python
class AbstractiveSummarizer:
    def __init__(self, model_name=None, max_source_length=None, 
                 max_target_length=None, num_beams=None):
        
        self.model_name = model_name or config.ABSTRACTIVE_MODEL_NAME  # "google/mt5-small"
        self.max_source_length = max_source_length or config.MAX_SOURCE_LENGTH  # 256
        self.max_target_length = max_target_length or config.MAX_TARGET_LENGTH  # 64
        self.num_beams = num_beams or config.NUM_BEAMS  # 4
        
        # Deteksi GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")   # Pakai GPU (lebih cepat)
        else:
            self.device = torch.device("cpu")    # Pakai CPU (lebih lambat)
        
        self.model = None        # Belum dimuat
        self.tokenizer = None    # Belum dimuat
```

### 9.2 Load Model — `_load_model()` dan `_load_checkpoint()`

```python
def _load_model(self):
    """Muat model pre-trained dari HuggingFace Hub (internet)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Download dan load tokenizer (vocabulary + aturan tokenisasi)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    # mT5 tokenizer menggunakan SentencePiece dengan ~250.000 subword
    
    # Download dan load model (arsitektur + bobot/weights)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    # mT5-small: 300 juta parameter, ~1.2GB
    
    # Pindahkan model ke device (GPU/CPU)
    self.model.to(self.device)

def _load_checkpoint(self, checkpoint_path):
    """Muat model yang sudah di-finetuning dari folder lokal."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Load dari folder output/checkpoints/best_model/
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    self.model.to(self.device)
    return True
```

**Perbedaan `_load_model` vs `_load_checkpoint`:**
- `_load_model`: Download dari internet (HuggingFace Hub). Model ini generik, belum dioptimalkan untuk summarization Indonesia.
- `_load_checkpoint`: Load dari disk lokal. Model ini sudah di-finetuning dengan dataset kita, hasilnya lebih baik.

### 9.3 Fine-Tuning — `fine_tune()`

```python
def fine_tune(self, train_texts, train_summaries, val_texts, val_summaries):
    """Latih ulang model dengan data kita sendiri."""
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    
    # Muat model base jika belum dimuat
    if self.model is None:
        self._load_model()
    
    # Buat dataset PyTorch dari data training & validasi
    train_dataset = SummarizationDataset(
        texts=train_texts, summaries=train_summaries,
        tokenizer=self.tokenizer,
        max_source_length=self.max_source_length,    # 256
        max_target_length=self.max_target_length,     # 64
    )
    val_dataset = SummarizationDataset(
        texts=val_texts, summaries=val_summaries,
        tokenizer=self.tokenizer,
        max_source_length=self.max_source_length,
        max_target_length=self.max_target_length,
    )
    
    # Konfigurasi training
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_CHECKPOINT_DIR,      # Simpan di output/checkpoints/
        num_train_epochs=config.NUM_EPOCHS,           # 3 epoch
        per_device_train_batch_size=config.BATCH_SIZE, # 1 (CPU mode)
        learning_rate=config.LEARNING_RATE,            # 5e-5
        weight_decay=config.WEIGHT_DECAY,              # 0.01
        eval_strategy="epoch",                         # Evaluasi setiap akhir epoch
        predict_with_generate=True,                    # Generate teks saat evaluasi
        seed=config.RANDOM_SEED,                       # 42
    )
    
    # Jalankan training
    trainer = Seq2SeqTrainer(
        model=self.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=self.tokenizer,
    )
    trainer.train()
    
    # Simpan model terbaik
    final_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model")
    trainer.save_model(final_path)
    self.tokenizer.save_pretrained(final_path)
    
    return final_path
```

**Apa yang terjadi saat fine-tuning:**

```
Epoch 1/3:
  Untuk setiap batch (1 dokumen):
    1. Forward pass: model membaca teks → generate ringkasan
    2. Hitung loss: bandingkan ringkasan generate vs ringkasan target
    3. Backward pass: hitung gradient (arah perbaikan)
    4. Update weights: sesuaikan 300 juta parameter sedikit demi sedikit
  
  Evaluasi di data validasi → catat performanya

Epoch 2/3:
  Ulangi proses di atas (model melihat semua data training lagi)
  Model semakin baik karena melihat data dua kali

Epoch 3/3:
  Ulangi lagi → model sudah cukup baik
  Simpan model ke output/checkpoints/best_model/
```

### 9.4 Inference (Generate Ringkasan) — `summarize(text)`

```python
def summarize(self, text: str) -> str:
    """Generate ringkasan untuk satu dokumen."""
    
    # 1. Tambahkan prefix tugas (untuk model mT5 mentah)
    if "mt5" in self.model_name.lower():
        text = "summarize: " + text
    # Contoh: "summarize: Penelitian ini bertujuan menganalisis..."
    
    # 2. TOKENISASI — Ubah teks jadi angka (token IDs)
    inputs = self.tokenizer(
        text,
        max_length=self.max_source_length,  # Maks 256 token
        padding="max_length",               # Tambahkan padding
        truncation=True,                    # Potong jika terlalu panjang
        return_tensors="pt",                # Return PyTorch tensor
    ).to(self.device)
    # inputs["input_ids"] = tensor([234, 1567, 89, 4523, ..., 0, 0, 0])
    # inputs["attention_mask"] = tensor([1, 1, 1, 1, ..., 0, 0, 0])
    
    # 3. GENERATE — Buat ringkasan
    self.model.eval()              # Set mode evaluasi (bukan training)
    with torch.no_grad():          # Tidak perlu hitung gradient
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_target_length,    # Maks 64 token
            num_beams=self.num_beams,              # 4 beam search
            early_stopping=True,                   # Berhenti jika semua beam selesai
            no_repeat_ngram_size=3,                # Cegah pengulangan 3-gram
        )
    # generated_ids = tensor([0, 234, 5678, 91, 234, 1])
    # Ini adalah TOKEN IDS, belum bisa dibaca manusia
    
    # 4. DECODE — Ubah angka kembali jadi teks
    summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # skip_special_tokens=True: hapus token <pad>, </s>, dll
    # summary = "Penelitian menunjukkan pengaruh negatif media sosial terhadap IPK"
    
    return summary
```

**Penjelasan Beam Search (num_beams=4):**

Tanpa beam search (greedy), model selalu pilih kata dengan probabilitas tertinggi di setiap langkah. Ini bisa menghasilkan output yang suboptimal.

Dengan beam search (num_beams=4), model menjajaki **4 jalur** secara paralel:

```
Step 1: Generate token pertama
  Beam 1: "Penelitian"  (prob: 0.35)
  Beam 2: "Hasil"       (prob: 0.25)
  Beam 3: "Media"       (prob: 0.20)
  Beam 4: "Studi"       (prob: 0.15)

Step 2: Untuk SETIAP beam, generate token kedua
  Beam 1: "Penelitian menunjukkan"  (prob: 0.35 × 0.40 = 0.14)
  Beam 2: "Penelitian ini"          (prob: 0.35 × 0.30 = 0.105)
  Beam 3: "Hasil menunjukkan"       (prob: 0.25 × 0.45 = 0.1125)
  Beam 4: "Hasil penelitian"        (prob: 0.25 × 0.35 = 0.0875)
  ... (pilih 4 terbaik dari semua kombinasi)

Step 3-N: Ulangi sampai max_length atau semua beam generate </s>

Final: Pilih beam dengan total probabilitas tertinggi
```

### 9.5 Batch Summarize — `batch_summarize(texts)`

```python
def batch_summarize(self, texts: List[str]) -> List[str]:
    """Generate ringkasan untuk banyak dokumen sekaligus."""
    from tqdm import tqdm
    
    summaries = []
    for text in tqdm(texts, desc="Abstractive Summarization", unit="doc"):
        # Proses satu per satu (bukan batch) untuk menghemat memori CPU
        summary = self.summarize(text)
        summaries.append(summary)
    
    return summaries
```

---

## 10. Tahap 4 — Evaluasi ROUGE (`src/evaluator.py`) — Penjelasan Detail

### 10.1 Inisialisasi Evaluator

```python
class Evaluator:
    def __init__(self, rouge_metrics=None, results_dir=None, summaries_dir=None):
        self.rouge_metrics = rouge_metrics or config.ROUGE_METRICS
        # ["rouge1", "rouge2", "rougeL"]
        
        # Buat ROUGE scorer dari library rouge-score
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_metrics,
            use_stemmer=False  # Tidak pakai stemmer bawaan (sudah stemming sendiri)
        )
```

### 10.2 Cara Kerja ROUGE — Dijelaskan dengan Contoh Nyata

#### ROUGE-1 (Unigram — kata tunggal)

```
Referensi:  "kucing hitam duduk di atas tikar merah"
Ringkasan:  "kucing duduk di atas tikar biru"

Kata di referensi:  {kucing, hitam, duduk, di, atas, tikar, merah}     → 7 kata
Kata di ringkasan:  {kucing, duduk, di, atas, tikar, biru}             → 6 kata
Kata yang COCOK:    {kucing, duduk, di, atas, tikar}                   → 5 kata

Precision = kata_cocok / total_kata_ringkasan = 5/6 = 0.8333 (83.33%)
  → "Dari 6 kata yang saya tulis, 5 di antaranya benar ada di referensi"

Recall = kata_cocok / total_kata_referensi = 5/7 = 0.7143 (71.43%)
  → "Dari 7 kata di referensi, 5 di antaranya berhasil saya tangkap"

F1 = 2 × (Precision × Recall) / (Precision + Recall) 
   = 2 × (0.8333 × 0.7143) / (0.8333 + 0.7143)
   = 1.1904 / 1.5476
   = 0.7692 (76.92%)
```

#### ROUGE-2 (Bigram — pasangan kata berurutan)

```
Referensi:  "kucing hitam duduk di atas tikar merah"
Bigram referensi:  {(kucing,hitam), (hitam,duduk), (duduk,di), 
                    (di,atas), (atas,tikar), (tikar,merah)}       → 6 bigram

Ringkasan:  "kucing duduk di atas tikar biru"
Bigram ringkasan:  {(kucing,duduk), (duduk,di), (di,atas), 
                    (atas,tikar), (tikar,biru)}                    → 5 bigram

Bigram COCOK:      {(duduk,di), (di,atas), (atas,tikar)}          → 3 bigram

Precision = 3/5 = 0.60 (60%)
Recall    = 3/6 = 0.50 (50%)
F1        = 2 × (0.60 × 0.50) / (0.60 + 0.50) = 0.5455 (54.55%)
```

ROUGE-2 lebih ketat dari ROUGE-1 karena harus cocok **urutan** 2 kata berturut-turut.

#### ROUGE-L (Longest Common Subsequence)

```
Referensi:  "kucing hitam duduk di atas tikar merah"
Ringkasan:  "kucing duduk di atas tikar biru"

LCS (urutan terpanjang yang cocok, tidak harus bersebelahan):
  "kucing" → "duduk" → "di" → "atas" → "tikar"
  LCS length = 5

Precision = LCS / panjang_ringkasan = 5/6 = 0.8333
Recall    = LCS / panjang_referensi = 5/7 = 0.7143
F1        = 0.7692

(Dalam kasus ini, ROUGE-L sama dengan ROUGE-1 karena LCS = jumlah unigram cocok)
```

ROUGE-L mengukur **urutan kata** — bukan hanya kata yang sama, tapi apakah urutannya juga mirip.

### 10.3 Compute ROUGE — Source Code Lengkap

```python
def compute_rouge(self, predictions: List[str], references: List[str]):
    """Hitung rata-rata skor ROUGE untuk semua dokumen."""
    
    # Validasi: jumlah prediksi harus sama dengan jumlah referensi
    if len(predictions) != len(references):
        raise ValueError("Jumlah prediksi dan referensi tidak sama!")
    
    # Siapkan akumulator skor
    all_scores = {
        metric: {"precision": [], "recall": [], "fmeasure": []}
        for metric in self.rouge_metrics
    }
    # all_scores = {
    #   "rouge1": {"precision": [], "recall": [], "fmeasure": []},
    #   "rouge2": {"precision": [], "recall": [], "fmeasure": []},
    #   "rougeL": {"precision": [], "recall": [], "fmeasure": []},
    # }
    
    # Hitung skor untuk setiap pasangan (prediksi, referensi)
    for pred, ref in zip(predictions, references):
        # Handle string kosong
        pred = pred if pred and pred.strip() else " "
        ref = ref if ref and ref.strip() else " "
        
        # Hitung ROUGE! (satu baris ini melakukan semua perhitungan)
        scores = self.scorer.score(ref, pred)
        # scores["rouge1"].precision = 0.83
        # scores["rouge1"].recall = 0.71
        # scores["rouge1"].fmeasure = 0.77
        # ... (sama untuk rouge2 dan rougeL)
        
        # Kumpulkan skor ke akumulator
        for metric in self.rouge_metrics:
            all_scores[metric]["precision"].append(scores[metric].precision)
            all_scores[metric]["recall"].append(scores[metric].recall)
            all_scores[metric]["fmeasure"].append(scores[metric].fmeasure)
    
    # Hitung rata-rata dari semua dokumen
    avg_scores = {}
    for metric in self.rouge_metrics:
        avg_scores[metric] = {
            "precision": sum(all_scores[metric]["precision"]) / len(predictions),
            "recall": sum(all_scores[metric]["recall"]) / len(predictions),
            "fmeasure": sum(all_scores[metric]["fmeasure"]) / len(predictions),
        }
    
    return avg_scores
    # Contoh output:
    # {
    #   "rouge1": {"precision": 0.45, "recall": 0.38, "fmeasure": 0.41},
    #   "rouge2": {"precision": 0.20, "recall": 0.17, "fmeasure": 0.18},
    #   "rougeL": {"precision": 0.40, "recall": 0.33, "fmeasure": 0.36},
    # }
```

### 10.4 Cara Membaca Hasil ROUGE

| Rentang F1 | Interpretasi | Contoh |
|---|---|---|
| **> 0.50 (50%)** | Sangat bagus — ringkasan sangat mirip referensi | Extractive pada teks pendek |
| **0.30 — 0.50** | Cukup bagus — ada overlap signifikan | Abstractive pada teks medium |
| **0.15 — 0.30** | Kurang — mirip tapi banyak yang berbeda | Teks sangat panjang |
| **< 0.15** | Buruk — ringkasan jauh dari referensi | Model belum di-train |

**Catatan penting:**
- ROUGE **bukan** ukuran kualitas sempurna — ringkasan bisa bagus tapi punya kata-kata berbeda dari referensi
- Yang paling penting adalah **perbandingan relatif** antara extractive vs abstractive
- ROUGE-2 biasanya lebih rendah dari ROUGE-1 (karena lebih ketat)

---

## 11. File `src/data_loader.py` — Memuat Dataset

```python
class DataLoader:
    """Memuat, memvalidasi, dan membagi dataset."""
    
    def __init__(self):
        self.dataset_path = config.DATASET_PATH        # data/raw/dataset.csv
        self.text_column = config.TEXT_COLUMN           # "full_text"
        self.summary_column = config.SUMMARY_COLUMN    # "summary"
        self.train_ratio = config.TRAIN_RATIO           # 0.8 (80%)
        self.val_ratio = config.VAL_RATIO               # 0.1 (10%)
        self.test_ratio = config.TEST_RATIO             # 0.1 (10%)
    
    def load_dataset(self):
        """Baca file CSV/JSON."""
        df = pd.read_csv(self.dataset_path)
        return df
    
    def validate_columns(self, df):
        """Cek apakah kolom 'full_text' dan 'summary' ada."""
        required = [self.text_column, self.summary_column]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan: {missing}")
    
    def clean_dataset(self, df):
        """Hapus baris dengan teks/summary kosong atau null."""
        df = df.dropna(subset=[self.text_column, self.summary_column])
        df = df[df[self.text_column].str.strip().str.len() > 0]
        df = df[df[self.summary_column].str.strip().str.len() > 0]
        return df
    
    def split_dataset(self, df):
        """Bagi dataset menjadi train/val/test."""
        # Dari 20 dokumen:
        # Train: 16 dokumen (80%)
        # Val:    2 dokumen (10%)
        # Test:   2 dokumen (10%)
```

**Fungsi file ini:** Digunakan oleh `main.py` saat menjalankan pipeline dari command line. Web app (`app.py`) tidak menggunakan `DataLoader` karena data sudah dikirim langsung dari frontend.

---

## 12. File `src/utils.py` — Fungsi Pembantu

```python
def setup_logging(level="INFO"):
    """Konfigurasi sistem logging agar semua module punya format seragam."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Contoh output log:
    # 2026-04-09 10:30:15 [INFO] src.preprocessor: Stopword list loaded with 759 words

def set_seed(seed=42):
    """Set random seed untuk semua library agar hasil reproducible."""
    random.seed(seed)       # Python random
    np.random.seed(seed)    # NumPy random
    # Jika ada PyTorch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(data, path):
    """Simpan dictionary/list ke file JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_text_stats(texts):
    """Hitung statistik teks (panjang rata-rata, min, max, dll)."""
```

---

## 13. File `main.py` — Pipeline CLI (Command Line)

File ini digunakan untuk menjalankan pipeline **tanpa web interface** — langsung dari terminal:

```python
# Cara pakai:
# python main.py                              → Pipeline penuh
# python main.py --mode train                 → Training model saja
# python main.py --mode evaluate              → Evaluasi saja
# python main.py --model extractive           → Extractive saja
# python main.py --model abstractive          → Abstractive saja
# python main.py --data path/to/data.csv      → Pakai dataset lain

def main():
    args = parse_arguments()
    
    # 1. LOAD DATA
    loader = DataLoader(dataset_path=args.data)
    df = loader.load_dataset()
    df = loader.clean_dataset(df)
    train_df, val_df, test_df = loader.split_dataset(df)
    
    # Ambil teks dan summary dari setiap split
    test_texts = test_df["full_text"].tolist()
    test_summaries = test_df["summary"].tolist()
    
    # 2. EXTRACTIVE SUMMARIZATION
    if args.model in ("extractive", "both"):
        ext_summaries = run_extractive(test_texts)
    
    # 3. ABSTRACTIVE SUMMARIZATION
    if args.model in ("abstractive", "both"):
        abs_summaries = run_abstractive(
            train_texts, train_summaries,  # Untuk training
            val_texts, val_summaries,       # Untuk validasi
            test_texts,                     # Untuk generate
            mode=args.mode
        )
    
    # 4. EVALUASI
    run_evaluation(ext_summaries, abs_summaries, test_summaries)
    # → Print tabel perbandingan ROUGE
    # → Simpan hasil ke output/results/
```

---

## 14. Frontend — `templates/index.html`

Single page web application (~800 baris) yang menampilkan seluruh pipeline secara visual dan interaktif.

### 14.1 Teknologi Frontend

| Teknologi | Kegunaan | Cara Load |
|---|---|---|
| **Tailwind CSS** | Styling (CSS utility classes) | CDN (`cdn.tailwindcss.com`) |
| **Lucide Icons** | Icon SVG | CDN (`unpkg.com/lucide`) |
| **Vanilla JavaScript** | Logic & interaksi | Inline `<script>` |
| **Chart.js** | (Tidak digunakan) | - |

### 14.2 State Management

Frontend menyimpan semua data di satu objek JavaScript:

```javascript
let appState = {
    texts: [],           // Array teks dokumen (dari upload/manual)
    summaries: [],       // Array ringkasan referensi
    currentDocIndex: 0,  // Indeks dokumen yang sedang ditampilkan
    currentTab: 'upload',// Tab aktif (upload/manual)
    file: null,          // File yang di-upload
    preprocessResult: null,   // Hasil dari /api/preprocess
    summarizeResult: null,    // Hasil dari /api/summarize
    evaluateResult: null,     // Hasil dari /api/evaluate
};
```

### 14.3 Alur JavaScript

```javascript
async function startPipeline() {
    // 1. Validasi input
    if (appState.texts.length === 0) return showToast("Upload dulu!");
    
    // 2. Tampilkan section hasil
    document.getElementById('results-section').classList.remove('hidden');
    
    // 3. Jalankan pipeline berurutan
    await doPreprocessing();   // Panggil /api/preprocess
    updateStepper(2);          // Update stepper ke langkah 2
    
    await doSummarization();   // Panggil /api/summarize
    updateStepper(3);          // Update stepper ke langkah 3
    
    await doEvaluation();      // Panggil /api/evaluate
}
```

### 14.4 Fitur Visual di Frontend

1. **Stepper Progress** — Lingkaran 1-2-3 yang menunjukkan langkah aktif
2. **Stats Dashboard** — 4 kartu statistik (jumlah kalimat, token awal, setelah stemming, % reduksi)
3. **Accordion Steps** — 6 panel preprocessing yang bisa dibuka/tutup
4. **Source Code Display** — Tombol "Lihat Source Code" di setiap langkah (dengan syntax highlighting Python)
5. **"Apa Yang Terjadi"** — Panel penjelasan kuning di setiap langkah
6. **Comparison View** — Side-by-side extractive vs abstractive
7. **Bar Chart ROUGE** — Visualisasi skor ROUGE dengan progress bar
8. **Winner Badge** — Menampilkan metode terbaik dengan skor rata-rata

---

## 15. Dataset yang Digunakan

### 15.1 Sumber Dataset

**XL-Sum (Cross-Lingual Summary)** dari `csebuetnlp/xlsum` di HuggingFace.

- **Bahasa:** Indonesia (dari koleksi 44 bahasa)
- **Asal data:** Artikel berita BBC Indonesian
- **Total corpus:** 38.242 artikel Indonesia
- **Yang kita gunakan:** 20 artikel (difilter)

### 15.2 Kriteria Seleksi

Dari 38.242 artikel, dipilih 20 dengan kriteria:
- Panjang teks > 800 karakter (artikel cukup substansial)
- Panjang ringkasan 100-800 karakter (ringkasan referensi cukup lengkap)

### 15.3 Format File

File: `data/raw/dataset.csv`

```csv
full_text,summary
"Teks artikel lengkap pertama...","Ringkasan referensi pertama..."
"Teks artikel lengkap kedua...","Ringkasan referensi kedua..."
...
```

| Kolom | Isi | Rata-rata Panjang |
|---|---|---|
| `full_text` | Teks lengkap artikel berita | ~4.000 karakter |
| `summary` | Ringkasan referensi buatan manusia | ~175 karakter |

### 15.4 Cara Mengganti Dataset

**Opsi 1:** Ganti file CSV
```bash
# Siapkan file CSV dengan kolom full_text dan summary
cp dataset_baru.csv data/raw/dataset.csv
python app.py  # Jalankan ulang server
```

**Opsi 2:** Upload via web
```
Buka browser → Drag & drop file CSV ke area upload
```

**Opsi 3:** Via CLI
```bash
python main.py --data path/ke/file_baru.csv
```

---

## 16. Cara Menjalankan Project

### 16.1 Persiapan Awal (Satu Kali Saja)

```bash
# 1. Clone repository (jika belum)
git clone https://github.com/devnolife/academic-text-summarization-id.git
cd academic-text-summarization-id

# 2. Install semua library Python
pip install -r requirements.txt
# Ini akan menginstall: flask, torch, transformers, nltk, sastrawi, dll
# Estimasi: ~2-5GB (karena PyTorch dan Transformers cukup besar)

# 3. Download resource NLTK (otomatis saat pertama dijalankan)
# Tidak perlu manual — preprocessor.py akan download otomatis
```

### 16.2 Menjalankan Web App

```bash
python app.py
```

Output:
```
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:3000
```

Buka browser → `http://localhost:3000`

### 16.3 Menggunakan Web App (Step by Step)

```
1. Buka http://localhost:3000

2. PILIH INPUT:
   a. Klik "Atau gunakan dataset contoh" → load 20 artikel default
   b. Atau drag & drop file CSV kamu sendiri
   c. Atau klik tab "Input Manual" dan paste teks

3. KLIK "Mulai Proses Summarization"

4. TUNGGU:
   - Step 1 (Preprocessing): ~5-10 detik untuk 20 dokumen
   - Step 2 (Summarization): 
     • Extractive: ~3-5 detik
     • Abstractive: ~60-300 detik (tergantung CPU)
       ⚠ PERTAMA KALI abstractive akan lambat karena harus load model ~1.2GB
   - Step 3 (Evaluasi): ~1-2 detik

5. HASIL:
   - Lihat preprocessing per tahap (klik accordion untuk expand)
   - Lihat ringkasan extractive vs abstractive
   - Lihat skor ROUGE (bar chart)
   - Lihat kesimpulan: metode mana yang lebih baik
```

### 16.4 Menjalankan Pipeline CLI

```bash
# Pipeline lengkap (training + evaluasi)
python main.py --mode full --model both

# Training model abstractive saja
python main.py --mode train --model abstractive

# Evaluasi saja (model harus sudah ada di output/checkpoints/best_model/)
python main.py --mode evaluate

# Hanya extractive (tanpa abstractive)
python main.py --model extractive

# Gunakan dataset custom
python main.py --data path/to/custom_dataset.csv
```

### 16.5 Deploy ke Production Server

```bash
# Jalankan dengan Gunicorn (production WSGI server)
gunicorn -c gunicorn_config.py app:app

# Atau jalankan setup script
bash deploy/setup.sh
```

---

## 17. API Endpoints — Request & Response Lengkap

### 17.1 `GET /` — Halaman Web

Mengembalikan file `templates/index.html`.

### 17.2 `POST /api/upload` — Upload Dataset

**Request:**
```
Content-Type: multipart/form-data
Body: file = [file CSV/JSON]
```

**Response (sukses):**
```json
{
  "success": true,
  "num_documents": 20,
  "texts": ["Teks dokumen 1...", "Teks dokumen 2...", ...],
  "summaries": ["Ringkasan 1...", "Ringkasan 2...", ...],
  "preview": [
    {"text": "Teks (maks 200 char)...", "summary": "Ringkasan (maks 150 char)..."},
    ...
  ],
  "stats": {
    "avg_text_length": 4041,
    "avg_summary_length": 175
  }
}
```

**Response (error):**
```json
{"error": "Column 'full_text' not found. Available: ['text', 'abstract']"}
```

### 17.3 `GET/POST /api/load-default` — Load Dataset Default

**Request:** Tidak perlu body.

**Response:** Sama seperti `/api/upload` — berisi teks dan summary dari `data/raw/dataset.csv`.

### 17.4 `POST /api/preprocess` — Preprocessing

**Request:**
```json
{
  "texts": [
    "PENELITIAN ini bertujuan untuk MENGANALISIS pengaruh https://example.com media sosial [1].",
    "Teks dokumen kedua..."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "original": "PENELITIAN ini bertujuan untuk MENGANALISIS pengaruh https://example.com media sosial [1].",
      "case_folding": "penelitian ini bertujuan untuk menganalisis pengaruh https://example.com media sosial [1].",
      "cleaning": "penelitian ini bertujuan untuk menganalisis pengaruh media sosial .",
      "sentences": ["penelitian ini bertujuan untuk menganalisis pengaruh media sosial ."],
      "word_tokens": ["penelitian", "ini", "bertujuan", "untuk", "menganalisis", "pengaruh", "media", "sosial", "."],
      "after_stopword_removal": ["penelitian", "bertujuan", "menganalisis", "pengaruh", "media", "sosial"],
      "after_stemming": ["teliti", "tuju", "analisis", "pengaruh", "media", "sosial"],
      "num_sentences": 1,
      "num_tokens_before": 9,
      "num_tokens_after": 6,
      "num_tokens_stemmed": 6
    }
  ]
}
```

### 17.5 `POST /api/summarize` — Summarization

**Request:**
```json
{
  "texts": ["Teks dokumen lengkap 1...", "Teks dokumen lengkap 2..."]
}
```

**Response:**
```json
{
  "success": true,
  "extractive": [
    "Kalimat terpilih 1. Kalimat terpilih 2. Kalimat terpilih 3.",
    "Ringkasan extractive dokumen 2..."
  ],
  "abstractive": [
    "Ringkasan baru yang digenerate AI untuk dokumen 1.",
    "Ringkasan baru untuk dokumen 2."
  ],
  "extractive_details": [
    {
      "sentences": ["kalimat 1", "kalimat 2", ...],
      "num_sentences": 15,
      "tfidf_shape": [15, 73],
      "num_features": 73,
      "avg_similarity": 0.1234,
      "max_similarity": 0.5678,
      "top_pairs": [{"i": 0, "j": 3, "sim": 0.5678}, ...],
      "sentence_scores": [
        {"index": 0, "sentence": "kalimat 1...", "score": 0.142, "rank": 1, "selected": true},
        {"index": 1, "sentence": "kalimat 2...", "score": 0.098, "rank": 6, "selected": false},
        ...
      ],
      "selected_indices": [0, 2, 5, 8, 12],
      "num_selected": 5,
      "summary": "Kalimat terpilih gabungan..."
    }
  ],
  "abstractive_details": {
    "model_name": "google/mt5-small",
    "device": "cpu",
    "max_source_length": 256,
    "max_target_length": 64,
    "num_beams": 4,
    "num_parameters": "300,000,000"
  }
}
```

### 17.6 `POST /api/evaluate` — Evaluasi ROUGE

**Request:**
```json
{
  "references": ["Ringkasan referensi 1...", "Ringkasan referensi 2..."],
  "extractive_preds": ["Hasil extractive 1...", "Hasil extractive 2..."],
  "abstractive_preds": ["Hasil abstractive 1...", "Hasil abstractive 2..."]
}
```

**Response:**
```json
{
  "success": true,
  "per_document": [
    {
      "doc_id": 0,
      "reference_preview": "Ringkasan referensi 1 (150 char)...",
      "extractive_preview": "Hasil extractive 1 (150 char)...",
      "abstractive_preview": "Hasil abstractive 1 (150 char)...",
      "ext": {
        "rouge1": {"p": 0.4523, "r": 0.3812, "f": 0.4138},
        "rouge2": {"p": 0.2041, "r": 0.1724, "f": 0.1869},
        "rougeL": {"p": 0.4012, "r": 0.3387, "f": 0.3673}
      },
      "abs": {
        "rouge1": {"p": 0.3856, "r": 0.4231, "f": 0.4035},
        "rouge2": {"p": 0.1538, "r": 0.1692, "f": 0.1612},
        "rougeL": {"p": 0.3462, "r": 0.3846, "f": 0.3644}
      }
    }
  ],
  "extractive_scores": {
    "rouge1": {"precision": 0.45, "recall": 0.38, "fmeasure": 0.41},
    "rouge2": {"precision": 0.20, "recall": 0.17, "fmeasure": 0.18},
    "rougeL": {"precision": 0.40, "recall": 0.34, "fmeasure": 0.37}
  },
  "abstractive_scores": {
    "rouge1": {"precision": 0.39, "recall": 0.42, "fmeasure": 0.40},
    "rouge2": {"precision": 0.15, "recall": 0.17, "fmeasure": 0.16},
    "rougeL": {"precision": 0.35, "recall": 0.38, "fmeasure": 0.36}
  },
  "extractive_avg_f1": 0.3200,
  "abstractive_avg_f1": 0.3067,
  "best_method": "Extractive"
}
```

---

## 18. Library yang Digunakan (Lengkap dengan Penjelasan)

### 18.1 NLP & Machine Learning

| Library | Versi | Kegunaan Detail |
|---|---|---|
| **nltk** | ≥3.8 | **Natural Language Toolkit.** Digunakan untuk tokenisasi kalimat (`sent_tokenize`) dan kata (`word_tokenize`) dengan model bahasa Indonesia. Juga menyediakan daftar 759 stopword Indonesia. |
| **PySastrawi** | ≥1.2 | **Stemmer Bahasa Indonesia** berdasarkan algoritma Nazief & Adriani. Punya kamus 28.000+ kata dasar. Mengubah "pembelajaran" → "ajar", "berlari" → "lari". |
| **scikit-learn** | ≥1.2 | **Machine learning toolkit.** Kita pakai `TfidfVectorizer` untuk mengubah kalimat jadi vektor TF-IDF, dan `cosine_similarity` untuk menghitung kemiripan antar kalimat. |
| **networkx** | ≥3.0 | **Library graph/network.** Kita pakai untuk membangun graph kemiripan kalimat (`from_numpy_array`) dan menjalankan algoritma PageRank (`nx.pagerank`). |
| **transformers** | ≥4.30 | **HuggingFace Transformers.** Library untuk memuat dan menjalankan model NLP transformer. Kita pakai `AutoModelForSeq2SeqLM` (model mT5) dan `AutoTokenizer`. Juga `Seq2SeqTrainer` untuk fine-tuning. |
| **torch** | ≥2.0 | **PyTorch.** Framework deep learning dari Meta/Facebook. Digunakan oleh Transformers untuk operasi tensor, forward/backward pass, dan gradient descent saat training. |
| **sentencepiece** | ≥0.1.99 | **Tokenizer subword** dari Google. Model mT5 menggunakan SentencePiece untuk memecah teks menjadi subword (potongan kata). Vocabulary: ~250.000 subword untuk 101 bahasa. |
| **rouge-score** | ≥0.1.2 | **ROUGE metric calculator** dari Google Research. Menghitung ROUGE-1, ROUGE-2, dan ROUGE-L antara ringkasan prediksi dan referensi. |
| **accelerate** | ≥0.20 | **HuggingFace Accelerate.** Membantu menjalankan PyTorch di berbagai device (CPU/GPU/TPU) tanpa perlu mengubah kode. |

### 18.2 Data & Utilitas

| Library | Versi | Kegunaan Detail |
|---|---|---|
| **pandas** | ≥1.5 | **Data manipulation.** Membaca file CSV (`pd.read_csv`), membersihkan data (`dropna`), dan manipulasi DataFrame. |
| **numpy** | ≥1.23 | **Numerical computing.** Operasi matriks untuk TF-IDF dan cosine similarity. Juga digunakan untuk indexing dan slicing array. |
| **tqdm** | ≥4.65 | **Progress bar.** Menampilkan progress saat batch processing (misal: "Abstractive Summarization: 5/20 [25%]"). |
| **datasets** | ≥2.12 | **HuggingFace Datasets.** Digunakan untuk mendownload dataset XL-Sum dari HuggingFace Hub. |
| **protobuf** | ≥3.20 | **Protocol Buffers.** Dependency dari SentencePiece dan Transformers. |

### 18.3 Web

| Library | Versi | Kegunaan Detail |
|---|---|---|
| **flask** | ≥3.0 | **Web framework.** Menjalankan server HTTP, routing (URL → fungsi), dan mengembalikan JSON response. |
| **flask-cors** | ≥4.0 | **CORS support.** Mengizinkan browser dari domain lain mengakses API kita (penting untuk development). |
| **gunicorn** | ≥21.2 | **Production server.** Menjalankan Flask app dengan multiple worker processes untuk menangani request secara parallel. Lebih stabil dan cepat dari Flask development server. |

### 18.4 Frontend (CDN, tidak perlu install)

| Library | Versi | Kegunaan Detail |
|---|---|---|
| **Tailwind CSS** | Latest | **Utility-first CSS framework.** Setiap class CSS melakukan satu hal: `text-sm` (font kecil), `bg-blue-500` (background biru), `rounded-xl` (border radius), dll. Tidak menulis CSS kustom. |
| **Lucide Icons** | Latest | **Icon library.** 1000+ icon SVG yang ringan. Contoh: `<i data-lucide="file-text">` menampilkan icon file teks. |

---

## 19. FAQ / Pertanyaan Umum

### "Apa beda extractive dan abstractive?"

**Extractive** = copy-paste kalimat terpenting dari teks asli. Kata-kata TIDAK berubah.
**Abstractive** = AI membaca teks lalu menulis ringkasan baru pakai kata-kata sendiri. Kalimat BARU yang mungkin tidak ada di teks asli.

### "Kenapa pakai dua metode? Bukankah satu cukup?"

Tujuan penelitian ini adalah **membandingkan** kedua pendekatan secara objektif. Dengan membandingkan skor ROUGE, kita bisa menjawab: "Untuk teks akademik Bahasa Indonesia, pendekatan mana yang lebih efektif?"

### "Apa itu ROUGE F1-Score dan kenapa itu yang paling penting?"

F1-Score menggabungkan **precision** (ketepatan) dan **recall** (kelengkapan) menjadi satu angka. Skor 0.0 = tidak mirip sama sekali, 1.0 = persis sama dengan referensi. F1 digunakan sebagai skor utama karena menyeimbangkan kedua aspek.

### "Apa itu TF-IDF? Jelaskan sesederhana mungkin."

Bayangkan kamu baca 10 paper:
- Kata "dan" muncul di semua 10 paper → **tidak penting** (TF-IDF rendah)
- Kata "transformer" muncul di 2 paper tapi sering di paper itu → **penting** (TF-IDF tinggi)
- Kata "kuantum" muncul di 1 paper saja → **sangat penting** untuk paper itu (TF-IDF sangat tinggi)

### "Apa itu PageRank? Jelaskan dengan analogi."

Bayangkan pemilihan ketua kelas:
- Setiap siswa bisa "memilih" siswa lain yang menurutnya layak jadi ketua
- Suara dari siswa populer (yang dipilih banyak orang) **lebih berbobot** dari suara siswa biasa
- Siswa yang mendapat banyak suara dari siswa-siswa populer → terpilih sebagai ketua

Di project ini: kalimat = siswa, kemiripan = "suara", kalimat dengan skor PageRank tertinggi = "ketua" (paling penting).

### "Kenapa model abstractive lambat?"

Model mT5 punya ~300 juta parameter. Setiap kali generate satu token (satu potongan kata), model harus melakukan jutaan operasi perkalian matriks. Di CPU tanpa GPU, ini bisa memakan 10-30 detik per dokumen. Di GPU NVIDIA, bisa 10× lebih cepat.

### "Kenapa pertama kali abstractive sangat lambat?"

Karena pertama kali harus **memuat model** dari disk ke RAM (~1.2GB). Setelah dimuat, request berikutnya jauh lebih cepat karena model sudah ada di memori.

### "Mau ganti dataset, gimana?"

1. Siapkan file CSV dengan 2 kolom: `full_text` (teks lengkap) dan `summary` (ringkasan referensi)
2. Ganti file di `data/raw/dataset.csv`, ATAU
3. Upload lewat web (drag & drop), ATAU
4. CLI: `python main.py --data path/to/file.csv`

### "Mau ganti model abstractive?"

Edit `config.py`, baris `ABSTRACTIVE_MODEL_NAME`:
```python
# Pilihan:
ABSTRACTIVE_MODEL_NAME = "google/mt5-small"                       # Pre-trained (perlu fine-tuning)
ABSTRACTIVE_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum-small" # Sudah fine-tuned untuk summarization
ABSTRACTIVE_MODEL_NAME = "LazarusNLP/IndoNanoT5-base"              # Khusus Indonesia
```
Setelah ganti, hapus folder `output/checkpoints/best_model/` dan jalankan training ulang: `python main.py --mode train`.

### "File apa saja yang perlu saya baca untuk paham kode?"

Baca **dalam urutan ini** (dari yang paling mudah):

1. **`config.py`** — Pahami semua pengaturan (5 menit)
2. **`src/preprocessor.py`** — Pahami 6 tahap preprocessing (15 menit)
3. **`src/extractive_model.py`** — Pahami TF-IDF + PageRank (20 menit)
4. **`src/evaluator.py`** — Pahami ROUGE (10 menit)
5. **`src/abstractive_model.py`** — Pahami transformer/mT5 (30 menit, paling kompleks)
6. **`app.py`** — Pahami bagaimana semuanya terhubung lewat API (15 menit)
7. **`templates/index.html`** — Pahami frontend jika perlu (opsional)

### "Kenapa `preprocess_for_extractive` berbeda dari `preprocess` biasa?"

Extractive summarization **memilih kalimat asli** utuh — jadi kalimatnya HARUS tetap bisa dibaca manusia. Jika kita lakukan stopword removal dan stemming, kalimatnya rusak:
```
Kalimat asli:    "Penelitian ini bertujuan untuk menganalisis pengaruh media sosial"
Setelah stem:    "teliti tuju analisis pengaruh media sosial"  ← RUSAK, tidak bisa dibaca
```
Maka untuk extractive, cukup **case folding + cleaning + tokenisasi kalimat** saja.

### "Apa itu lazy loading dan kenapa abstractive model di-lazy load?"

Lazy loading = **dimuat hanya saat dibutuhkan pertama kali**, bukan saat server startup.

Model mT5 berukuran ~1.2GB. Jika di-load saat startup:
- Server butuh 30-60 detik untuk siap ❌
- RAM langsung terpakai 1.2GB meskipun user belum tentu butuh abstractive ❌

Dengan lazy loading:
- Server siap dalam 2-3 detik ✅
- Model di-load hanya jika user memang menjalankan summarization ✅
- Setelah di-load pertama kali, model tetap di memori untuk request selanjutnya ✅
