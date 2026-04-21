# Pedoman Project: NLP Text Summarization Pipeline

> **Panduan lengkap dan mendetail untuk memahami seluruh project Summarization Teks Bahasa Indonesia — dari konsep dasar hingga implementasi kode.**
> Dibuat oleh: Ainur Ridha Surya (D082251037), Nirwana Samrin (D082251003), Andi Agung Dwi Arya B (D082251057)

---

## Daftar Isi

1. [Apa Itu Project Ini?](#1-apa-itu-project-ini)
2. [Konsep Dasar yang Harus Dipahami](#2-konsep-dasar-yang-harus-dipahami)
3. [Alur Kerja Pipeline dari Awal Sampai Akhir](#3-alur-kerja-pipeline-dari-awal-sampai-akhir)
4. [Konfigurasi NLP (`config.py`)](#4-konfigurasi-nlp-configpy)
5. [Tahap 1 — Preprocessing (`src/preprocessor.py`)](#5-tahap-1--preprocessing-srcpreprocessorpy)
6. [Tahap 2 — Extractive Summarization (`src/extractive_model.py`)](#6-tahap-2--extractive-summarization-srcextractive_modelpy)
7. [Tahap 3 — Abstractive Summarization (`src/abstractive_model.py`)](#7-tahap-3--abstractive-summarization-srcabstractive_modelpy)
8. [Tahap 4 — Evaluasi ROUGE (`src/evaluator.py`)](#8-tahap-4--evaluasi-rouge-srcevaluatorpy)
9. [Dataset yang Digunakan](#9-dataset-yang-digunakan)
10. [Library NLP yang Digunakan](#10-library-nlp-yang-digunakan)
11. [FAQ / Pertanyaan Umum](#11-faq--pertanyaan-umum)

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

### 1.4 Teknologi NLP yang Digunakan

- **Extractive**: Algoritma TextRank (berbasis TF-IDF + PageRank dari Google)
- **Abstractive**: Model Transformer mT5 dari Google (deep learning)
- **Evaluasi**: Metrik ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)

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

## 3. Alur Kerja Pipeline dari Awal Sampai Akhir

### 3.1 Diagram Alur NLP Pipeline

```
TEKS INPUT (Dokumen Bahasa Indonesia)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  TAHAP 1: PREPROCESSING                                      │
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
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│  TAHAP 2: SUMMARIZATION                                      │
│                                                              │
│  ┌─── EXTRACTIVE ─────────────────────────────────────────┐  │
│  │ 1. preprocess_for_extractive()  → list kalimat bersih   │  │
│  │ 2. _build_tfidf_matrix()       → matriks TF-IDF        │  │
│  │ 3. cosine_similarity()         → matriks kemiripan      │  │
│  │ 4. nx.from_numpy_array()       → buat graph             │  │
│  │ 5. nx.pagerank()               → skor setiap kalimat    │  │
│  │ 6. sorted() top-N              → pilih 5 kalimat terbaik│  │
│  │ 7. " ".join()                  → gabung jadi ringkasan  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─── ABSTRACTIVE ────────────────────────────────────────┐  │
│  │ 1. tokenizer(text)             → ubah teks ke token ID  │  │
│  │ 2. model.generate()            → beam search decode     │  │
│  │ 3. tokenizer.decode()          → ubah token ID ke teks  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│  TAHAP 3: EVALUASI                                           │
│                                                              │
│  Untuk setiap dokumen:                                       │
│  1. scorer.score(referensi, prediksi_extractive)             │
│     → ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)     │
│  2. scorer.score(referensi, prediksi_abstractive)            │
│     → ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)     │
│                                                              │
│  3. Hitung rata-rata F1 extractive vs abstractive            │
│  4. Tentukan best_method (yang rata-rata F1 lebih tinggi)    │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│  TAHAP 4: PERBANDINGAN & KESIMPULAN                          │
│                                                              │
│  - Skor ROUGE extractive vs abstractive                      │
│  - Tabel detail precision/recall/F1                          │
│  - Metode Terbaik berdasarkan rata-rata F1                   │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Alur Data di Kode (Bagaimana File-File Terhubung)

```
Teks Input
  │
  ▼
src/preprocessor.py (TextPreprocessor)
  │  case_folding() → clean_text() → sentence_tokenize()
  │  → word_tokenize() → remove_stopwords() → stem_tokens()
  │
  ├──→ src/extractive_model.py (ExtractiveSummarizer)
  │       _get_sentences() → _build_tfidf_matrix()
  │       → _build_similarity_graph() → _rank_sentences()
  │       → summarize()
  │
  ├──→ src/abstractive_model.py (AbstractiveSummarizer)
  │       tokenizer() → model.generate() → tokenizer.decode()
  │
  └──→ src/evaluator.py (Evaluator)
          scorer.score() → compute_rouge()
          → generate_comparison_table()
```

---

## 4. Konfigurasi NLP (`config.py`)

Berikut parameter-parameter NLP yang mengontrol perilaku pipeline:

### 4.1 Preprocessing

```python
CUSTOM_STOPWORDS = []  # Tambahkan stopword kustom di sini jika perlu
                       # Contoh: ["tersebut", "merupakan", "sehingga"]
```

### 4.2 Extractive Summarization

```python
NUM_EXTRACTIVE_SENTENCES = 5    # Jumlah kalimat yang akan dipilih
                                # Semakin banyak → ringkasan semakin panjang
TFIDF_MAX_FEATURES = 5000       # Maks fitur TF-IDF (dimensi vektor)
                                # Semakin tinggi → lebih detail tapi lebih lambat
```

### 4.3 Abstractive Summarization

```python
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
```

**Penjelasan penting:**

- **`MAX_SOURCE_LENGTH = 256`**: Model abstractive hanya membaca **256 token pertama** dari teks. Jika teksnya sangat panjang, bagian akhir akan dipotong. Ini trade-off antara memori/kecepatan vs kelengkapan.
- **`NUM_BEAMS = 4`**: Beam search mengeksplorasi 4 kandidat ringkasan secara paralel, mengurangi risiko output yang jelek. Tapi 4 beam membutuhkan 4× komputasi.
- **`RANDOM_SEED = 42`**: Angka 42 tidak spesial — hanya konvensi. Yang penting semua operasi random menggunakan seed yang sama agar hasilnya reproducible.

### 4.4 Evaluasi

```python
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
RANDOM_SEED = 42          # Seed agar hasil bisa direproduksi (sama setiap kali)
```

---

## 5. Tahap 1 — Preprocessing (`src/preprocessor.py`)

### 5.0 Inisialisasi Class

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

### 5.1 Case Folding — `case_folding(text)`

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

### 5.2 Cleaning — `clean_text(text)`

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

### 5.3 Tokenisasi Kalimat — `sentence_tokenize(text)`

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

### 5.4 Tokenisasi Kata — `word_tokenize(text)`

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

### 5.5 Stopword Removal — `remove_stopwords(tokens)`

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

### 5.6 Stemming — `stem_tokens(tokens)`

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

### 5.7 Method Tambahan: `preprocess_for_extractive(text)`

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

**Kenapa berbeda?** Extractive summarization memilih **kalimat utuh** sebagai ringkasan. Jika kita lakukan stopword removal dan stemming, kalimatnya rusak dan tidak bisa dibaca manusia:
```
Kalimat asli:    "Penelitian ini bertujuan untuk menganalisis pengaruh media sosial"
Setelah stem:    "teliti tuju analisis pengaruh media sosial"  ← RUSAK, tidak bisa dibaca
```
Maka untuk extractive, cukup **case folding + cleaning + tokenisasi kalimat** saja.

---

## 6. Tahap 2 — Extractive Summarization (`src/extractive_model.py`)

### 6.0 Inisialisasi Class

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

### 6.1 Step 1: Ambil Kalimat — `_get_sentences(text)`

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

### 6.2 Step 2: TF-IDF Matrix — `_build_tfidf_matrix(sentences)`

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

### 6.3 Step 3: Cosine Similarity — `_build_similarity_graph(tfidf_matrix)`

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

### 6.4 Step 4: PageRank — `_rank_sentences(graph)`

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

### 6.5 Step 5: Seleksi Kalimat — `summarize(text)`

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

### 6.6 Contoh End-to-End Extractive

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

## 7. Tahap 3 — Abstractive Summarization (`src/abstractive_model.py`)

### 7.0 Class `SummarizationDataset` (untuk Training)

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

### 7.1 Inisialisasi Class `AbstractiveSummarizer`

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

### 7.2 Load Model — `_load_model()` dan `_load_checkpoint()`

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

### 7.3 Fine-Tuning — `fine_tune()`

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

### 7.4 Inference (Generate Ringkasan) — `summarize(text)`

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

---

## 8. Tahap 4 — Evaluasi ROUGE (`src/evaluator.py`)

### 8.1 Inisialisasi Evaluator

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

### 8.2 Cara Kerja ROUGE — Dijelaskan dengan Contoh Nyata

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

### 8.3 Compute ROUGE — Source Code Lengkap

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

### 8.4 Cara Membaca Hasil ROUGE

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

## 9. Dataset yang Digunakan

### 9.1 Sumber Dataset

**XL-Sum (Cross-Lingual Summary)** dari `csebuetnlp/xlsum` di HuggingFace.

- **Bahasa:** Indonesia (dari koleksi 44 bahasa)
- **Asal data:** Artikel berita BBC Indonesian
- **Total corpus:** 38.242 artikel Indonesia
- **Yang kita gunakan:** 20 artikel (difilter)

### 9.2 Kriteria Seleksi

Dari 38.242 artikel, dipilih 20 dengan kriteria:
- Panjang teks > 800 karakter (artikel cukup substansial)
- Panjang ringkasan 100-800 karakter (ringkasan referensi cukup lengkap)

### 9.3 Format File

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

---

## 10. Library NLP yang Digunakan

| Library | Kegunaan dalam Project |
|---|---|
| **nltk** | Tokenisasi kalimat (`sent_tokenize`) dan kata (`word_tokenize`) dengan model bahasa Indonesia. Menyediakan daftar 759 stopword Indonesia. |
| **PySastrawi** | Stemmer Bahasa Indonesia berdasarkan algoritma **Nazief & Adriani**. Punya kamus 28.000+ kata dasar. Mengubah "pembelajaran" → "ajar", "berlari" → "lari". |
| **scikit-learn** | `TfidfVectorizer` untuk mengubah kalimat jadi vektor TF-IDF, dan `cosine_similarity` untuk menghitung kemiripan antar kalimat. |
| **networkx** | Membangun graph kemiripan kalimat (`from_numpy_array`) dan menjalankan algoritma **PageRank** (`nx.pagerank`). |
| **transformers** | Library dari HuggingFace untuk memuat dan menjalankan model Transformer. Kita pakai `AutoModelForSeq2SeqLM` (model mT5), `AutoTokenizer`, dan `Seq2SeqTrainer` untuk fine-tuning. |
| **torch (PyTorch)** | Framework deep learning dari Meta. Digunakan oleh Transformers untuk operasi tensor, forward/backward pass, dan gradient descent saat training. |
| **sentencepiece** | Tokenizer subword dari Google. Model mT5 menggunakan SentencePiece untuk memecah teks menjadi subword (potongan kata). Vocabulary: ~250.000 subword untuk 101 bahasa. |
| **rouge-score** | Menghitung metrik ROUGE-1, ROUGE-2, dan ROUGE-L antara ringkasan prediksi dan referensi. Library dari Google Research. |

---

## 11. FAQ / Pertanyaan Umum

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

### "Kenapa `preprocess_for_extractive` berbeda dari `preprocess` biasa?"

Extractive summarization **memilih kalimat asli** utuh — jadi kalimatnya HARUS tetap bisa dibaca manusia. Jika kita lakukan stopword removal dan stemming, kalimatnya rusak:
```
Kalimat asli:    "Penelitian ini bertujuan untuk menganalisis pengaruh media sosial"
Setelah stem:    "teliti tuju analisis pengaruh media sosial"  ← RUSAK, tidak bisa dibaca
```
Maka untuk extractive, cukup **case folding + cleaning + tokenisasi kalimat** saja.

### "Mau ganti model abstractive?"

Edit `config.py`, baris `ABSTRACTIVE_MODEL_NAME`:
```python
# Pilihan:
ABSTRACTIVE_MODEL_NAME = "google/mt5-small"                       # Pre-trained (perlu fine-tuning)
ABSTRACTIVE_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum-small" # Sudah fine-tuned untuk summarization
ABSTRACTIVE_MODEL_NAME = "LazarusNLP/IndoNanoT5-base"              # Khusus Indonesia
```
Setelah ganti, hapus folder `output/checkpoints/best_model/` dan jalankan training ulang.

### "File apa saja yang perlu saya baca untuk paham kode NLP-nya?"

Baca **dalam urutan ini** (dari yang paling mudah):

1. **`config.py`** — Pahami semua pengaturan NLP
2. **`src/preprocessor.py`** — Pahami 6 tahap preprocessing (paling mudah)
3. **`src/extractive_model.py`** — Pahami TF-IDF + PageRank
4. **`src/evaluator.py`** — Pahami ROUGE
5. **`src/abstractive_model.py`** — Pahami transformer/mT5 (paling kompleks)
