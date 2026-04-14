"""
Translate English rows in dataset.csv to Indonesian using Google Translate.
Saves after each row so progress is not lost on interruption.
"""
import pandas as pd
import re
import time
import os
from deep_translator import GoogleTranslator

CSV_PATH = 'data/raw/dataset.csv'
BACKUP_PATH = 'data/raw/dataset_backup.csv'

# Backup only if not already backed up
if not os.path.exists(BACKUP_PATH):
    df_orig = pd.read_csv(CSV_PATH)
    df_orig.to_csv(BACKUP_PATH, index=False)
    print("Backup saved to", BACKUP_PATH)

df = pd.read_csv(CSV_PATH)
print(f"Total rows: {len(df)}")

indo_words = {'yang','dan','di','ini','itu','dengan','untuk','pada','dari','adalah',
              'dalam','tidak','akan','telah','sudah','juga','atau','ke','oleh','bahwa',
              'karena','serta','dapat','harus','mereka','kami','kita','saya','sangat',
              'lebih','antara','tersebut','merupakan','menjadi','memiliki','seperti',
              'maka','namun','sedangkan','melalui','terhadap','tentang','kepada','hingga',
              'berdasarkan','penelitian','hasil','metode','data','analisis','bahasa',
              'indonesia','siswa','pembelajaran'}

def is_indonesian(text):
    words = re.findall(r'[a-z]+', str(text).lower())
    if not words:
        return True
    indo = sum(1 for w in words if w in indo_words)
    return (indo / len(words)) > 0.08

def translate_text(text, src='en', dest='id'):
    """Translate text, splitting into chunks if too long (Google limit ~5000 chars)."""
    text = str(text).strip()
    if not text:
        return text
    
    max_chunk = 4500
    if len(text) <= max_chunk:
        try:
            result = GoogleTranslator(source=src, target=dest).translate(text)
            return result if result else text
        except Exception as e:
            print(f"  Error: {e}")
            return text
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chunk:
            if current:
                chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current)
    
    translated_parts = []
    for chunk in chunks:
        try:
            result = GoogleTranslator(source=src, target=dest).translate(chunk)
            translated_parts.append(result if result else chunk)
        except Exception as e:
            print(f"  Error chunk: {e}")
            translated_parts.append(chunk)
        time.sleep(0.3)
    
    return " ".join(translated_parts)

# Find English rows that still need translation
en_rows = [i for i, row in df.iterrows() if not is_indonesian(row['full_text'])]
print(f"English rows remaining: {len(en_rows)}")

if not en_rows:
    print("All rows are already Indonesian!")
else:
    for count, idx in enumerate(en_rows):
        print(f"[{count+1}/{len(en_rows)}] Row {idx}...", end=" ", flush=True)
        
        df.at[idx, 'full_text'] = translate_text(df.at[idx, 'full_text'])
        time.sleep(0.2)
        df.at[idx, 'summary'] = translate_text(df.at[idx, 'summary'])
        
        # Save after each row
        df.to_csv(CSV_PATH, index=False)
        print("OK")
        time.sleep(0.3)
    
    print(f"\nDone!")

# Verify
id_count = sum(1 for _, row in df.iterrows() if is_indonesian(row['full_text']))
print(f"Indonesian: {id_count}/{len(df)}")
