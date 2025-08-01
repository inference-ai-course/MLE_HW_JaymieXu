import json
import glob
import re

from langdetect import detect
from bs4 import BeautifulSoup
from datasketch import MinHash
import hashlib

# load sample text start
sample_file = "sample_text_start.txt"

with open(sample_file, "r", encoding="utf-8") as f:
    sample_text = f.read()

# load abstracts
with open('../bonus_hw_1/arxiv_clean.json', 'r', encoding='utf-8') as f:
    arxiv_data = json.load(f)  # List of dicts

arxiv_texts = [item['abstract'] for item in arxiv_data]

# load OCR pdf
txt_files = glob.glob('../bonus_hw_2/pdf_ocr/*.txt')  # Adjust folder as needed
ocr_texts = []
if txt_files:
    with open(txt_files[0], 'r', encoding='utf-8') as f:
        ocr_texts.append(f.read())

# transcripts
transcript_texts = []
with open('../bonus_hw_3/talks_transcripts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        # Each obj['segments'] is a list of dicts with 'text'
        for seg in obj['segments']:
            transcript_texts.append(seg['text'])

all_texts_list = [sample_text] + arxiv_texts + ocr_texts + transcript_texts

chunks = []
for text in all_texts_list:
    for line in text.splitlines():
        if line.strip():
            chunks.append(line.strip())

def count_tokens(text):
    # Basic word count using regex
    return len(re.findall(r'\w+', text))

def strip_html_and_pii(text):
    # Strip HTML using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ", strip=True)

    # Remove simple PII: emails, phones, credit cards (add more patterns if needed)
    stripped_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', stripped_text)  # Email
    stripped_text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', stripped_text)  # Phone
    stripped_text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD]', stripped_text)  # Credit card
    return stripped_text

def remove_repetitive_ngrams(text, min_n=3, max_n=5):
    tokens = text.split()
    for n in range(max_n, min_n-1, -1):
        i = 0
        while i <= len(tokens) - 2*n:
            ngram = tokens[i:i+n]
            next_ngram = tokens[i+n:i+2*n]
            if ngram == next_ngram:
                del tokens[i+n:i+2*n]  # Remove the repeat
            else:
                i += 1
    return ' '.join(tokens)

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for token in set(text.split()):
        m.update(token.encode('utf8'))
    return m

total_tokens_before = sum(count_tokens(chunk) for chunk in chunks)
total_chunks_before = len(chunks)

final_chunks = []
removed_count = 0
removed_tokens = 0

accepted_minhashes = []

for chunk in chunks:
    try:
        lang = detect(chunk)
        if lang == 'en':
            # html and pii
            tokens_before_html = count_tokens(chunk)
            stripped_chunk = strip_html_and_pii(chunk)
            tokens_after_html = count_tokens(stripped_chunk)
            tokens_removed = tokens_before_html - tokens_after_html
            removed_tokens += tokens_removed

            # Remove repetitive n-grams
            ngram_chunk = remove_repetitive_ngrams(stripped_chunk)
            tokens_after_ngram = count_tokens(ngram_chunk)
            removed_tokens += (tokens_after_html - tokens_after_ngram)

            minhash = get_minhash(ngram_chunk)
            is_duplicate = False
            for prev_mh in accepted_minhashes:
                if minhash.jaccard(prev_mh) >= 0.7:
                    is_duplicate = True
                    break
            if is_duplicate:
                removed_count += 1
                removed_tokens += count_tokens(ngram_chunk)
                continue

            accepted_minhashes.append(minhash)

            final_chunks.append(stripped_chunk)
        else:
            removed_count += 1
            removed_tokens += count_tokens(chunk)
    except Exception:
        # If langdetect fails, treat as non-English and remove
        removed_count += 1
        removed_tokens += count_tokens(chunk)

total_tokens_after = sum(count_tokens(chunk) for chunk in final_chunks)
total_chunks_after = len(final_chunks)
removal_percentage = (removed_count / total_chunks_before) * 100 if total_chunks_before > 0 else 0
token_removal_percentage = (removed_tokens / total_tokens_before) * 100 if total_tokens_before > 0 else 0

print(f"Chunks before: {total_chunks_before}, Tokens before: {total_tokens_before}")
print(f"Chunks after: {total_chunks_after}, Tokens after: {total_tokens_after}")
print(f"Removal percentage (chunks): {removal_percentage:.2f}%")
print(f"Removal percentage (tokens): {token_removal_percentage:.2f}%")

# Save clean_corpus.txt
with open('clean_corpus.txt', 'w', encoding='utf-8') as f:
    for chunk in final_chunks:
        f.write(chunk + '\n')

# Save stats.md
with open('stats.md', 'w', encoding='utf-8') as f:
    f.write(f"Chunks before: {total_chunks_before}, Tokens before: {total_tokens_before}\n")
    f.write(f"Chunks after: {total_chunks_after}, Tokens after: {total_tokens_after}\n")
    f.write(f"Removal percentage (chunks): {removal_percentage:.2f}%\n")
    f.write(f"Removal percentage (tokens): {token_removal_percentage:.2f}%\n")