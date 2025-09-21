import datetime
import json
import sqlite3
import re
from pathlib import Path
from typing import Any, Dict, List, Iterator

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from rag2_config import cfg, get_rag2_config_path
import rag2_sql_util as rag2_sql_util

# Global path variables
PARSED_FILE   = get_rag2_config_path("proc") / "parsed.json"
CHUNKS_FILE   = get_rag2_config_path("proc") / cfg.files.chunks_file
INDEX_DIR     = get_rag2_config_path("indexed")
FAISS_PATH    = INDEX_DIR / cfg.files.faiss_index
SIDECAR_PATH  = INDEX_DIR / cfg.files.side_car
MANIFEST_PATH = INDEX_DIR / cfg.files.manifest


def load_tokenizer() -> Any:
    """Load tokenizer with configuration validation"""
    tok = AutoTokenizer.from_pretrained(cfg.model.embed_model, use_fast=True)

    max_len = getattr(tok, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < cfg.model.chunk_size:
        print(f"[warn] CHUNK_SIZE={cfg.model.chunk_size} > tokenizer max={max_len}; "
              f"the embedder may truncate later.")

    return tok


def tokenize(tok, text: str) -> List[int]:
    """Convert text to token IDs"""
    return tok.encode(text, add_special_tokens=False)


def detokenize(tok, ids: List[int]) -> str:
    """Convert token IDs back to text"""
    return tok.decode(ids, skip_special_tokens=True)


def make_windows(token_ids: List[int], chunk_size: int, overlap_frac: float):
    """Generate sliding windows of tokens with overlap"""
    overlap = int(round(chunk_size * overlap_frac))
    step = max(1, chunk_size - overlap)

    n = len(token_ids)
    if n == 0:
        return

    start = 0
    while True:
        end = min(start + chunk_size, n)
        yield start, end

        if end >= n:
            break

        start = start + step


def normalize_text(s: str) -> str:
    """Basic text normalization - main cleanup now done in parser"""
    if not s or not s.strip():
        return ""

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def chunk_text_segment(tok, text: str, doc_id: str, segment_idx: int,
                      chunk_size: int = None, overlap_frac: float = None) -> List[Dict]:
    """Chunk a single text segment into overlapping pieces"""
    if chunk_size is None:
        chunk_size = cfg.model.chunk_size
    if overlap_frac is None:
        overlap_frac = cfg.model.overlap_frac

    # Normalize text first
    text = normalize_text(text)

    # Skip empty or very short segments
    if len(text) < cfg.model.min_char:
        return []

    ids = tokenize(tok, text)
    out = []
    cid = 0

    for start, end in make_windows(ids, chunk_size, overlap_frac):
        piece_ids = ids[start:end]
        chunk_text = detokenize(tok, piece_ids).strip()

        # More strict filtering for better quality chunks
        if (len(chunk_text) < cfg.model.min_char or
            len(chunk_text.split()) < 8 or  # At least 8 words
            chunk_text.count('.') == 0):    # Should have some sentence structure
            continue

        # Add BGE-specific optimization: prefix for better retrieval
        prefixed_text = f"passage: {chunk_text}"

        out.append({
            "chunk_id": f"{doc_id}::s{segment_idx}::c{cid}",
            "doc_id": doc_id,
            "page": segment_idx,  # Use segment index as "page"
            "text": chunk_text,      # Store original text
            "embed_text": prefixed_text,  # Text for embedding
        })

        cid += 1

    return out


def load_parsed_data() -> Dict[str, List[str]]:
    """Load the parsed.json data"""
    if not PARSED_FILE.exists():
        raise RuntimeError(f"Parsed data not found at {PARSED_FILE}")

    with open(PARSED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_embedder() -> SentenceTransformer:
    """Load the sentence transformer model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg.model.embed_model, device=device)
    return model


def _iter_chunks() -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate over chunks from database"""
    conn = sqlite3.connect(CHUNKS_FILE)
    cursor = conn.execute("SELECT text, chunk_id, doc_id, page FROM chunks WHERE text IS NOT NULL AND text != ''")

    for row in cursor.fetchall():
        txt = row[0].strip()
        if not txt:
            continue

        # Use BGE-specific prefix for embedding
        embed_text = f"passage: {txt}"

        yield embed_text, {
            "chunk_id": row[1],
            "doc_id": row[2],
            "page": row[3],
        }

    conn.close()


def run_chunk():
    """Process parsed data and create chunks"""
    print("[INFO] Starting chunking process...")

    # Load data and tokenizer
    parsed_data = load_parsed_data()
    tok = load_tokenizer()

    # Initialize chunks database
    rag2_sql_util.init_chunks_db()

    # Connect to chunks database
    chunks_conn = sqlite3.connect(CHUNKS_FILE)

    try:
        total_chunks = 0

        for doc_id, text_segments in parsed_data.items():
            print(f"[INFO] Processing document: {doc_id}")

            doc_chunks = 0
            for segment_idx, text_segment in enumerate(text_segments):
                if not text_segment or not text_segment.strip():
                    continue

                chunks = chunk_text_segment(tok, text_segment, doc_id, segment_idx)

                for ch in chunks:
                    # Insert into chunks table
                    chunks_conn.execute('''
                        INSERT INTO chunks (chunk_id, doc_id, page, text)
                        VALUES (?, ?, ?, ?)
                    ''', (ch["chunk_id"], ch["doc_id"], ch["page"], ch["text"]))

                    # Insert into FTS table
                    chunks_conn.execute('''
                        INSERT INTO chunks_fts (chunk_id, doc_id, text)
                        VALUES (?, ?, ?)
                    ''', (ch["chunk_id"], ch["doc_id"], ch["text"]))

                doc_chunks += len(chunks)

            total_chunks += doc_chunks
            print(f"[OK] {doc_id}: {doc_chunks} chunks")

        chunks_conn.commit()
        print(f"[INFO] Chunking completed. Total chunks: {total_chunks}")

    finally:
        chunks_conn.close()


def run_faiss():
    """Build FAISS index from chunks"""
    print("[INFO] Starting FAISS index building...")

    if not CHUNKS_FILE.exists():
        print(f"[ERROR] Chunks file not found: {CHUNKS_FILE}")
        return

    # Initialize paths
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Clean rebuild
    for p in (FAISS_PATH, SIDECAR_PATH, MANIFEST_PATH):
        if p.exists():
            p.unlink()

    # Load embedder
    model = _load_embedder()

    # Initialize sidecar database
    rag2_sql_util.init_chunk_meta_db()

    index: faiss.Index = None
    total = 0
    dim = None

    sidecar_conn = sqlite3.connect(SIDECAR_PATH)

    def flush(flush_buf_txt: list[str], flush_buf_meta: list[dict]):
        nonlocal index, dim, total

        if not flush_buf_txt:
            return

        # Generate embeddings with BGE optimizations
        vecs = model.encode(
            flush_buf_txt,
            batch_size=cfg.model.batch,
            convert_to_numpy=True,
            show_progress_bar=True,  # Show progress for long operations
            normalize_embeddings=cfg.model.normalize,
            convert_to_tensor=False,
            device=model.device,
        ).astype("float32")

        # Initialize index on first batch
        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)

        # Add vectors to index
        index.add(vecs)

        # Store metadata in sidecar
        for m in flush_buf_meta:
            sidecar_conn.execute('''
                INSERT INTO chunk_meta (chunk_id, doc_id, page, title)
                VALUES (?, ?, ?, ?)
            ''', (m["chunk_id"], m["doc_id"], m["page"], m.get("title", "")))

        total += len(flush_buf_txt)
        flush_buf_txt.clear()
        flush_buf_meta.clear()

    try:
        buf_txt, buf_meta = [], []

        for txt, meta in _iter_chunks():
            buf_txt.append(txt)
            buf_meta.append(meta)

            if len(buf_txt) >= cfg.model.batch:
                flush(buf_txt, buf_meta)

        # Flush remaining
        flush(buf_txt, buf_meta)

        sidecar_conn.commit()

    finally:
        sidecar_conn.close()

    if index is None:
        print("[WARN] No chunks to index.")
        return

    # Save index
    faiss.write_index(index, str(FAISS_PATH))

    # Save manifest
    manifest_data = {
        "created_at": datetime.datetime.now().isoformat(),
        "embed_model": cfg.model.embed_model,
        "metric": "cosine",
        "count": total,
        "dim": dim,
        "chunks_source": str(CHUNKS_FILE),
        "sidecar": str(SIDECAR_PATH),
    }

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[OK] FAISS built: {total} vectors, dim={dim}")
    print(f"      index   → {FAISS_PATH}")
    print(f"      sidecar → {SIDECAR_PATH}")


def main():
    """Main build process"""
    print("=== RAG2 Build Process ===")
    run_chunk()
    run_faiss()
    print("=== Build Complete ===")


if __name__ == "__main__":
    main()

