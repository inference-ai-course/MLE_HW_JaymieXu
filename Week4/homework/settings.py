from pathlib import Path

# ----- base folders -----
BASE    = Path(__file__).resolve().parent
DATA    = BASE / "data"
RAW     = DATA / "raw_pdfs"
META    = DATA / "metadata"
PROC    = DATA / "processed"
INDEXED = BASE / "data" / "index"

# ----- files -----
META_FILE        = META / "arxiv_metadata.jsonl"
DOCS_OUT         = PROC / "documents.jsonl"
PAGES_OUT        = PROC / "pages.jsonl"
CHUNKS_OUT       = PROC / "chunks.jsonl"
CHUNKED_IDS      = PROC / "chunks.done"

FAISS_INDEX_PATH = INDEXED / "faiss.index"
SIDE_CAR_PATH    = INDEXED / "chunk_meta.jsonl"
MANIFEST_PATH    = INDEXED / "manifest.json"

# ----- text & API -----
PAGE_BREAK = "\n\n<<<PAGE_BREAK>>>\n\n"
ARXIV_API  = "http://export.arxiv.org/api/query"
MIN_SCORE  = 0.25   # Bullsh*t cut off score

# Toggle: write pages.jsonl for debugging provenance
PRODUCE_DEBUG_PAGES = False

# ----- embedding / chunking knobs -----
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 256
OVERLAP_FRAC = 0.20  # 20%
MIN_CHARS    = 40

# ----- embedding batch / metric -----
BATCH       = 8                         # adjust to your VRAM/CPU
NORMALIZE   = True                      # cosine via IP when True

# ----- ensure directories exist on import -----
for p in (DATA, RAW, META, PROC, INDEXED):
    p.mkdir(parents=True, exist_ok=True)
