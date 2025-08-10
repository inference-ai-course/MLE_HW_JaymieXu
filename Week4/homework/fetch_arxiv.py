import time, json, math, hashlib, re, datetime
from pathlib import Path
from typing import List, Dict
import requests
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch, faiss
import numpy as np

BASE    = Path(__file__).resolve().parent
DATA    = BASE / "data"
RAW     = DATA / "raw_pdfs"
META    = DATA / "metadata"
PROC    = DATA / "processed"
INDEXED = BASE / "data" / "index"

META_FILE   = META / "arxiv_metadata.jsonl"
DOCS_OUT    = PROC / "documents.jsonl"
PAGES_OUT   = PROC / "pages.jsonl"
CHUNKS_OUT  = PROC / "chunks.jsonl"
CHUNKED_IDS = PROC / "chunks.done"

FAISS_INDEX_PATH  = INDEXED / "faiss.index"
SIDE_CAR_PATH     = INDEXED / "chunk_meta.jsonl"
MANIFEST_PATH     = INDEXED / "manifest.json"

PAGE_BREAK = "\n\n<<<PAGE_BREAK>>>\n\n"

RAW.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
INDEXED.mkdir(parents=True, exist_ok=True)

PRODUCE_PAGES = True

ARXIV_API = "http://export.arxiv.org/api/query"

EMBEDDING_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 256
OVERLAP_FRAC = 0.20 #20%
MIN_CHARS    = 40

BATCH       = 8                         # adjust to your VRAM/CPU
NORMALIZE   = True                      # cosine via IP when True

def load_tokenizer():
    # use_fast=True gives us a performant Rust tokenizer
    tok = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER, use_fast=True)

    max_len = getattr(tok, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < CHUNK_SIZE:
        print(f"[warn] CHUNK_SIZE={CHUNK_SIZE} > tokenizer max={max_len}; "
              f"the embedder may truncate later.")

    return tok

def tokenize(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)

def detokenize(tok, ids: List[int]) -> str:
    # reconstruct readable text for retrieval UI/debugging
    return tok.decode(ids, skip_special_tokens=True)

def make_windows(token_ids: List[int], chunk_size: int, overlap_frac: float):
    overlap = int(round(chunk_size * overlap_frac))
    step    = max(1, chunk_size - overlap)

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

def chunk_page(tok,
               page_text: str,
               *,
               doc_id: str,
               title: str,
               year: str,
               page_no: int,
               chunk_size=CHUNK_SIZE,
               overlap_frac=OVERLAP_FRAC) -> List[Dict]:
    ids = tokenize(tok, page_text)
    out = []
    cid = 0

    for start, end in make_windows(ids, chunk_size, overlap_frac):
        piece_ids = ids[start:end]
        text = detokenize(tok, piece_ids).strip()

        if len(text) < MIN_CHARS:
            continue

        out.append({
            "chunk_id":    f"{doc_id}::p{page_no}::c{cid}",
            "doc_id":      doc_id,
            "title":       title,
            "year":        year,
            "page_start":  page_no,
            "page_end":    page_no,
            "token_start": start,
            "token_end":   end,
            "token_count": end - start,
            "char_count":  len(text),
            "text":        text,
        })

        cid += 1

    return out

def chunk_document(tok, doc: Dict) -> List[Dict]:
    pages  = doc["text"].split(PAGE_BREAK)
    title  = doc.get("title") or ""
    year   = (doc.get("published") or "")[:4]
    doc_id = doc["doc_id"]

    all_chunks = []
    for idx, page_text in enumerate(pages, start=1):
        page_text = (page_text or "").strip()

        if not page_text:
            continue

        page_chunks = chunk_page(tok, page_text, doc_id=doc_id, title=title, year=year, page_no=idx)
        all_chunks.extend(page_chunks)

    return all_chunks

def load_chunked_ids() -> set[str]:
    done = set()

    if CHUNKED_IDS.exists():
        for line in CHUNKED_IDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(line.strip())

    return done

def append_chunked_id(doc_id: str):
    with CHUNKED_IDS.open("a", encoding="utf-8") as f:
        f.write(doc_id + "\n")

def extract_arxiv_id(id_field: str) -> str:
    # arXiv gives IDs like "http://arxiv.org/abs/xxxx.19478v1"
    return id_field.strip().split("/")[-1]

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)

    return h.hexdigest()

_whitespace_re = re.compile(r"[ \t\f\v]+")
def normalize_text(s: str) -> str:
    # collapse excessive spaces, but keep newlines (paragraphs)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)

    # collapse super-long runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:128]

def arxiv_search(query: str, total: int = 50, per_request: int = 50) -> List[Dict]:
    import feedparser

    out = []
    pages = math.ceil(total / per_request) # Ceil so we do not end up requesting fraction number of PDFs

    # Request chunks
    for i in range(pages):
        start = i * per_request
        n = min(per_request, total - start) # Min should be (total - start) on the last run if total and per_request do not nicely add up

        params = {
            "search_query": query,
            "start":        start,
            "max_results":  n,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        feed = feedparser.parse(
            ARXIV_API + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        )

        for entry in feed.entries:
            pdf_url = None

            # PDF find
            for link in getattr(entry, "links", []):
                if getattr(link, "type", "") != "application/pdf":
                    continue

                # prefer the canonical one
                if getattr(link, "rel", "") == "related":
                    pdf_url = link.href
                    break

                 # otherwise remember the FIRST pdf we saw (don’t overwrite later)
                if pdf_url is None:
                    pdf_url = link.href

            out.append({
                "id":               entry.id,
                "title":            entry.title.strip(),
                "authors":          [a.name for a in getattr(entry, "authors", [])],
                "published":        entry.published,
                "updated":          getattr(entry, "updated", entry.published),
                "summary":          entry.summary.strip(),
                "pdf_url":          pdf_url,
                "primary_category": entry.tags[0]["term"] if entry.tags else None,
            })
        time.sleep(1)  # be polite

    return out[:total]

def download_pdf(url: str, dest: Path, retries: int = 3) -> bool:
    # Check if file already exists and not corrupted
    if dest.exists() and dest.stat().st_size > 0:
        return True

    for attempt in range(1, retries+1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1<<15):
                        if chunk:
                            f.write(chunk)

            # Check if file is not corrupted
            if dest.stat().st_size > 0:
                return True

        except Exception as e:
            if attempt == retries:
                print(f"[!] Failed {url}: {e}")
                return False

            time.sleep(2 * attempt)

    return False

def extract_pages(pdf_path: Path) -> list[str]:
    pages = []

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            return pages

        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = page.get_text("text")
            pages.append(normalize_text(txt))

    return pages

def load_seen_doc_ids() -> set[str]:
    seen = set()
    if DOCS_OUT.exists():
        for line in DOCS_OUT.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["doc_id"])
            except Exception:
                pass

    return seen

def _iter_chunks():
    """Yield (text, minimal_meta) in file order."""
    with CHUNKS_OUT.open(encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            txt = (ch.get("text") or "").strip()

            if not txt:
                continue

            yield txt, {
                "chunk_id": ch["chunk_id"],
                "doc_id":   ch.get("doc_id"),
                "title":    ch.get("title"),
                "page":     ch.get("page_start"),
            }

def _load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_TOKENIZER, device=device)

    return model

def _embed(model, texts):
    vecs = model.encode(texts,
                        batch_size=BATCH,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=False).astype("float32")
    if NORMALIZE:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
    return vecs

def run_faiss():
    if not CHUNKS_OUT.exists():
        print(f"[warn] {CHUNKS_OUT} not found. Run chunking first.")
        return

    # clean rebuild
    for p in (FAISS_INDEX_PATH, SIDE_CAR_PATH, MANIFEST_PATH):
        if p.exists():
            p.unlink()

    model = _load_embedder()  # uses CUDA if available

    index = None
    total = 0
    dim   = None

    def flush(buf_txt: list[str], buf_meta: list[dict]):
        nonlocal index, dim, total

        if not buf_txt:
            return

        vecs = model.encode(
            buf_txt,
            batch_size=BATCH,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine: L2-normalize in model
        ).astype("float32")

        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)  # cosine via IP on normalized vectors

        index.add(vecs)

        for m in buf_meta:
            sidecar.write(json.dumps(m, ensure_ascii=False) + "\n")

        total += len(buf_txt)
        buf_txt.clear(); buf_meta.clear()

    with SIDE_CAR_PATH.open("w", encoding="utf-8") as sidecar:
        buf_txt, buf_meta = [], []

        for txt, meta in _iter_chunks():
            buf_txt.append(txt)
            buf_meta.append(meta)

            if len(buf_txt) >= BATCH:
                flush(buf_txt, buf_meta)
        # flush tail
        flush(buf_txt, buf_meta)

    if index is None:
        print("[warn] No chunks to index.")
        return

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    MANIFEST_PATH.write_text(json.dumps({
        "created_at":    datetime.datetime.utcnow().isoformat() + "Z",
        "embed_model":   EMBEDDING_TOKENIZER,
        "metric":        "cosine",
        "count":         total,
        "dim":           dim,
        "chunks_source": str(CHUNKS_OUT),
        "sidecar":       str(SIDE_CAR_PATH),
    }, indent=2), encoding="utf-8")

    print(f"[OK] FAISS built: {total} vecs, dim={dim}")
    print(f"      index  → {FAISS_INDEX_PATH}")
    print(f"      sidecar→ {SIDE_CAR_PATH}")

def run_extract(limit: int | None = None):
    lines = META_FILE.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]

    seen_doc_ids = load_seen_doc_ids()

    with DOCS_OUT.open("a", encoding="utf-8") as doc_f:
        page_f = PAGES_OUT.open("a", encoding="utf-8") if PRODUCE_PAGES else None
        try:
            n_new = 0
            for line in lines:
                rec = json.loads(line)
                arxiv_id = extract_arxiv_id(rec["id"])

                if arxiv_id in seen_doc_ids:
                    continue

                pdf_local = rec.get("local_pdf")
                if not pdf_local or not Path(pdf_local).exists():
                    print(f"[--] Missing PDF → {rec.get('title')}")
                    continue

                pdf_path = Path(pdf_local)
                pages = extract_pages(pdf_path)

                if not pages:  # encrypted or empty
                    print(f"[SKIP] {pdf_path.name} — encrypted or no extractable text")
                    continue

                full_text = PAGE_BREAK.join(pages)
                out_doc = {
                    "doc_id":           arxiv_id,
                    "title":            rec.get("title"),
                    "authors":          rec.get("authors"),
                    "published":        rec.get("published"),
                    "updated":          rec.get("updated"),
                    "primary_category": rec.get("primary_category"),
                    "n_pages":          len(pages),
                    "pdf_sha1":         sha1_of_file(pdf_path),
                    "source_pdf":       str(pdf_path),
                    "text":             full_text,
                    "created_at":       datetime.datetime.utcnow().isoformat() + "Z",
                }
                doc_f.write(json.dumps(out_doc, ensure_ascii=False) + "\n")

                if page_f:  # optional page-level lines
                    title = out_doc.get("title") or ""
                    year  = (out_doc.get("published") or "")[:4]

                    for i, ptxt in enumerate(pages, start=1):
                        page_f.write(json.dumps({
                            "doc_id": arxiv_id,
                            "page":   i,
                            "text":   ptxt,
                            "title":  title,
                            "year":   year,
                        }, ensure_ascii=False) + "\n")

                n_new += 1
                print(f"[OK] {pdf_path.name}  pages={len(pages)}")

            print(f"Done extract. new_docs={n_new} → {DOCS_OUT}")

        finally:
            if page_f:
                page_f.close()

def run_chunk(limit_docs: int | None = None):
    tok = load_tokenizer()
    lines = DOCS_OUT.read_text(encoding="utf-8").splitlines()

    if limit_docs is not None:
        lines = lines[:limit_docs]

    already  = load_chunked_ids()
    n_docs   = 0
    n_chunks = 0

    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

    with CHUNKS_OUT.open("a", encoding="utf-8") as out_f:
        for line in lines:
            doc = json.loads(line)
            doc_id = doc.get("doc_id")

            if not doc_id or doc_id in already:
                continue
            if not doc.get("text"):
                continue

            chunks = chunk_document(tok, doc)
            for ch in chunks:
                ch["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

            append_chunked_id(doc_id)

            n_docs += 1
            n_chunks += len(chunks)

            print(f"[OK] {doc.get('title','(untitled)')[:60]}…  chunks={len(chunks)}")

    print(f"Done chunk. new_docs={n_docs}  new_chunks={n_chunks} → {CHUNKS_OUT}")

def run_fetch_arxiv(total: int, per_request: int = 50, query: str = "cat:cs.CL"):
    results = arxiv_search(query, total=total, per_request=per_request)

    seen_ids = set()

    # Add existing pdfs if this is not first run
    if META_FILE.exists():
        for line in META_FILE.read_text(encoding="utf-8").splitlines():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    with META_FILE.open("a", encoding="utf-8") as mf:
        for rec in results:
            if rec["id"] in seen_ids:
                continue

            h = hashlib.md5(rec["id"].encode("utf-8")).hexdigest()[:10]
            fn = safe_name(f"{rec['title']}_{h}.pdf")
            pdf_path = RAW / fn

            ok = False
            if rec["pdf_url"]:
                ok = download_pdf(rec["pdf_url"], pdf_path)

            rec_out = dict(rec)
            rec_out["local_pdf"] = str(pdf_path) if ok else None

            mf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(f"[{'OK' if ok else '--'}] {rec_out['title']}")

def _load_index_and_sidecar():
    idx     = faiss.read_index(str(FAISS_INDEX_PATH))
    sidecar = [json.loads(l) for l in SIDE_CAR_PATH.read_text(encoding="utf-8").splitlines()]

    # quick consistency check
    if idx.ntotal != len(sidecar):
        print(f"[warn] index has {idx.ntotal} vecs but sidecar has {len(sidecar)} lines.")
    return idx, sidecar

def _load_query_model():
    model_name = EMBEDDING_TOKENIZER

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        model_name = manifest.get("embed_model", model_name)
    except Exception:
        pass

    return SentenceTransformer(model_name)

def search_local(query: str, k: int = 5, with_text: bool = True):
    idx, sidecar = _load_index_and_sidecar()
    model        = _load_query_model()

    # embed query; normalize so IP == cosine (matches how we built the index)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = idx.search(q, k)
    ids, scores = ids[0].tolist(), scores[0].tolist()

    # map result ids -> sidecar meta
    metas = []
    for i, s in zip(ids, scores):
        if 0 <= i < len(sidecar):
            m = sidecar[i]
            metas.append({"rank": len(metas)+1, "score": float(s), **m})

    previews = {}
    if with_text and metas:
        wanted = {m["chunk_id"] for m in metas}
        with CHUNKS_OUT.open(encoding="utf-8") as f:
            for line in f:
                ch = json.loads(line)
                cid = ch.get("chunk_id")
                if cid in wanted:
                    previews[cid] = (ch.get("text") or "")[:300].replace("\n", " ")
                    if len(previews) == len(wanted):
                        break

    # print results
    print(f'\nQuery: "{query}"  (top {k})')
    for m in metas:
        prev = f' — "{previews.get(m["chunk_id"], "")}…"' if with_text else ""
        print(f"{m['rank']:>2}. score={m['score']:.3f}  p{m.get('page')}  {m.get('title')}{prev}")

    return metas

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=str, help="run a quick local search")
    ap.add_argument("-k", type=int, default=3)
    args, _ = ap.parse_known_args()

    if args.search:
        # TEST: python -X utf8 fetch_arxiv.py --search "transformer attention mechanism" -k 3
        search_local(args.search, k=args.k, with_text=True)
    else:
        run_fetch_arxiv(5)
        run_extract(limit=None)
        run_chunk(limit_docs=None)
        run_faiss()

if __name__ == "__main__":
    main()