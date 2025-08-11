import datetime
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from urllib.parse import urlencode

import faiss
import fitz  # PyMuPDF
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import settings as cfg


def load_tokenizer() -> Any:
    # use_fast=True gives us a performant Rust tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.EMBED_MODEL, use_fast=True)

    max_len = getattr(tok, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < cfg.CHUNK_SIZE:
        print(f"[warn] CHUNK_SIZE={cfg.CHUNK_SIZE} > tokenizer max={max_len}; "
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
               page_no: int,
               chunk_size=cfg.CHUNK_SIZE,
               overlap_frac=cfg.OVERLAP_FRAC) -> List[Dict]:
    ids = tokenize(tok, page_text)
    out = []
    cid = 0

    for start, end in make_windows(ids, chunk_size, overlap_frac):
        piece_ids = ids[start:end]
        text = detokenize(tok, piece_ids).strip()

        if len(text) < cfg.MIN_CHARS:
            continue

        out.append({
            "chunk_id": f"{doc_id}::p{page_no}::c{cid}",
            "doc_id":   doc_id,
            "page":     page_no,
            "text":     text,
        })

        cid += 1

    return out


def chunk_document(tok, doc: Dict) -> List[Dict]:
    pages  = doc["text"].split(cfg.PAGE_BREAK)
    doc_id = doc["doc_id"]

    all_chunks = []
    for idx, page_text in enumerate(pages, start=1):
        page_text = (page_text or "").strip()

        if not page_text:
            continue

        page_chunks = chunk_page(tok, page_text, doc_id=doc_id, page_no=idx)
        all_chunks.extend(page_chunks)

    return all_chunks


def load_chunked_ids() -> set[str]:
    done = set()

    if cfg.CHUNKED_IDS.exists():
        for line in cfg.CHUNKED_IDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(line.strip())

    return done


def append_chunked_id(doc_id: str):
    with cfg.CHUNKED_IDS.open("a", encoding="utf-8") as f:
        f.write(doc_id + "\n")


def extract_arxiv_id(id_field: str) -> str:
    # arXiv gives IDs like "http://arxiv.org/abs/xxxx.19478v1"
    return id_field.strip().split("/")[-1]


_whitespace_re = re.compile(r"[ \t\f\v]+")


def normalize_text(s: str) -> str:
    # collapse excessive spaces, but keep newlines (paragraphs)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)

    # collapse super-long runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def normalize_title(s: str | None) -> str:
    if not s:
        return ""

    # Collapse all whitespace (including \r, \n, tabs) to single spaces.
    return " ".join(s.replace("\r", "\n").split())


def arxiv_search(query: str, total: int = 50, per_request: int = 50) -> List[Dict]:
    import feedparser
    out: List[Dict] = []
    start           = 0

    while len(out) < total:
        n = min(per_request, total - len(out))
        params = {
            "search_query": query,
            "start":        start,
            "max_results":  n,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        qs = urlencode(params)
        feed = feedparser.parse(f"{cfg.ARXIV_API}?{qs}")

        entries = getattr(feed, "entries", []) or []
        got = len(entries)
        if got == 0:
            # nothing more to fetch
            break

        for entry in entries:
            # find a pdf link (prefer rel=related when type=application/pdf)
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
                "title":            normalize_title(entry.title),
                "authors":          [a.name for a in getattr(entry, "authors", [])],
                "published":        entry.published,
                "updated":          getattr(entry, "updated", entry.published),
                "summary":          entry.summary.strip(),
                "pdf_url":          pdf_url,
                "primary_category": entry.tags[0]["term"] if entry.tags else None,
            })

        # advance by what we actually got
        start += got

        # be polite to arXiv
        time.sleep(3)

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
                    for chunk in r.iter_content(chunk_size=1 << 15):
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

    with fitz.open(pdf_path, filetype="pdf") as doc:
        if doc.is_encrypted:
            return pages

        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = page.get_text("text")
            pages.append(normalize_text(txt))

    return pages


def load_seen_doc_ids() -> set[str]:
    seen = set()
    if cfg.DOCS_OUT.exists():
        for line in cfg.DOCS_OUT.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["doc_id"])
            except Exception:
                pass

    return seen


def _load_doc_titles() -> dict[str, str]:
    m = {}

    if cfg.DOCS_OUT.exists():
        with cfg.DOCS_OUT.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                m[d["doc_id"]] = d.get("title") or ""
    return m


def _iter_chunks() -> Iterator[tuple[str, dict[str, Any]]]:
    with cfg.CHUNKS_OUT.open(encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            txt = (ch.get("text") or "").strip()

            if not txt:
                continue

            yield txt, {
                "chunk_id": ch["chunk_id"],
                "doc_id":   ch.get("doc_id"),
                "page":     ch.get("page"),
            }


def _load_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg.EMBED_MODEL, device=device)

    return model


def run_faiss() -> None:
    if not cfg.CHUNKS_OUT.exists():
        print(f"[warn] {cfg.CHUNKS_OUT} not found. Run chunking first.")
        return

    # clean rebuild
    for p in (cfg.FAISS_INDEX_PATH, cfg.SIDE_CAR_PATH, cfg.MANIFEST_PATH):
        if p.exists():
            p.unlink()

    model = _load_embedder()  # uses CUDA if available

    index: faiss.Index | None = None
    total = 0
    dim   = None

    titles = _load_doc_titles()

    def flush(flush_buf_txt: list[str], flush_buf_meta: list[dict]):
        nonlocal index, dim, total

        if not flush_buf_txt:
            return

        vecs = model.encode(
            flush_buf_txt,
            batch_size=cfg.BATCH,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine: L2-normalize in model
        ).astype("float32")

        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)  # cosine via IP on normalized vectors

        index.add(vecs)

        for m, t in zip(flush_buf_meta, flush_buf_txt):
            # attach title
            m["title"]   = titles.get(m["doc_id"], "")
            sidecar.write(json.dumps(m, ensure_ascii=False) + "\n")

        total += len(flush_buf_txt)
        flush_buf_txt.clear()
        flush_buf_meta.clear()

    with cfg.SIDE_CAR_PATH.open("w", encoding="utf-8") as sidecar:
        buf_txt, buf_meta = [], []

        for txt, meta in _iter_chunks():
            buf_txt.append(txt)
            buf_meta.append(meta)

            if len(buf_txt) >= cfg.BATCH:
                flush(buf_txt, buf_meta)
        # flush tail
        flush(buf_txt, buf_meta)

    if index is None:
        print("[warn] No chunks to index.")
        return

    faiss.write_index(index, str(cfg.FAISS_INDEX_PATH))
    cfg.MANIFEST_PATH.write_text(json.dumps({
        "created_at":    datetime.datetime.now().isoformat(),
        "embed_model":   cfg.EMBED_MODEL,
        "metric":        "cosine",
        "count":         total,
        "dim":           dim,
        "chunks_source": str(cfg.CHUNKS_OUT),
        "sidecar":       str(cfg.SIDE_CAR_PATH),
    }, indent=2), encoding="utf-8")

    print(f"[OK] FAISS built: {total} vecs, dim={dim}")
    print(f"      index  → {cfg.FAISS_INDEX_PATH}")
    print(f"      sidecar→ {cfg.SIDE_CAR_PATH}")


def run_extract(limit: int | None = None):
    lines = cfg.META_FILE.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]

    seen_doc_ids = load_seen_doc_ids()

    with cfg.DOCS_OUT.open("a", encoding="utf-8") as doc_f:
        page_f = cfg.PAGES_OUT.open("a", encoding="utf-8") if cfg.PRODUCE_DEBUG_PAGES else None
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

                full_text = cfg.PAGE_BREAK.join(pages)
                out_doc = {
                    "doc_id":           arxiv_id,
                    "title":            rec.get("title"),
                    "authors":          rec.get("authors"),
                    "published":        rec.get("published"),
                    "primary_category": rec.get("primary_category"),
                    "n_pages":          len(pages),
                    "source_pdf":       str(pdf_path),
                    "text":             full_text,
                    "created_at":       datetime.datetime.now().isoformat(),
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

            print(f"Done extract. new_docs={n_new} → {cfg.DOCS_OUT}")

        finally:
            if page_f:
                page_f.close()


def run_chunk(limit_docs: int | None = None):
    tok      = load_tokenizer()
    already  = load_chunked_ids()
    n_docs   = 0
    n_chunks = 0

    cfg.CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

    with cfg.CHUNKS_OUT.open("a", encoding="utf-8") as out_f:
        with cfg.DOCS_OUT.open(encoding="utf-8") as fin:
            for line in fin:
                if limit_docs is not None and n_docs >= limit_docs:
                    break  # stop after we have chunked `limit_docs` number of docs

                doc = json.loads(line)
                doc_id = doc.get("doc_id")

                if not doc_id or doc_id in already:
                    continue
                if not doc.get("text"):
                    continue

                chunks = chunk_document(tok, doc)
                for ch in chunks:
                    out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

                append_chunked_id(doc_id)

                n_docs += 1
                n_chunks += len(chunks)

                print(f"[OK] {doc.get('title','(untitled)')[:60]}…  chunks={len(chunks)}")

    print(f"Done chunk. new_docs={n_docs}  new_chunks={n_chunks} → {cfg.CHUNKS_OUT}")


def run_fetch_arxiv(total: int, per_request: int = 50, query: str = "cat:cs.CL"):
    results = arxiv_search(query, total=total, per_request=per_request)

    seen_ids = set()

    # Add existing pdfs if this is not first run
    if cfg.META_FILE.exists():
        for line in cfg.META_FILE.read_text(encoding="utf-8").splitlines():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    with cfg.META_FILE.open("a", encoding="utf-8") as mf:
        for rec in results:
            if rec["id"] in seen_ids:
                continue

            arxiv_id = extract_arxiv_id(rec["id"])
            stem     = arxiv_id.replace("/", "_")
            fn       = f"{stem}.pdf"
            pdf_path = cfg.RAW / fn

            ok = False
            if rec["pdf_url"]:
                ok = download_pdf(rec["pdf_url"], pdf_path)

            rec_out = dict(rec)
            rec_out["local_pdf"] = str(pdf_path) if ok else None

            mf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(f"[{'OK' if ok else '--'}] {rec_out['title']}")


def main() -> None:
    run_fetch_arxiv(50)
    run_extract(limit=None)
    run_chunk(limit_docs=None)
    run_faiss()


if __name__ == "__main__":
    main()
    