from __future__ import annotations
import json
from pathlib import Path
import re
from collections import Counter
from typing import List, Tuple

import fitz  # PyMuPDF

from rag2_config import cfg, get_rag2_config_path
from rag2_anonymizer import anonymize_text

# ---------- helpers: normalization / heuristics ----------

_whitespace_re = re.compile(r"[ \t\f\v]+")
_bullet_or_list = re.compile(r"^\s*(?:[-*•·]|(?:\d+|[A-Za-z])[\.\)])\s+")
_heading_like = re.compile(r"^[A-Z0-9][A-Z0-9 \-:]{4,}$")  # crude ALLCAPS-ish
_soft_hyphen = "\u00ad"

def contains_figure_ref(text: str) -> bool:
    """
    Return True if the text contains 'fig.', 'fig', or 'figure' or 'figure.'
    (case-insensitive, by lowering input first).
    """
    text = text.lower()
    return "fig." in text or "fig " in text or "figure" in text or "figure." in text

def normalize_text(s: str) -> str:
    # collapse excessive spaces, but keep newlines (paragraphs)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace(_soft_hyphen, "")  # remove soft hyphens if any
    s = _whitespace_re.sub(" ", s)

    # collapse super-long runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def clean_rag_text(s: str) -> str:
    """Enhanced text cleaning for better RAG performance"""
    if not s or not s.strip():
        return ""

    # More aggressive text cleaning for better RAG
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace but preserve paragraph structure
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n[ \t]*\n", "\n\n", s)  # Clean up paragraph breaks
    s = re.sub(r"\n{3,}", "\n\n", s)      # Max 2 consecutive newlines

    # Remove common academic paper artifacts
    s = re.sub(r"arXiv:\d+\.\d+v?\d*", "", s)  # Remove arXiv IDs
    s = re.sub(r"\[[\d\s,\-]+\]", "", s)       # Remove citation numbers like [1, 2-5]

    # Remove repeated punctuation patterns
    s = re.sub(r";(\s*;)+", ";", s)  # Multiple semicolons: ; ; ; ; -> ;
    s = re.sub(r",(\s*,)+", ",", s)  # Multiple commas: , , , , -> ,
    s = re.sub(r"\.(\s*\.)+", ".", s)  # Multiple periods: . . . . -> .
    s = re.sub(r"-(\s*-)+", "-", s)  # Multiple dashes: - - - - -> -

    return s.strip()


def is_text_suitable_for_rag(text: str, min_chars: int = 40, min_words: int = 8) -> bool:
    """Check if text meets quality criteria for RAG indexing"""
    if not text or len(text) < min_chars:
        return False

    # Check word count
    if len(text.split()) < min_words:
        return False

    # Should have some sentence structure (at least one period)
    if text.count('.') == 0:
        return False

    return True


def _dehyphenate_lines(lines: List[str]) -> List[str]:
    """
    Join words split across line breaks like 'algo-\nrithm' -> 'algorithm'.
    Only join when next line starts with lowercase/letter and current line ends with a word- hyphen.
    """
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        
        if cur.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1]
            # Safe join if next line begins with lowercase letter (or a digit continuation)
            
            if len(nxt) > 0 and re.match(r"^[a-z0-9]", nxt):
                # drop hyphen, join tightly
                cur = cur[:-1] + nxt.lstrip()
                out.append(cur)
                i += 2
                continue
            
        out.append(cur)
        
        i += 1
        
    return out


def _merge_wrapped_lines(lines: List[str]) -> str:
    """
    Reflow within a block: keep blank lines and list/heading boundaries;
    otherwise join lines with spaces (to undo layout wraps).
    """
    if not lines:
        return ""

    # pre de-hyphenate across line boundaries
    lines = _dehyphenate_lines(lines)

    paras: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf:
            return
        text = " ".join(s.strip() for s in buf if s.strip())
        text = normalize_text(text)
        if text:
            paras.append(text)

    for ln in lines:
        if not ln.strip():
            # blank line → paragraph boundary
            flush()
            buf = []
            continue

        if _bullet_or_list.match(ln):
            # list item → start new paragraph
            flush()
            buf = [ln]
            flush()
            buf = []
            continue

        # If it looks like a stand-alone heading line, keep as its own paragraph
        if _heading_like.match(ln.strip()):
            flush()
            paras.append(ln.strip())
            buf = []
            continue

        buf.append(ln)

    flush()
    return "\n\n".join(paras)


def _detect_repeating_headers_footers(doc: fitz.Document, max_lines: int = 2) -> Tuple[set, set]:
    """
    Heuristic: grab the first/last 2 non-empty lines of each page (using simple text extraction),
    count frequency, and treat lines that appear on >=60% of pages as header/footer.
    """
    header_counts = Counter()
    footer_counts = Counter()
    total_pages = len(doc)

    for i in range(total_pages):
        page = doc.load_page(i)
        raw = page.get_text("text").replace("\r\n", "\n").replace("\r", "\n")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        
        if not lines:
            continue

        for l in lines[:max_lines]:
            header_counts[l] += 1
        for l in lines[-max_lines:]:
            footer_counts[l] += 1

    thresh = max(2, int(0.6 * total_pages))
    headers = {l for l, c in header_counts.items() if c >= thresh}
    footers = {l for l, c in footer_counts.items() if c >= thresh}
    
    return headers, footers


def _strip_headers_footers(text: str, headers: set, footers: set) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip() not in headers and ln.strip() not in footers]
    return "\n".join(lines)

# ---------- main: page extraction with blocks & light layout handling ----------

def _page_blocks_to_text(page: fitz.Page) -> str:
    """
    Convert a page to clean, paragraph-like text using block extraction.
    - Sort by (column, y, x)
    - Light two-column handling by splitting blocks around page midline (with margin)
    - Reflow lines within each block
    """
    # get text blocks: (x0, y0, x1, y1, text, block_no, block_type, ...)
    blocks = page.get_text("blocks")
    if not blocks:
        return ""

    # keep only text blocks (block_type==0) when present
    cleaned = []
    for b in blocks:
        # Some PyMuPDF versions return 5 elements, some 8. We guard by length.
        x0, y0, x1, y1 = b[:4]
        txt = b[4] if len(b) > 4 else ""
        block_type = b[6] if len(b) > 6 else 0
        
        if block_type == 0 and txt.strip():
            cleaned.append((x0, y0, x1, y1, txt))

    if not cleaned:
        return ""

    page_w = page.rect.width
    mid = page_w / 2
    gutter = max(18.0, page_w * 0.03)  # small margin around the midline

    # Decide if it's likely multi-column: if many blocks sit clearly left/right of mid
    left_count = sum(1 for x0, *_ in cleaned if x0 < mid - gutter)
    right_count = sum(1 for x0, *_ in cleaned if x0 > mid + gutter)
    two_cols = left_count > 2 and right_count > 2  # crude but works in practice

    if two_cols:
        left = [(x0, y0, x1, y1, t) for (x0, y0, x1, y1, t) in cleaned if x0 <= mid]
        right = [(x0, y0, x1, y1, t) for (x0, y0, x1, y1, t) in cleaned if x0 > mid]
        
        # Sort reading order: left col top→bottom, then right col top→bottom
        left.sort(key=lambda r: (round(r[1], 1), round(r[0], 1)))
        right.sort(key=lambda r: (round(r[1], 1), round(r[0], 1)))
        ordered = left + right
    else:
        # Single column: sort top→bottom, then left→right
        ordered = sorted(cleaned, key=lambda r: (round(r[1], 1), round(r[0], 1)))

    paras: List[str] = []
    for x0, y0, x1, y1, txt in ordered:
        # Work line-wise within a block
        block_lines = txt.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        # remove lines that are pure page numbers (simple heuristic)
        block_lines = [ln for ln in block_lines if not re.fullmatch(r"\s*\d+\s*", ln)]
        block_text = _merge_wrapped_lines(block_lines)
        
        if block_text.strip():
            paras.append(block_text.strip())

    # Join with blank line between blocks
    page_text = "\n\n".join(paras)
    return normalize_text(page_text)


def extract_pages(pdf_path: Path) -> List[str]:
    """
    Extracts text per page with paragraph-ish structure,
    cleans headers/footers, and saves result as JSON.
    Output file: {cfg.paths.proc}/parsed.json
    """
    pages: List[str] = []

    with fitz.open(pdf_path, filetype="pdf") as doc:
        if doc.is_encrypted:
            return pages

        headers, footers = _detect_repeating_headers_footers(doc)

        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = _page_blocks_to_text(page)

            if txt:
                txt = _strip_headers_footers(txt, headers, footers)
                txt = normalize_text(txt)
            
            processed_text = anonymize_text(txt)

            # Apply enhanced RAG text cleaning
            processed_text = clean_rag_text(processed_text)

            # Skip text with figure references or that doesn't meet quality criteria
            if (not contains_figure_ref(processed_text) and
                is_text_suitable_for_rag(processed_text)):
                pages.append(processed_text)

    return pages


def extract_pdfs(directory_path: Path, max_pdfs: int | None = None):
    """
    Process all files in cfg.paths.raw with extract_pages
    and save into a single parsed.json inside cfg.paths.proc.

    Parameters
    ----------
    directory_path : Path
        Path to the folder containing PDFs.
    max_pdfs : int | None
        If set, stop after processing this many PDFs (for testing).
        If None, process all PDFs.

    Output
    ------
    parsed.json : dict
        {
            "filename1": [page_texts...],
            "filename2": [page_texts...]
        }
    """
    proc_dir = get_rag2_config_path("proc")
    proc_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}

    for idx, pdf_path in enumerate(directory_path.iterdir(), start=1):
        print(f"[INFO] Processing {pdf_path.name} ...")
        pages = extract_pages(pdf_path)

        # Skip PDFs that result in empty pages
        if not pages:
            print(f"[SKIP] {pdf_path.name} - no usable pages extracted")
            continue

        results[pdf_path.stem] = pages
        print(f"[OK] Parsed {len(pages)} pages from {pdf_path.name}")

        if max_pdfs is not None and idx >= max_pdfs:
            print(f"[INFO] Reached max_pdfs={max_pdfs}, stopping early.")
            break

    out_path = proc_dir / "parsed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(results)} documents → {out_path}")


if __name__ == "__main__":
    extract_pdfs(get_rag2_config_path("raw"))