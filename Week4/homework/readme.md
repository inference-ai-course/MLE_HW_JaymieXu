## RAG Retrieval Stack — Quick Start

Minimal steps to fetch papers, build the index, run search, and produce a retrieval report.

## Requirements
- Python **3.10**
- Install deps:
  ~~~bash
  pip install sentence-transformers faiss-cpu transformers fastapi uvicorn requests pymupdf feedparser numpy
  ~~~

## 1) Configure (optional)
Edit **`settings.py`** to adjust paths and knobs:
- `EMBED_MODEL`, `CHUNK_SIZE`, `OVERLAP_FRAC`, `MIN_SCORE`
- arXiv query is set inside **`fetch_arxiv.py`**

## 2) Build corpus & index
Runs: fetch → extract → chunk → embed → FAISS.
~~~bash
python fetch_arxiv.py
~~~
Outputs:
- `data/processed/chunks.jsonl`
- `data/index/faiss.index`, `data/index/chunk_meta.jsonl`, `data/index/manifest.json`

## 3) Run the search API
~~~bash
python search_api.py
~~~
Open docs: http://127.0.0.1:8000/docs → **POST `/search`**  
Example body:
~~~json
{ "query": "transformer attention mechanism", "k": 3, "min_score": 0.30 }
~~~

## 4) Generate the retrieval report (from a notebook)
~~~python
from make_retrieval_report import generate_report, display_report

queries = [
    "transformer attention mechanism",
    "beam search decoding",
    "contrastive learning for embeddings",
    "reinforcement learning for language models",
    "CTC loss",
]

p = generate_report(queries, out_path="reports/retrieval_report.md", k=3)
display_report(p)  # renders the .md in the notebook
~~~

## Notes
- PDFs are saved as `<arxiv_id>.pdf`; text is indexed as **256-token** chunks with **20% overlap**.
- The API returns **full chunk text** plus score/title/page.
- Raise/lower `MIN_SCORE` in `settings.py` to control relevance filtering.